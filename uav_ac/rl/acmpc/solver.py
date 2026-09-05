"""Tracking cost and deterministic adapter around the fixed-point iLQR layer."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time

import torch
from torch import nn

from .dynamics import QuadrotorDynamics
from .vendor.mpc import MPC, QuadCost, GradMethods

SOLVER_VERSION = "63732fa85ab2a151045493c4e67653210ca3d7ff-local1"


@dataclass(frozen=True)
class MPCSettings:
    horizon_steps: int = 20
    dt: float = 0.01
    iterations: int = 1
    retry_iterations: int = 1
    tolerance: float = 1e-4
    chunk_size: int = 32
    terminal_scale: float = 4.0

    def __post_init__(self):
        if self.horizon_steps < 2 or self.dt <= 0 or not math.isfinite(self.dt):
            raise ValueError("invalid MPC horizon or dt")
        if self.iterations < 1 or self.retry_iterations < self.iterations or self.chunk_size < 1:
            raise ValueError("invalid MPC iteration or chunk settings")
        if self.tolerance <= 0 or not math.isfinite(self.tolerance) or self.terminal_scale <= 0 or not math.isfinite(self.terminal_scale):
            raise ValueError("invalid MPC tolerance or terminal scale")


def reference_targets(ref, initial_q, dynamics):
    """Flat acceleration/yaw to sign-aligned quaternion, body rate and thrust."""
    force = -ref[..., 6:9].clone()
    force[..., 2] += dynamics.gravity
    b3 = torch.nn.functional.normalize(force, dim=-1)
    yaw = ref[..., 9]
    heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
    b2 = torch.nn.functional.normalize(torch.linalg.cross(b3, heading), dim=-1)
    b1 = torch.linalg.cross(b2, b3)
    rotation = torch.stack((b1, b2, b3), -1)
    # Stable matrix-to-quaternion conversion by choosing the largest component.
    r = rotation
    candidates = torch.stack((1+r[...,0,0]+r[...,1,1]+r[...,2,2],
        1+r[...,0,0]-r[...,1,1]-r[...,2,2], 1-r[...,0,0]+r[...,1,1]-r[...,2,2],
        1-r[...,0,0]-r[...,1,1]+r[...,2,2]), -1).clamp_min(1e-16).sqrt()
    rows = torch.stack((
        torch.stack((candidates[...,0]**2, r[...,2,1]-r[...,1,2], r[...,0,2]-r[...,2,0], r[...,1,0]-r[...,0,1]), -1),
        torch.stack((r[...,2,1]-r[...,1,2], candidates[...,1]**2, r[...,1,0]+r[...,0,1], r[...,0,2]+r[...,2,0]), -1),
        torch.stack((r[...,0,2]-r[...,2,0], r[...,1,0]+r[...,0,1], candidates[...,2]**2, r[...,2,1]+r[...,1,2]), -1),
        torch.stack((r[...,1,0]-r[...,0,1], r[...,2,0]+r[...,0,2], r[...,2,1]+r[...,1,2], candidates[...,3]**2), -1),
    ), -2) / (2*candidates[..., :, None])
    idx = candidates.argmax(-1)
    q = rows.gather(-2, idx[..., None, None].expand(*idx.shape, 1, 4)).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    aligned = []
    previous = initial_q
    for k in range(q.shape[1]):
        current = torch.where((q[:, k]*previous).sum(-1, keepdim=True) < 0, -q[:, k], q[:, k])
        aligned.append(current)
        previous = current
    q = torch.stack(aligned, 1)
    delta = r[:, :-1].transpose(-1, -2) @ r[:, 1:]
    vee = torch.stack((delta[...,2,1]-delta[...,1,2], delta[...,0,2]-delta[...,2,0], delta[...,1,0]-delta[...,0,1]), -1)
    cosine = ((delta.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    angle = cosine.acos()
    ratio = torch.where(angle.abs() < 1e-7, torch.full_like(angle, 0.5), angle/(2*angle.sin().clamp_min(1e-8)))
    rates = vee * ratio[..., None] / dynamics.dt
    rates = torch.cat((rates, rates[:, -1:]), 1)
    thrust_delta = dynamics.mass * force.norm(dim=-1) - dynamics.hover
    thrust = (thrust_delta / torch.where(thrust_delta <= 0, dynamics.thrust_down, dynamics.thrust_up)).clamp(-1, 1)
    actions = torch.cat((thrust[..., None], torch.zeros_like(ref[..., :3])), -1)
    if dynamics.internal_controls:
        actions = dynamics.to_internal(actions)
    return torch.cat((ref[..., :3], q, ref[..., 3:6], rates, actions), -1)


class DifferentiableMPC(nn.Module):
    def __init__(self, parameters, settings=None):
        super().__init__()
        self.settings = MPCSettings(**(settings or {}))
        self.dynamics = QuadrotorDynamics(parameters, self.settings.dt, internal_controls=True)
        self.register_buffer("scales", torch.tensor([1]*7 + [2]*3 + [5,5,3] + [1]*4, dtype=torch.float64))
        self.register_buffer("base_weights", torch.tensor([40,40,60] + [6]*4 + [6,6,8] + [.25,.25,.15] + [.08,.8,.8,.4], dtype=torch.float64))
        self.last_diagnostics = {}

    @property
    def cost_size(self):
        return 2 * (17*self.settings.horizon_steps + 13)

    def build_cost(self, state, ref, residual):
        n = self.settings.horizon_steps
        target = reference_targets(ref, state[:, 3:7], self.dynamics)
        # A local origin reduces cancellation in large-world trajectories.
        origin = ref[:, :1, :3]
        state = torch.cat((state[:, :3]-origin[:, 0], state[:, 3:]), -1)
        target = torch.cat((target[..., :3]-origin, target[..., 3:]), -1)
        h, b = residual.double().chunk(2, -1)
        pad = torch.zeros_like(h[:, :4])
        h = torch.cat((h, pad), -1).reshape(-1, n+1, 17)
        b = torch.cat((b, pad), -1).reshape(-1, n+1, 17)
        base = self.base_weights.expand(n+1, -1).clone()
        base[-1, :13] *= self.settings.terminal_scale
        weights = base * torch.exp(math.log(10) * torch.tanh(h))
        linear = .5 * base * torch.tanh(b)
        diagonal = weights / self.scales.square()
        c = linear / self.scales - diagonal * target
        c = torch.cat((c[:, :-1], torch.cat((c[:, -1:, :13], torch.zeros_like(c[:, -1:, 13:])), -1)), 1)
        return state, QuadCost(torch.diag_embed(diagonal).transpose(0, 1), c.transpose(0, 1))

    def forward(self, state, ref, previous_action, residual, *, strict):
        started = time.perf_counter()
        outer_grad = torch.is_grad_enabled()
        cfg = self.settings
        if state.ndim != 2 or state.shape[1] != 13 or ref.shape != (len(state),cfg.horizon_steps+1,10):
            raise ValueError("incompatible MPC state or reference shape")
        if not all(bool(torch.isfinite(t).all()) for t in (state, ref, previous_action, residual)):
            raise ValueError("MPC inputs must be finite")
        if bool((state[:,3:7].norm(dim=-1) < 1e-10).any()):
            raise ValueError("MPC requires a nonzero quaternion")
        means, residuals = [], []
        failures = retries = 0
        sample_failures, sample_retries = [], []
        for start in range(0, len(state), cfg.chunk_size):
            end = start + cfg.chunk_size
            x0, cost = self.build_cost(state[start:end].double(), ref[start:end].double(), residual[start:end])
            previous = previous_action[start:end].double().clamp(-1, 1)
            batch = len(x0)
            lower = torch.full((cfg.horizon_steps+1, batch, 4), -1., device=x0.device, dtype=x0.dtype)
            upper = -lower
            lower[-1] = 0
            upper[-1] = 0  # terminal dummy control never applied to dynamics
            initial = self.dynamics.to_internal(previous).unsqueeze(0).expand(cfg.horizon_steps+1, -1, -1).clone()
            initial[-1] = 0
            retried = torch.zeros(batch, dtype=torch.bool, device=x0.device)
            for iterations in (cfg.iterations, cfg.retry_iterations):
                solver = MPC(13, 4, cfg.horizon_steps+1, u_lower=lower, u_upper=upper,
                    u_init=initial, n_batch=batch, lqr_iter=iterations, eps=cfg.tolerance,
                    back_eps=cfg.tolerance, grad_method=GradMethods.ANALYTIC,
                    exit_unconverged=False,
                    detach_unconverged=not (cfg.iterations == cfg.retry_iterations == 1),
                    backprop=outer_grad,
                    verbose=-1, best_cost_eps=1e-10)
                with torch.enable_grad():
                    predicted, actions, _ = solver(x0, cost, self.dynamics)
                finite = torch.isfinite(actions).all(dim=(0, 2)) & torch.isfinite(predicted).all(dim=(0, 2))
                # The paper uses one iLQR update as the differentiable actor.
                # In that mode, a fixed-point residual is expected and is not
                # a solver failure; multi-iteration diagnostic runs retain the
                # stricter convergence criterion.
                valid = finite if cfg.iterations == cfg.retry_iterations == 1 else (
                    (solver.last_residual < cfg.tolerance) & finite)
                if bool(valid.all()) or iterations == cfg.retry_iterations:
                    break
                retries += int((~valid).sum())
                retried = ~valid
            failures += int((~valid).sum())
            residuals.extend(solver.last_residual.cpu().tolist())
            sample_failures.extend((~valid).int().cpu().tolist())
            sample_retries.extend(retried.int().cpu().tolist())
            if strict and not bool(valid.all()):
                self.last_diagnostics = {"failures": failures, "residuals": residuals, "retries": retries,
                    "states": state[start:end].detach().cpu().tolist(), "previous_actions": previous.detach().cpu().tolist()}
                raise RuntimeError(f"ACMPC did not converge: {self.last_diagnostics}")
            mean = torch.where(valid[:, None], self.dynamics.to_external(actions[0]), previous.detach())
            means.append(mean if outer_grad else mean.detach())
        self.last_diagnostics = {"failures": failures, "retries": retries,
            "max_residual": max(residuals, default=0.), "seconds": time.perf_counter()-started,
            "samples": len(state), "sample_failures": sample_failures, "sample_retries": sample_retries}
        return torch.cat(means).float()
