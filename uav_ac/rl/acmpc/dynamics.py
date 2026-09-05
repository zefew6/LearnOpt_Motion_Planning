"""Batched NED/FRD rigid-body prediction with normalized wrench inputs."""
from __future__ import annotations

import torch
from torch import nn


class QuadrotorDynamics(nn.Module):
    def __init__(self, parameters: dict, dt: float, *, internal_controls=False):
        super().__init__()
        self.dt = float(dt)
        self.mass = float(parameters["mass"])
        self.gravity = float(parameters["gravity"])
        low, high = parameters["thrust_limits"]
        hover = self.mass * self.gravity
        if not 4 * low < hover < 4 * high:
            raise ValueError("hover must lie strictly inside thrust limits")
        self.hover = hover
        self.thrust_down = hover - 4 * low
        self.thrust_up = 4 * high - hover
        self.thrust_mid = 2 * (low + high)
        self.thrust_half = 2 * (high - low)
        self.internal_controls = internal_controls
        margin = min(hover / 4 - low, high - hover / 4)
        self.register_buffer("inertia", torch.tensor(parameters["inertia"], dtype=torch.float64))
        self.register_buffer("moment_scale", torch.tensor([
            4 * parameters["arm_length"] * margin,
            4 * parameters["arm_length"] * margin,
            4 * parameters["drag_to_thrust"] * margin,
        ], dtype=torch.float64))

    def wrench(self, action):
        if self.internal_controls:
            thrust = self.thrust_mid + self.thrust_half * action[..., 0]
        else:
            thrust = self.hover + action[..., 0] * torch.where(
                action[..., 0] <= 0, self.thrust_down, self.thrust_up)
        return torch.cat((thrust.unsqueeze(-1), action[..., 1:] * self.moment_scale), -1)

    def to_internal(self, action):
        thrust = self.hover + action[..., 0] * torch.where(action[..., 0] <= 0, self.thrust_down, self.thrust_up)
        return torch.cat((((thrust-self.thrust_mid)/self.thrust_half).unsqueeze(-1), action[..., 1:]), -1)

    def to_external(self, control):
        delta = self.thrust_mid + self.thrust_half * control[..., 0] - self.hover
        normalized = delta / torch.where(delta <= 0, self.thrust_down, self.thrust_up)
        return torch.cat((normalized.unsqueeze(-1), control[..., 1:]), -1)

    def derivative(self, state, action):
        q = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
        w, x, y, z = q.unbind(-1)
        omega = state[..., 10:13]
        p, r, s = omega.unbind(-1)
        qdot = 0.5 * torch.stack((
            -x*p-y*r-z*s, w*p+y*s-z*r, w*r+z*p-x*s, w*s+x*r-y*p), -1)
        b3 = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)
        wrench = self.wrench(action)
        gravity = torch.zeros_like(b3)
        gravity[..., 2] = self.gravity
        acceleration = gravity - wrench[..., :1] / self.mass * b3
        angular = (wrench[..., 1:] - torch.linalg.cross(omega, self.inertia * omega)) / self.inertia
        return torch.cat((state[..., 7:10], qdot, acceleration, angular), -1)

    def forward(self, state, action):
        h = self.dt
        k1 = self.derivative(state, action)
        k2 = self.derivative(state + h/2*k1, action)
        k3 = self.derivative(state + h/2*k2, action)
        k4 = self.derivative(state + h*k3, action)
        result = state + h/6*(k1 + 2*k2 + 2*k3 + k4)
        return torch.cat((result[..., :3], torch.nn.functional.normalize(
            result[..., 3:7], dim=-1), result[..., 7:]), -1)

    def grad_input(self, state, action):
        # Jacobians are local linearizations; the solver supplies its own VJP.
        with torch.enable_grad():
            a, b = torch.func.vmap(torch.func.jacrev(self.forward, argnums=(0, 1)))(state, action)
        return a.detach(), b.detach()
