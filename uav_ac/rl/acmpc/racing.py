"""Gate-conditioned quadratic MPC policy with no trajectory input."""
import math

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from uav_ac.tasks.gate_racing import FEATURE_SIZE, observation_space as racing_observation_space
from .policy import ACMPCPolicy
from .solver import DifferentiableMPC
from .vendor.mpc import QuadCost


class RacingMPC(DifferentiableMPC):
    def __init__(self, parameters, settings=None):
        super().__init__(parameters, settings)
        self.scales.copy_(torch.tensor([10.]*3 + [1.]*4 + [10.]*3 + [10.]*3 + [1.]*4))
        self.base_weights.copy_(torch.tensor([1.]*3 + [6.]*4 + [1.]*3 + [.25]*3 + [.08, .8, .8, .4]))

    def build_racing_cost(self, state, outputs):
        n = self.settings.horizon_steps
        if outputs.shape != (len(state), self.cost_size):
            raise ValueError("incompatible racing cost output shape")
        local = torch.cat((torch.zeros_like(state[:, :3]), state[:, 3:]), -1)
        h, b = outputs.double().chunk(2, -1)
        pad = torch.zeros_like(h[:, :4])
        h = torch.cat((h, pad), -1).reshape(-1, n+1, 17)
        b = torch.cat((b, pad), -1).reshape(-1, n+1, 17)
        base = self.base_weights.expand(n+1, -1).clone()
        base[-1, :13] *= self.settings.terminal_scale
        diagonal = base * torch.exp(math.log(10) * torch.tanh(h)) / self.scales.square()
        # Zero network output stabilizes level hover at the current position.
        # This is a local initialization, not a gate reference or planned path.
        anchor = torch.zeros_like(diagonal)
        q = state[:, 3:7]
        yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
        hover_q = torch.stack(((yaw/2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw/2).sin()), -1)
        hover_q = torch.where((hover_q*q).sum(-1, keepdim=True) < 0, -hover_q, hover_q)
        anchor[:, :, 3:7] = hover_q[:, None]
        hover_action = self.dynamics.to_internal(torch.zeros_like(state[:, :4]))
        anchor[:, :, 13:] = hover_action[:, None]
        linear = 2 * base * torch.tanh(b) / self.scales - diagonal * anchor
        linear = linear.clone()
        # Keep positive curvature on the solver's fixed-zero dummy input so
        # its box-QP factorization is nonsingular; it contributes zero cost.
        linear[:, -1, 13:] = 0
        return local, QuadCost(torch.diag_embed(diagonal).transpose(0, 1), linear.transpose(0, 1))

    def forward(self, state, previous_action, outputs, *, strict):
        if state.ndim != 2 or state.shape[1] != 13 or not bool(torch.isfinite(state).all()):
            raise ValueError("racing MPC requires a finite physical state of shape (batch, 13)")
        state, cost = self.build_racing_cost(state.double(), outputs)
        return self.solve_cost(state, cost, previous_action, strict=strict)


class RacingACMPCPolicy(ACMPCPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *,
                 quad_parameters, mpc_settings=None, **kwargs):
        self.quad_parameters = quad_parameters
        self.mpc_settings = mpc_settings or {}
        if observation_space != racing_observation_space(True):
            raise ValueError("racing ACMPC requires gate features and physical state")
        kwargs.setdefault("log_std_init", -2.)
        kwargs["ortho_init"] = False
        ActorCriticPolicy.__init__(self, observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        if self.use_sde or self.action_space.shape != (4,):
            raise ValueError("racing ACMPC requires four Gaussian wrench actions")
        self.mpc = RacingMPC(self.quad_parameters, self.mpc_settings)
        architecture = self.net_arch or {"pi": [512, 512], "vf": [512, 512]}
        def network(widths, output):
            layers, previous = [], FEATURE_SIZE
            for width in widths:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            layers.append(nn.Linear(previous, output))
            return nn.Sequential(*layers)
        self.cost_net = network(architecture["pi"], self.mpc.cost_size)
        nn.init.zeros_(self.cost_net[-1].weight)
        nn.init.zeros_(self.cost_net[-1].bias)
        self.value_net = network(architecture["vf"], 1)
        self.log_std = nn.Parameter(torch.full((4,), self.log_std_init))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.solver_strict = False

    def get_distribution(self, obs):
        features = obs["features"].float()
        mean = self.mpc(obs["mpc_state"], obs["previous_action"], self.cost_net(features),
                        strict=self.training or self.solver_strict)
        return self.action_dist.proba_distribution(mean, self.log_std)
