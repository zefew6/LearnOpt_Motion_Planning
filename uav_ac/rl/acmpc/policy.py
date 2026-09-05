"""Stock SB3 policy whose Gaussian mean is a differentiable MPC solution."""
from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from .solver import DifferentiableMPC
from .observation import observation_space as mpc_observation_space


class ACMPCPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *,
                 quad_parameters, mpc_settings=None, **kwargs):
        self.quad_parameters = quad_parameters
        self.mpc_settings = mpc_settings or {}
        if observation_space != mpc_observation_space(self.mpc_settings.get("horizon_steps",20)):
            raise ValueError("ACMPC requires physical state, reference and feature observations")
        kwargs.setdefault("log_std_init", -2.)
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        if self.use_sde or self.action_space.shape != (4,):
            raise ValueError("ACMPC requires a four-dimensional diagonal Gaussian policy")
        self.mpc = DifferentiableMPC(self.quad_parameters, self.mpc_settings)
        architecture = self.net_arch or {"pi": [512, 512], "vf": [512, 512]}
        def network(widths, output):
            layers = []
            previous = 59
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
        mean = self.mpc(obs["mpc_state"], obs["reference"], features[:, 22:26],
                        self.cost_net(features), strict=self.training or self.solver_strict)
        return self.action_dist.proba_distribution(mean, self.log_std)

    def predict_values(self, obs):
        return self.value_net(obs["features"].float())

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(quad_parameters=self.quad_parameters, mpc_settings=self.mpc_settings)
        return data
