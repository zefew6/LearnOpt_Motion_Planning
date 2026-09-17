"""Gate-conditioned quadratic MPC policy with no trajectory input."""
import math

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from uav_ac.tasks.gate_racing import FEATURE_SIZE, observation_space as racing_observation_space
from .policy import ACMPCPolicy
from .solver import DifferentiableMPC
from .vendor.mpc import QuadCost
from .racing_targets import RacingSettings, gate_targets, body_rotation

COST_VERSION = 2


class RacingMPC(DifferentiableMPC):
    def __init__(self, parameters, settings=None, racing_settings=None):
        super().__init__(parameters, settings)
        self.racing_settings = RacingSettings(**(racing_settings or {}))
        self.base_weights.copy_(torch.tensor([4., 4., 4.] + [6.]*4 + [2.]*3 + [.25]*3 + [.08, .8, .8, .4]))

    @property
    def cost_size(self):
        return (self.settings.horizon_steps+1)*17

    def build_racing_cost(self, state, features, outputs):
        n = self.settings.horizon_steps
        if outputs.shape != (len(state), self.cost_size):
            raise ValueError("incompatible racing cost output shape")
        local = torch.cat((torch.zeros_like(state[:, :3]), state[:, 3:]), -1)
        h = outputs.double().reshape(-1, n+1, 17)
        base = self.base_weights.expand(n+1, -1).clone()
        base[-1, :13] *= self.settings.terminal_scale
        diagonal = base * torch.exp(math.log(10) * torch.tanh(h))
        position, velocity, frame, clearance, speed = gate_targets(
            state, features, n, self.settings.dt, self.racing_settings)
        diagonal = diagonal.clone()
        diagonal[:, :, 1:3] = diagonal[:, :, 1:3] / clearance[:, None].square()
        anchor = torch.zeros_like(diagonal)
        anchor[:, :, :3] = position
        anchor[:, :, 7:10] = velocity
        q = state[:, 3:7]
        yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
        hover_q = torch.stack(((yaw/2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw/2).sin()), -1)
        hover_q = torch.where((hover_q*q).sum(-1, keepdim=True) < 0, -hover_q, hover_q)
        anchor[:, :, 3:7] = hover_q[:, None]
        hover_action = self.dynamics.to_internal(torch.zeros_like(state[:, :4]))
        anchor[:, :, 13:] = hover_action[:, None]
        quadratic = torch.diag_embed(diagonal)
        # Position weights are defined in the gate frame, not world XYZ.
        quadratic[:, :, :3, :3] = frame[:, None] @ torch.diag_embed(diagonal[:, :, :3]) @ frame[:, None].transpose(-1, -2)
        linear = -(quadratic @ anchor[..., None]).squeeze(-1)
        # Keep positive curvature on the solver's fixed-zero dummy input so
        # its box-QP factorization is nonsingular; it contributes zero cost.
        linear[:, -1, 13:] = 0
        self.last_target_speed = speed.detach()
        self.target_position, self.target_velocity = position.detach(), velocity.detach()
        return local, QuadCost(quadratic.transpose(0, 1), linear.transpose(0, 1))

    @torch.no_grad()
    def initial_controls(self, state):
        """Stateless stabilizing rollout; never hold a transient torque for 1 s."""
        x = state.clone()
        controls = []
        for k in range(self.settings.horizon_steps):
            acceleration = (2*(self.target_position[:, k+1]-x[:, :3])
                            + 3*(self.target_velocity[:, k+1]-x[:, 7:10])).clamp(-8, 8)
            force = -acceleration
            force[:, 2] += self.dynamics.gravity
            desired_b3 = torch.nn.functional.normalize(force, dim=-1)
            rotation = body_rotation(x[:, 3:7])
            b3 = rotation[:, :, 2]
            error = (rotation.transpose(-1, -2) @ torch.linalg.cross(b3, desired_b3)[..., None]).squeeze(-1)
            moment = self.dynamics.inertia * (40*error-10*x[:, 10:13])
            thrust = self.dynamics.mass*(force*b3).sum(-1)
            action = torch.cat((((thrust-self.dynamics.thrust_mid)/self.dynamics.thrust_half)[:, None],
                                moment/self.dynamics.moment_scale), -1).clamp(-1, 1)
            controls.append(action)
            x = self.dynamics(x, action)
        controls.append(torch.zeros_like(controls[-1]))
        return torch.stack(controls)

    def forward(self, state, previous_action, features, outputs, *, strict):
        if state.ndim != 2 or state.shape[1] != 13 or not bool(torch.isfinite(state).all()):
            raise ValueError("racing MPC requires a finite physical state of shape (batch, 13)")
        if features.shape != (len(state), FEATURE_SIZE) or not bool(torch.isfinite(features).all()):
            raise ValueError("racing MPC requires finite gate features")
        state, cost = self.build_racing_cost(state.double(), features, outputs)
        return self.solve_cost(state, cost, previous_action, strict=strict,
                               initial_controls=self.initial_controls(state))


class RacingACMPCPolicy(ACMPCPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *,
                 quad_parameters, mpc_settings=None, racing_settings=None, **kwargs):
        self.quad_parameters = quad_parameters
        self.mpc_settings = mpc_settings or {}
        self.racing_settings = racing_settings or {}
        if observation_space != racing_observation_space(True):
            raise ValueError("racing ACMPC requires gate features and physical state")
        kwargs.setdefault("log_std_init", -2.)
        kwargs["ortho_init"] = False
        ActorCriticPolicy.__init__(self, observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        if self.use_sde or self.action_space.shape != (4,):
            raise ValueError("racing ACMPC requires four Gaussian wrench actions")
        self.mpc = RacingMPC(self.quad_parameters, self.mpc_settings, self.racing_settings)
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
        mean = self.mpc(obs["mpc_state"], obs["previous_action"], features, self.cost_net(features),
                        strict=self.training or self.solver_strict)
        return self.action_dist.proba_distribution(mean, self.log_std)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["racing_settings"] = self.racing_settings
        return data
