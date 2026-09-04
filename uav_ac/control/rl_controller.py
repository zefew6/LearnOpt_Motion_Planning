"""Shared RL policy interface for training and trajectory-controller deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_ac.quadrotor.quad import Quad

from .trajectory_controller import ControlCommand, TrajectoryReference


OBSERVATION_SIZE = 59
ACTION_SIZE = 4
OBSERVATION_CLIP = 5.0
FUTURE_OFFSETS_SECONDS = np.array([0.05, 0.10, 0.15, 0.20])
POSITION_SCALE = 1.0
VELOCITY_SCALE = 2.0
ACCELERATION_SCALE = 5.0
BODY_RATE_SCALE = np.array([5.0, 5.0, 3.0])
RL_CONFIG_FILENAME = "rl_config.json"
BEST_MODEL_FILENAME = "best_model.zip"
RL_CONFIG_VERSION = 1


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to the half-open interval [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def heading_rotation(yaw: float) -> np.ndarray:
    """Return the heading-frame to NED rotation for a desired yaw angle."""
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def normalized_rotor_thrusts(quad: Quad) -> np.ndarray:
    """Map current per-rotor thrust to [-1, 1] with hover at zero."""
    hover = quad.m * quad.g / 4.0
    _validate_hover_authority(quad, hover)
    forces = quad.kf * np.square(quad.omega)
    below_hover = (forces - hover) / (hover - quad.min_thrust)
    above_hover = (forces - hover) / (quad.max_thrust - hover)
    return np.clip(np.where(forces <= hover, below_hover, above_hover), -1.0, 1.0)


def action_to_command(action: np.ndarray, quad: Quad) -> ControlCommand:
    """Map a normalized policy action to collective thrust and body moments.

    Collective action zero is exactly hover.  Moment authority is symmetric
    about hover and is based on the smaller upward/downward per-rotor margin;
    the quadrotor allocator remains the final feasibility guard for combined
    roll, pitch, and yaw requests.
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
        raise ValueError("RL action must contain four finite values")
    action = np.clip(action, -1.0, 1.0)

    hover_total = quad.m * quad.g
    hover_per_rotor = hover_total / 4.0
    _validate_hover_authority(quad, hover_per_rotor)
    if action[0] <= 0.0:
        thrust = hover_total + action[0] * (
            hover_total - 4.0 * quad.min_thrust)
    else:
        thrust = hover_total + action[0] * (
            4.0 * quad.max_thrust - hover_total)

    rotor_margin = min(
        hover_per_rotor - quad.min_thrust,
        quad.max_thrust - hover_per_rotor,
    )
    moment_scale = np.array([
        4.0 * quad.l * rotor_margin,
        4.0 * quad.l * rotor_margin,
        4.0 * quad.kappa * rotor_margin,
    ])
    return ControlCommand(thrust, action[1:] * moment_scale)


def encode_observation(
        quad: Quad,
        reference: TrajectoryReference,
        previous_action: np.ndarray,
) -> np.ndarray:
    """Encode the fixed, deployment-identical 59-dimensional policy input."""
    previous_action = np.asarray(previous_action, dtype=float)
    if previous_action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(previous_action)):
        raise ValueError("previous_action must contain four finite values")

    target = reference.current
    yaw = float(target[9])
    world_from_heading = heading_rotation(yaw)
    heading_from_world = world_from_heading.T
    position_error = heading_from_world @ (target[:3] - quad.position)
    velocity_error = heading_from_world @ (target[3:6] - quad.velocity)
    relative_rotation = heading_from_world @ quad.R()
    rotation_6d = np.concatenate((relative_rotation[:, 0], relative_rotation[:, 1]))
    reference_acceleration = heading_from_world @ target[6:9]

    denominator = max(len(reference.trajectory) - 1, 1)
    remaining_fraction = (len(reference.trajectory) - 1 - reference.index) / denominator

    pieces = [
        position_error / POSITION_SCALE,
        velocity_error / VELOCITY_SCALE,
        rotation_6d,
        quad.body_angular_velocity / BODY_RATE_SCALE,
        reference_acceleration / ACCELERATION_SCALE,
        normalized_rotor_thrusts(quad),
        np.clip(previous_action, -1.0, 1.0),
        np.array([remaining_fraction]),
    ]
    for offset_seconds in FUTURE_OFFSETS_SECONDS:
        offset = max(1, int(round(offset_seconds / reference.dt)))
        future_index = min(reference.index + offset, len(reference.trajectory) - 1)
        future = reference.trajectory[future_index]
        relative_position = heading_from_world @ (future[:3] - quad.position)
        relative_velocity = heading_from_world @ (future[3:6] - quad.velocity)
        yaw_difference = float(wrap_angle(future[9] - yaw))
        pieces.extend((
            relative_position / POSITION_SCALE,
            relative_velocity / VELOCITY_SCALE,
            np.array([np.sin(yaw_difference), np.cos(yaw_difference)]),
        ))

    observation = np.concatenate(pieces)
    if observation.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(f"RL observation has unexpected shape {observation.shape}")
    return np.clip(observation, -OBSERVATION_CLIP, OBSERVATION_CLIP).astype(np.float32)


def quad_parameters(quad: Quad) -> dict[str, Any]:
    """Return the physical parameters that define policy compatibility."""
    return {
        "gravity": float(quad.g),
        "physics_dt": float(quad.dt),
        "mass": float(quad.m),
        "inertia": [float(quad.i_x), float(quad.i_y), float(quad.i_z)],
        "arm_length": float(quad.l),
        "force_coefficient": float(quad.kf),
        "drag_to_thrust": float(quad.kappa),
        "thrust_limits": [float(quad.min_thrust), float(quad.max_thrust)],
        "motor_time_constants": [
            float(quad.motor_rise_time_constant),
            float(quad.motor_fall_time_constant),
        ],
    }


class RLController:
    """Run an SB3-compatible policy at control ticks and hold its command."""

    def __init__(
            self,
            policy: Any,
            quad: Quad,
            *,
            control_dt: float,
            steps_per_action: int,
    ):
        if control_dt <= 0.0 or not np.isfinite(control_dt):
            raise ValueError("control_dt must be positive and finite")
        if steps_per_action < 1:
            raise ValueError("steps_per_action must be positive")
        if not np.isclose(control_dt, quad.dt * steps_per_action):
            raise ValueError("control_dt must equal quad.dt * steps_per_action")
        self.policy = policy
        self.control_dt = float(control_dt)
        self.steps_per_action = int(steps_per_action)
        self._quad = quad
        self.reset()

    def reset(self) -> None:
        """Clear the held command, previous action, and optional policy state."""
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._command = action_to_command(self._previous_action, self._quad)
        self._policy_state = None
        self._episode_start = True

    def step(self, quad: Quad, reference: TrajectoryReference) -> ControlCommand:
        """Evaluate the policy at 100 Hz and hold the command at physics ticks."""
        if quad is not self._quad:
            raise ValueError("RLController must be used with the Quad it was configured for")
        if not np.isclose(reference.dt, self.control_dt):
            raise ValueError("trajectory reference dt does not match the RL control period")
        if not reference.is_control_tick:
            return self._command

        observation = encode_observation(quad, reference, self._previous_action)
        prediction = self.policy.predict(
            observation,
            state=self._policy_state,
            episode_start=np.array([self._episode_start]),
            deterministic=True,
        )
        if isinstance(prediction, tuple):
            action, self._policy_state = prediction
        else:
            action = prediction
            self._policy_state = None
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise ValueError("RL policy returned an invalid action")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._command = action_to_command(action, quad)
        self._previous_action = action
        self._episode_start = False
        return self._command

    @classmethod
    def from_run(
            cls,
            run_dir: str | Path,
            quad: Quad,
            device: str = "cpu",
    ) -> "RLController":
        """Load and validate the best SB3 PPO policy from a training run."""
        run_dir = Path(run_dir)
        config_path = run_dir / RL_CONFIG_FILENAME
        model_path = run_dir / BEST_MODEL_FILENAME
        if not config_path.is_file():
            raise FileNotFoundError(f"missing RL configuration: {config_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"missing best RL model: {model_path}")
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        _validate_run_config(config, quad)

        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise ImportError(
                "RL deployment requires the 'rl' extra: uv sync --extra rl"
            ) from error
        policy = PPO.load(model_path, device=device)
        if policy.observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError("loaded RL model has an incompatible observation space")
        if not (np.allclose(policy.observation_space.low, -OBSERVATION_CLIP)
                and np.allclose(policy.observation_space.high, OBSERVATION_CLIP)):
            raise ValueError("loaded RL model observation bounds are incompatible")
        if policy.action_space.shape != (ACTION_SIZE,):
            raise ValueError("loaded RL model has an incompatible action space")
        if not (np.allclose(policy.action_space.low, -1.0)
                and np.allclose(policy.action_space.high, 1.0)):
            raise ValueError("loaded RL model action bounds must be [-1, 1]")
        return cls(
            policy,
            quad,
            control_dt=float(config["control_dt"]),
            steps_per_action=int(config["steps_per_action"]),
        )


def _validate_hover_authority(quad: Quad, hover_per_rotor: float) -> None:
    if not quad.min_thrust < hover_per_rotor < quad.max_thrust:
        raise ValueError("quadrotor hover thrust must lie strictly within rotor limits")


def _validate_run_config(config: dict[str, Any], quad: Quad) -> None:
    required = {
        "schema_version", "observation_dim", "action_shape", "action_low",
        "action_high", "steps_per_action", "control_dt", "quad_parameters",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"RL configuration is missing: {', '.join(missing)}")
    if int(config["schema_version"]) != RL_CONFIG_VERSION:
        raise ValueError("unsupported RL configuration schema")
    if int(config["observation_dim"]) != OBSERVATION_SIZE:
        raise ValueError("RL configuration observation dimension is incompatible")
    if config["action_shape"] != [ACTION_SIZE]:
        raise ValueError("RL configuration action shape is incompatible")
    if not (np.allclose(config["action_low"], -np.ones(ACTION_SIZE))
            and np.allclose(config["action_high"], np.ones(ACTION_SIZE))):
        raise ValueError("RL configuration action bounds are incompatible")
    steps_per_action = int(config["steps_per_action"])
    control_dt = float(config["control_dt"])
    if steps_per_action < 1 or not np.isclose(control_dt, quad.dt * steps_per_action):
        raise ValueError("RL configuration control timing is incompatible")

    expected = quad_parameters(quad)
    actual = config["quad_parameters"]
    if set(actual) != set(expected):
        raise ValueError("RL configuration quadrotor parameters are incomplete")
    for name, expected_value in expected.items():
        if not np.allclose(actual[name], expected_value, rtol=1.0e-7, atol=1.0e-10):
            raise ValueError(f"RL configuration quadrotor parameter '{name}' is incompatible")


__all__ = [
    "ACTION_SIZE",
    "BEST_MODEL_FILENAME",
    "FUTURE_OFFSETS_SECONDS",
    "OBSERVATION_SIZE",
    "RL_CONFIG_FILENAME",
    "RL_CONFIG_VERSION",
    "RLController",
    "action_to_command",
    "encode_observation",
    "heading_rotation",
    "normalized_rotor_thrusts",
    "quad_parameters",
    "wrap_angle",
]
