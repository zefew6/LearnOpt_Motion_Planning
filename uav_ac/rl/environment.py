"""Gymnasium environment for direct-wrench MuJoCo trajectory tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from uav_ac.control.rl_controller import (
    ACTION_SIZE,
    OBSERVATION_CLIP,
    OBSERVATION_SIZE,
    action_to_command,
    encode_observation,
    heading_rotation,
    wrap_angle,
)
from uav_ac.control.trajectory_controller import TrajectoryReference
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation
from uav_ac.simulation.mujoco_sim import ENU_TO_NED
from uav_ac.simulation.wind_disturb import (
    GustingCrosswind,
    RandomWindConfig,
    sample_gusting_crosswind,
)

from .assets import InitializationLibrary, ideal_initialization_library
from .trajectory_bank import TrajectoryBank


MAX_POSITION_ERROR = 2.5
MAX_TILT_RADIANS = np.deg2rad(75.0)
SUCCESS_POSITION_ERROR = 0.5
SUCCESS_VELOCITY_ERROR = 0.5
SUCCESS_TILT_RADIANS = np.deg2rad(15.0)
SUCCESS_YAW_ERROR = np.deg2rad(20.0)
TERMINAL_HOLD_SECONDS = 1.0
RANDOM_START_PROBABILITY = 0.75
MINIMUM_REMAINING_SECONDS = 2.0


class MujocoTrajectoryTrackingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Track one controller-ready trajectory with normalized wrench actions."""

    metadata = {"render_modes": []}

    def __init__(
            self,
            trajectory: np.ndarray | TrajectoryBank,
            initialization_library: InitializationLibrary | None = None,
            *,
            model_path: str | Path = DEFAULT_SCENE_PATH,
            steps_per_action: int = 10,
            random_start: bool = True,
            perturb_initial_state: bool = True,
            curriculum_progress: float = 0.0,
            split: str = "train",
            wind_config: RandomWindConfig | None = None,
    ):
        super().__init__()
        if steps_per_action < 1:
            raise ValueError("steps_per_action must be positive")

        self.simulation = MujocoSimulation(
            model_path, record_actual_trajectory=False)
        self.quad = self.simulation.quad
        self.steps_per_action = int(steps_per_action)
        self.control_dt = self.quad.dt * self.steps_per_action
        self.random_start = bool(random_start)
        self.perturb_initial_state = bool(perturb_initial_state)
        self.wind_config = wind_config
        self.trajectory_bank = trajectory if isinstance(trajectory, TrajectoryBank) else None
        self._requested_split = split
        if self.trajectory_bank is None:
            trajectory_array = np.asarray(trajectory, dtype=float)
            self._trajectory_indices = np.array([0], dtype=np.int64)
            self._single_initialization_library = initialization_library
        else:
            if initialization_library is not None:
                raise ValueError("initialization_library is stored inside a TrajectoryBank")
            trajectory_array = self.trajectory_bank.trajectory(
                int(self.trajectory_bank.indices(split)[0]))
            self._trajectory_indices = self.trajectory_bank.indices(split)
            self._single_initialization_library = None
        self._validate_trajectory(trajectory_array)
        self.trajectory = trajectory_array
        self._trajectory_id = int(self._trajectory_indices[0])
        self._trajectory_metadata: dict[str, Any] = {}
        self._activate_trajectory(self._trajectory_id)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-OBSERVATION_CLIP,
            high=OBSERVATION_CLIP,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._curriculum_progress = 0.0
        self.set_training_progress(curriculum_progress)
        self._reference_index = 0
        self._start_index = 0
        self._terminal_hold_steps = 0
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._episode_position_squared_error = 0.0
        self._episode_metric_steps = 0
        self._wind: GustingCrosswind | None = None
        self._wind_enabled = False
        self._current_wind_force_ned = np.zeros(3)
        self._maximum_wind_force = 0.0

    def set_training_progress(self, progress: float) -> None:
        """Set completed-training fraction used by the 30% perturbation ramp."""
        if not np.isfinite(progress):
            raise ValueError("training progress must be finite")
        self._curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    @property
    def perturbation_scale(self) -> float:
        return min(self._curriculum_progress / 0.30, 1.0)

    @property
    def wind_scale(self) -> float:
        if self.wind_config is None:
            return 0.0
        fraction = self.wind_config.curriculum_fraction
        return 1.0 if fraction == 0.0 else min(self._curriculum_progress / fraction, 1.0)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else options
        self._activate_trajectory(self._select_trajectory_id(options))
        start_index = self._select_start_index(options)
        perturbation_scale = float(options.get(
            "perturbation_scale",
            self.perturbation_scale if self.perturb_initial_state else 0.0,
        ))
        if not 0.0 <= perturbation_scale <= 1.0:
            raise ValueError("perturbation_scale option must be in [0, 1]")

        base_state = self.initialization_library.states[start_index]
        motor_speeds = self.initialization_library.motor_speeds[start_index]
        reset_succeeded = False
        for _ in range(32):
            state = self._perturb_state(base_state, perturbation_scale)
            self.simulation.reset(state, motor_speeds)
            if not self.simulation.collision_detected and self._is_in_bounds():
                reset_succeeded = True
                break
        if not reset_succeeded:
            self.simulation.reset(base_state, motor_speeds)
            if self.simulation.collision_detected or not self._is_in_bounds():
                raise RuntimeError(f"initialization snapshot {start_index} is not collision-free")

        self._reference_index = start_index
        self._start_index = start_index
        self._terminal_hold_steps = 0
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._episode_position_squared_error = 0.0
        self._episode_metric_steps = 0
        self._reset_wind(options)
        observation = encode_observation(
            self.quad, self._reference(), self._previous_action)
        return observation, self._info(self._metrics(), success=False)

    def step(
            self,
            action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain four finite values")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        previous_action = self._previous_action.copy()
        command = action_to_command(action, self.quad)

        finite_state = True
        for _ in range(self.steps_per_action):
            self._apply_wind()
            self.quad.set_propeller_speed(command.thrust, command.moment)
            self.simulation.step()
            finite_state = self._state_is_valid()
            if not finite_state or self.simulation.collision_detected:
                break

        if self._reference_index < len(self.trajectory) - 1:
            self._reference_index += 1
        else:
            self._terminal_hold_steps += 1
        metrics = self._metrics() if finite_state else self._nonfinite_metrics()
        if np.isfinite(metrics["position_error"]):
            self._episode_position_squared_error += metrics["position_error"] ** 2
        self._episode_metric_steps += 1
        failure_reason = self._failure_reason(metrics, finite_state)
        success = failure_reason is None and self._is_success(metrics)
        terminated = failure_reason is not None or success
        maximum_hold_steps = int(round(TERMINAL_HOLD_SECONDS / self.control_dt))
        truncated = bool(
            not terminated
            and self._reference_index == len(self.trajectory) - 1
            and self._terminal_hold_steps >= maximum_hold_steps
        )

        action_cost = 0.005 * float(np.dot(action, action))
        action_delta = action - previous_action
        action_delta_cost = 0.01 * float(np.dot(action_delta, action_delta))
        tracking_reward = float(np.exp(-metrics["tracking_cost"]))
        reward = tracking_reward - action_cost - action_delta_cost
        if success:
            reward += 50.0
        elif failure_reason is not None:
            reward -= 50.0

        self._previous_action = action
        observation = (
            encode_observation(self.quad, self._reference(), self._previous_action)
            if finite_state else np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        )
        info = self._info(
            metrics,
            success=success,
            action_cost=action_cost,
            action_delta_cost=action_delta_cost,
            tracking_reward=tracking_reward,
            failure_reason=failure_reason,
            position_rmse=(
                float(np.sqrt(self._episode_position_squared_error / self._episode_metric_steps))
                if self._episode_metric_steps else float("inf")
            ),
        )
        return observation, float(reward), bool(terminated), truncated, info

    @staticmethod
    def _validate_trajectory(trajectory: np.ndarray) -> None:
        if (trajectory.ndim != 2 or trajectory.shape[1] < 10
                or len(trajectory) == 0 or not np.all(np.isfinite(trajectory))):
            raise ValueError("trajectory must have finite shape (n, m), n >= 1 and m >= 10")

    def _activate_trajectory(self, trajectory_id: int) -> None:
        if self.trajectory_bank is None:
            if trajectory_id != 0:
                raise ValueError("single-trajectory environment only supports trajectory_id 0")
            trajectory = getattr(self, "trajectory", None)
            if trajectory is None:
                raise RuntimeError("single trajectory was not initialized")
            library = self._single_initialization_library
            metadata = {"trajectory_id": 0, "split": "legacy"}
        else:
            if trajectory_id not in self._trajectory_indices:
                raise ValueError(
                    f"trajectory_id {trajectory_id} does not belong to split '{self._requested_split}'")
            trajectory = self.trajectory_bank.trajectory(trajectory_id)
            library = self.trajectory_bank.initialization_library(trajectory_id)
            metadata = self.trajectory_bank.entry(trajectory_id)
        self._validate_trajectory(np.asarray(trajectory))
        self.trajectory = np.asarray(trajectory)
        self.initialization_library = (
            ideal_initialization_library(self.trajectory, self.quad)
            if library is None else library
        )
        if len(self.initialization_library.states) != len(self.trajectory):
            raise ValueError("initialization library must contain one snapshot per trajectory row")
        self._trajectory_id = int(trajectory_id)
        self._trajectory_metadata = metadata

    def _select_trajectory_id(self, options: dict[str, Any]) -> int:
        if "trajectory_id" in options:
            return int(options["trajectory_id"])
        if len(self._trajectory_indices) == 1:
            return int(self._trajectory_indices[0])
        return int(self.np_random.choice(self._trajectory_indices))

    def _reset_wind(self, options: dict[str, Any]) -> None:
        self.simulation.set_external_force_world(np.zeros(3))
        self._current_wind_force_ned = np.zeros(3)
        self._maximum_wind_force = 0.0
        if self.wind_config is None:
            self._wind_enabled = False
            self._wind = None
            return
        enabled = bool(options.get(
            "wind_enabled",
            self.np_random.random() < self.wind_config.probability,
        ))
        scale = float(options.get("wind_scale", self.wind_scale))
        if not 0.0 <= scale <= 1.0:
            raise ValueError("wind_scale option must be in [0, 1]")
        self._wind_enabled = enabled
        self._wind = (
            sample_gusting_crosswind(self.np_random, self.wind_config, scale=scale)
            if enabled else None
        )

    def _apply_wind(self) -> None:
        self._current_wind_force_ned = (
            np.zeros(3) if self._wind is None
            else self._wind.force_ned(float(self.simulation.data.time))
        )
        self._maximum_wind_force = max(
            self._maximum_wind_force,
            float(np.linalg.norm(self._current_wind_force_ned)),
        )
        self.simulation.set_external_force_world(
            ENU_TO_NED @ self._current_wind_force_ned)

    def _select_start_index(self, options: dict[str, Any]) -> int:
        if "start_index" in options:
            index = int(options["start_index"])
            if not 0 <= index < len(self.trajectory):
                raise ValueError("start_index option lies outside the trajectory")
            return index
        use_random_start = bool(options.get("random_start", self.random_start))
        if not use_random_start or self.np_random.random() >= RANDOM_START_PROBABILITY:
            return 0
        remaining_samples = int(np.ceil(MINIMUM_REMAINING_SECONDS / self.control_dt))
        last_start = max(0, len(self.trajectory) - 1 - remaining_samples)
        if last_start < 1:
            return 0
        return int(self.np_random.integers(1, last_start + 1))

    def _perturb_state(self, base_state: np.ndarray, scale: float) -> np.ndarray:
        state = base_state.copy()
        if scale == 0.0:
            return state
        state[:3] += self.np_random.uniform(-0.25, 0.25, size=3) * scale
        state[7:10] += self.np_random.uniform(-0.25, 0.25, size=3) * scale
        angle_limit = np.deg2rad(5.0) * scale
        angles = self.np_random.uniform(-angle_limit, angle_limit, size=3)
        state[3:7] = _quaternion_multiply(state[3:7], _euler_quaternion(angles))
        state[3:7] /= np.linalg.norm(state[3:7])
        state[10:13] += self.np_random.uniform(-0.2, 0.2, size=3) * scale
        return state

    def _reference(self) -> TrajectoryReference:
        return TrajectoryReference(
            trajectory=self.trajectory,
            index=self._reference_index,
            dt=self.control_dt,
            is_terminal=self._reference_index == len(self.trajectory) - 1,
            is_control_tick=True,
        )

    def _metrics(self) -> dict[str, float]:
        target = self.trajectory[self._reference_index]
        position_error = float(np.linalg.norm(target[:3] - self.quad.position))
        velocity_error = float(np.linalg.norm(target[3:6] - self.quad.velocity))
        desired_rotation = heading_rotation(float(target[9]))
        relative_rotation = desired_rotation.T @ self.quad.R()
        attitude_error = float(np.arccos(np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)))
        tilt = float(np.arccos(np.clip(self.quad.R()[2, 2], -1.0, 1.0)))
        yaw_error = abs(float(wrap_angle(self.quad.psi - target[9])))
        rates_scaled = self.quad.body_angular_velocity / np.array([5.0, 5.0, 3.0])
        tracking_cost = (
            position_error ** 2
            + 0.25 * (velocity_error / 2.0) ** 2
            + 0.25 * (attitude_error / 0.5) ** 2
            + 0.05 * float(np.dot(rates_scaled, rates_scaled))
        )
        return {
            "position_error": position_error,
            "velocity_error": velocity_error,
            "attitude_error": attitude_error,
            "tilt": tilt,
            "yaw_error": yaw_error,
            "tracking_cost": float(tracking_cost),
        }

    def _state_is_valid(self) -> bool:
        """Check finite state and a usable free-joint quaternion."""
        return bool(
            np.all(np.isfinite(self.quad.X))
            and np.linalg.norm(self.quad.quaternion) > 1.0e-12
        )

    @staticmethod
    def _nonfinite_metrics() -> dict[str, float]:
        return {
            "position_error": float("inf"),
            "velocity_error": float("inf"),
            "attitude_error": float("inf"),
            "tilt": float("inf"),
            "yaw_error": float("inf"),
            "tracking_cost": float("inf"),
        }

    def _failure_reason(
            self,
            metrics: dict[str, float],
            finite_state: bool,
    ) -> str | None:
        if not finite_state:
            return "non_finite_state"
        if self.simulation.collision_detected:
            return "collision"
        if not self._is_in_bounds():
            return "out_of_bounds"
        if metrics["position_error"] > MAX_POSITION_ERROR:
            return "position_error"
        if metrics["tilt"] > MAX_TILT_RADIANS:
            return "tilt"
        return None

    def _is_success(self, metrics: dict[str, float]) -> bool:
        return bool(
            self._reference_index == len(self.trajectory) - 1
            and metrics["position_error"] < SUCCESS_POSITION_ERROR
            and metrics["velocity_error"] < SUCCESS_VELOCITY_ERROR
            and metrics["tilt"] < SUCCESS_TILT_RADIANS
            and metrics["yaw_error"] < SUCCESS_YAW_ERROR
        )

    def _is_in_bounds(self) -> bool:
        return bool(np.all(self.quad.position >= self.simulation.space_limits[0])
                    and np.all(self.quad.position <= self.simulation.space_limits[1]))

    def _info(
            self,
            metrics: dict[str, float],
            *,
            success: bool,
            action_cost: float = 0.0,
            action_delta_cost: float = 0.0,
            tracking_reward: float = 0.0,
            failure_reason: str | None = None,
            position_rmse: float = 0.0,
    ) -> dict[str, Any]:
        return {
            **metrics,
            "action_cost": float(action_cost),
            "action_delta_cost": float(action_delta_cost),
            "tracking_reward": float(tracking_reward),
            "position_rmse": float(position_rmse),
            "collision": bool(self.simulation.collision_detected),
            "success": bool(success),
            "failure_reason": failure_reason,
            "start_index": int(self._start_index),
            "reference_index": int(self._reference_index),
            "trajectory_id": int(self._trajectory_id),
            "trajectory_split": str(self._trajectory_metadata.get("split", "legacy")),
            "navigation_waypoint_count": int(
                self._trajectory_metadata.get("navigation_waypoint_count", 0)),
            "trajectory_average_speed": float(
                self._trajectory_metadata.get("average_speed", 0.0)),
            "trajectory_peak_speed": float(
                self._trajectory_metadata.get("peak_speed", 0.0)),
            "wind_enabled": bool(self._wind_enabled),
            "wind_force_ned": self._current_wind_force_ned.astype(float).copy(),
            "wind_force_norm": float(np.linalg.norm(self._current_wind_force_ned)),
            "maximum_wind_force": float(self._maximum_wind_force),
        }


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return np.array([
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    ])


def _euler_quaternion(angles: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = angles
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


__all__ = ["MujocoTrajectoryTrackingEnv"]
