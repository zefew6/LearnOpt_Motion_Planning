"""Trajectory and reset-state assets used by RL training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uav_ac.control import CascadedController, TrajectoryController
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation


TRAJECTORY_FILENAME = "trajectory.npy"
INITIALIZATION_LIBRARY_FILENAME = "initialization_states.npz"


@dataclass(frozen=True)
class InitializationLibrary:
    """One legal vehicle and motor snapshot for every reference sample."""

    states: np.ndarray
    motor_speeds: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        motor_speeds = np.asarray(self.motor_speeds, dtype=float)
        if states.ndim != 2 or states.shape[1] != 13 or not np.all(np.isfinite(states)):
            raise ValueError("initialization states must have shape (n, 13) and be finite")
        if (motor_speeds.shape != (len(states), 4)
                or not np.all(np.isfinite(motor_speeds))
                or np.any(motor_speeds < 0.0)):
            raise ValueError("initialization motor speeds must have shape (n, 4) and be non-negative")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "motor_speeds", motor_speeds)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, states=self.states, motor_speeds=self.motor_speeds)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "InitializationLibrary":
        with np.load(Path(path)) as data:
            return cls(states=data["states"], motor_speeds=data["motor_speeds"])


def generate_initialization_library(
        trajectory: np.ndarray,
        *,
        steps_per_reference: int = 10,
        model_path: str | Path = DEFAULT_SCENE_PATH,
) -> InitializationLibrary:
    """Track once with the cascaded controller and capture reset snapshots."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 10 or len(trajectory) == 0:
        raise ValueError("trajectory must have shape (n, m), n >= 1 and m >= 10")
    simulation = MujocoSimulation(model_path, record_actual_trajectory=False)
    trajectory_dt = simulation.quad.dt * steps_per_reference
    controller = CascadedController(simulation.quad.g, trajectory_dt)
    tracker = TrajectoryController(
        controller, simulation.quad, trajectory, steps_per_reference)
    states = []
    motor_speeds = []
    for reference_index in range(len(trajectory)):
        if simulation.collision_detected:
            raise RuntimeError(
                f"cascaded initialization flight collided at reference {reference_index}")
        states.append(simulation.quad.X.copy())
        motor_speeds.append(simulation.quad.omega.copy())
        if reference_index == len(trajectory) - 1:
            break
        for _ in range(steps_per_reference):
            tracker.step()
            simulation.step()
            if not np.all(np.isfinite(simulation.quad.X)):
                raise RuntimeError("cascaded initialization flight became non-finite")
    return InitializationLibrary(np.asarray(states), np.asarray(motor_speeds))


def ideal_initialization_library(trajectory: np.ndarray, quad) -> InitializationLibrary:
    """Create deterministic reference states for tests when no flight cache is supplied."""
    trajectory = np.asarray(trajectory, dtype=float)
    states = np.zeros((len(trajectory), 13))
    states[:, :3] = trajectory[:, :3]
    states[:, 7:10] = trajectory[:, 3:6]
    half_yaw = trajectory[:, 9] / 2.0
    states[:, 3] = np.cos(half_yaw)
    states[:, 6] = np.sin(half_yaw)
    hover_speed = np.sqrt(quad.m * quad.g / (4.0 * quad.kf))
    motor_speeds = np.full((len(trajectory), 4), hover_speed)
    return InitializationLibrary(states, motor_speeds)


__all__ = [
    "INITIALIZATION_LIBRARY_FILENAME",
    "TRAJECTORY_FILENAME",
    "InitializationLibrary",
    "generate_initialization_library",
    "ideal_initialization_library",
]
