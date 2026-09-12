"""Training-environment composition for trajectory tracking."""

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH
from uav_ac.simulation.wind_disturb import RandomWindConfig
from ...common.environment import MujocoTrajectoryTrackingEnv
from ...common.trajectory_bank import TrajectoryBank

def make_environment_factory(
        bank_path: Path,
        monitor_path: Path,
        *,
        rank: int,
        steps_per_action: int,
        wind_settings: dict[str, Any],
        observation_mode: str = "mlp",
        mpc_horizon_steps: int = 20,
        model_path: str | Path = OPEN_FIELD_SCENE_PATH,
) -> Callable[[], gym.Env]:
    """Create one picklable, memory-mapped training worker."""
    def factory() -> gym.Env:
        environment = MujocoTrajectoryTrackingEnv(
            TrajectoryBank.load(bank_path),
            model_path=model_path,
            steps_per_action=steps_per_action,
            random_start=True,
            perturb_initial_state=True,
            curriculum_progress=0.0,
            split="train",
            wind_config=RandomWindConfig(**wind_settings),
            observation_mode=observation_mode,
            mpc_horizon_steps=mpc_horizon_steps,
        )
        return Monitor(
            environment,
            filename=str(monitor_path / f"worker_{rank}.csv"),
            info_keywords=(
                "success", "position_error", "collision", "failure_reason",
                "position_rmse", "start_index", "trajectory_id",
                "wind_enabled", "maximum_wind_force",
            ),
        )
    return factory
