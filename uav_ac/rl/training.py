"""Unified reinforcement-learning training entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH
# Compatibility exports retained for callers that imported implementation
# dependencies from this former monolithic module.
from uav_ac.control.rl_controller import (
    ACTION_SIZE, OBSERVATION_SIZE, RL_CONFIG_FILENAME, RL_CONFIG_VERSION,
    quad_parameters,
)
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.simulation.wind_disturb import RandomWindConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .common.config import deep_merge, load_yaml_mapping
from .common.environment import MujocoTrajectoryTrackingEnv
from .common.registry import get_workflow, task_name
from .common.trajectory_bank import (
    TRAJECTORY_BANK_DIRECTORY, TRAJECTORY_BANK_SCHEMA_VERSION, TrajectoryBank,
    generate_trajectory_bank,
)
from .tasks.trajectory_tracking.assets import prepare_assets
from .tasks.trajectory_tracking.config import (
    CHECKPOINT_INTERVAL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENVIRONMENTS,
    DEFAULT_SEED,
    DEFAULT_STEPS_PER_ACTION,
    DEFAULT_TOTAL_TIMESTEPS,
    DEFAULT_TRAINING_CONFIG,
    EVALUATION_INTERVAL,
    validate_training_config,
)
from .tasks.trajectory_tracking.environment import make_environment_factory
from .tasks.trajectory_tracking.training import (
    CurriculumCallback,
    EpisodeMetricsCallback,
    TrajectoryBankEvaluationCallback,
    linear_learning_rate,
)

_deep_merge = deep_merge
_validate_training_config = validate_training_config


def load_training_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML and validate it with the explicitly registered task workflow."""
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    values = load_yaml_mapping(path, missing_ok=config_path is None)
    return get_workflow(task_name(values)).settings_from(values)


def train(
        run_dir: str | Path,
        *,
        total_timesteps: int | None = None,
        n_envs: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        resume: str | Path | None = None,
        config_path: str | Path | None = None,
        settings: dict[str, Any] | None = None,
        model_path: str | Path = OPEN_FIELD_SCENE_PATH,
) -> Path:
    """Route training to the workflow selected by config or validated settings."""
    if settings is not None and config_path is not None:
        raise ValueError("settings and config_path are mutually exclusive")
    if settings is not None:
        selected_task = task_name(settings)
    elif config_path is not None:
        selected_task = task_name(load_yaml_mapping(config_path))
    else:
        selected_task = "trajectory_tracking"
    return get_workflow(selected_task).train(
        run_dir,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        seed=seed,
        device=device,
        resume=resume,
        config_path=config_path,
        settings=settings,
        model_path=model_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run = Path("runs/ppo_trajectory") / datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--total-timesteps", type=int, help="override YAML total_timesteps")
    parser.add_argument("--n-envs", type=int, help="override YAML n_envs")
    parser.add_argument("--seed", type=int, help="override YAML seed")
    parser.add_argument("--device", help="override YAML device (cpu/cuda)")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model-path", type=Path, default=OPEN_FIELD_SCENE_PATH)
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="prepare task assets without training",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.prepare_only:
        settings = load_training_config(arguments.config)
        workflow = get_workflow(task_name(settings))
        if workflow.prepare is None:
            raise ValueError(f"{workflow.name.replace('_', ' ')} has no trajectory bank; omit --prepare-only")
        seed = int(settings["seed"] if arguments.seed is None else arguments.seed)
        assets, _ = workflow.prepare(
            arguments.run_dir,
            settings=settings,
            steps_per_action=int(settings["steps_per_action"]),
            seed=seed,
            model_path=arguments.model_path,
        )
        print(f"Trajectory bank: {settings.get('trajectory_bank_path') or arguments.run_dir / 'trajectory_bank'} ({len(assets)} trajectories)")
        return
    output = train(
        arguments.run_dir,
        total_timesteps=arguments.total_timesteps,
        n_envs=arguments.n_envs,
        seed=arguments.seed,
        device=arguments.device,
        resume=arguments.resume,
        config_path=arguments.config,
        model_path=arguments.model_path,
    )
    print(f"RL training artifacts: {output}")


if __name__ == "__main__":
    main()
