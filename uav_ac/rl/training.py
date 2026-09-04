"""Train PPO for MuJoCo trajectory tracking.

Run with ``python -m uav_ac.rl.training`` after installing the ``rl`` extra.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from uav_ac import utils
from uav_ac.control.rl_controller import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    RL_CONFIG_FILENAME,
    RL_CONFIG_VERSION,
    quad_parameters,
)
from uav_ac.main import _plan_trajectory
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation

from .assets import (
    INITIALIZATION_LIBRARY_FILENAME,
    TRAJECTORY_FILENAME,
    InitializationLibrary,
    generate_initialization_library,
)
from .environment import MujocoTrajectoryTrackingEnv


DEFAULT_TOTAL_TIMESTEPS = 10_000_000
DEFAULT_ENVIRONMENTS = 6
DEFAULT_SEED = 42
DEFAULT_STEPS_PER_ACTION = 10
EVALUATION_INTERVAL = 100_000
CHECKPOINT_INTERVAL = 250_000
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ppo_trajectory.yaml"


DEFAULT_TRAINING_CONFIG = {
    "seed": DEFAULT_SEED,
    "device": "cuda",
    "n_envs": DEFAULT_ENVIRONMENTS,
    "total_timesteps": DEFAULT_TOTAL_TIMESTEPS,
    "steps_per_action": DEFAULT_STEPS_PER_ACTION,
    "evaluation_interval": EVALUATION_INTERVAL,
    "checkpoint_interval": CHECKPOINT_INTERVAL,
    "ppo": {
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "learning_rate_start": 3.0e-4,
        "learning_rate_end": 3.0e-5,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "max_grad_norm": 0.5,
        "activation": "relu",
        "net_arch": {"pi": [512, 512], "vf": [512, 512]},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_training_config(config_path: str | Path | None = None) -> dict:
    """Load and validate the editable YAML training configuration."""
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    values = {}
    if not path.is_file():
        if config_path is not None:
            raise FileNotFoundError(f"training config does not exist: {path}")
    else:
        with path.open("r", encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
        if not isinstance(values, dict):
            raise ValueError(f"training config must contain a YAML mapping: {path}")
    settings = _deep_merge(DEFAULT_TRAINING_CONFIG, values)
    ppo = settings["ppo"]
    required_positive = (
        "total_timesteps", "n_envs", "steps_per_action",
        "evaluation_interval", "checkpoint_interval",
    )
    if any(int(settings[key]) < 1 for key in required_positive):
        raise ValueError("total_timesteps, n_envs, steps_per_action and intervals must be positive")
    for key in ("n_steps", "batch_size", "n_epochs"):
        if int(ppo[key]) < 1:
            raise ValueError(f"ppo.{key} must be positive")
    if int(ppo["batch_size"]) > int(ppo["n_steps"]) * int(settings["n_envs"]):
        raise ValueError("ppo.batch_size cannot exceed n_steps * n_envs")
    for key in ("gamma", "gae_lambda", "clip_range", "ent_coef", "max_grad_norm",
                "learning_rate_start", "learning_rate_end"):
        if float(ppo[key]) < 0.0:
            raise ValueError(f"ppo.{key} must be non-negative")
    if str(ppo["activation"]).lower() != "relu":
        raise ValueError("only the planned ReLU policy is currently supported")
    for branch in ("pi", "vf"):
        architecture = ppo["net_arch"].get(branch)
        if not architecture or any(int(width) < 1 for width in architecture):
            raise ValueError(f"ppo.net_arch.{branch} must contain positive layer widths")
    return settings


class CurriculumCallback(BaseCallback):
    """Broadcast the state-perturbation ramp to every training worker."""

    def __init__(self, update_frequency: int = 1_000):
        super().__init__(verbose=0)
        self.update_frequency = int(update_frequency)

    def _on_training_start(self) -> None:
        self._update_environments()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_frequency == 0:
            self._update_environments()
        return True

    def _update_environments(self) -> None:
        total_timesteps = max(int(self.model._total_timesteps), 1)
        progress = self.model.num_timesteps / total_timesteps
        self.training_env.env_method("set_training_progress", progress)


class EpisodeMetricsCallback(BaseCallback):
    """Expose environment episode diagnostics in SB3 console/TensorBoard logs."""

    def __init__(self, window_size: int = 100):
        super().__init__(verbose=0)
        self.window_size = int(window_size)
        self._recent_episodes = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for index, info in enumerate(infos):
            if index >= len(dones) or not dones[index]:
                continue
            episode_info = info.get("episode", {})
            self._recent_episodes.append({
                "success": bool(info.get("success", episode_info.get("success", False))),
                "collision": bool(info.get("collision", episode_info.get("collision", False))),
                "failure_reason": info.get(
                    "failure_reason", episode_info.get("failure_reason")),
                "position_rmse": float(info.get(
                    "position_rmse", episode_info.get("position_rmse", np.nan))),
            })

        if self._recent_episodes:
            episodes = list(self._recent_episodes)
            successful = [episode for episode in episodes if episode["success"]]
            self.logger.record(
                "rollout/success_rate",
                np.mean([episode["success"] for episode in episodes]),
            )
            self.logger.record(
                "rollout/collision_rate",
                np.mean([episode["collision"] for episode in episodes]),
            )
            successful_rmses = [
                episode["position_rmse"] for episode in successful
                if np.isfinite(episode["position_rmse"])
            ]
            self.logger.record(
                "rollout/successful_mean_position_rmse",
                np.mean(successful_rmses) if successful_rmses else 0.0,
            )
            self.logger.record("rollout/successful_episode_count", len(successful))
            failure_reasons = {
                str(episode["failure_reason"])
                for episode in episodes
                if episode["failure_reason"] is not None
            }
            failed_count = sum(not episode["success"] for episode in episodes)
            for reason in sorted(failure_reasons):
                reason_rate = sum(
                    episode["failure_reason"] == reason for episode in episodes
                ) / max(failed_count, 1)
                self.logger.record(f"rollout/failure_{reason}_rate", reason_rate)
            last_failure = next(
                (episode["failure_reason"] for episode in reversed(episodes)
                 if episode["failure_reason"] is not None),
                None,
            )
            if last_failure is not None:
                # Human/CSV output is useful for the categorical reason; avoid
                # sending a string scalar to TensorBoard's numeric dashboard.
                self.logger.record(
                    "rollout/last_failure_reason", last_failure, exclude="tensorboard")
        return True


def linear_learning_rate(
        progress_remaining: float,
        start: float = 3.0e-4,
        end: float = 3.0e-5,
) -> float:
    """Linearly decay the configured learning rate over one learn call."""
    return float(end + progress_remaining * (start - end))


def prepare_assets(
        run_dir: Path,
        *,
        steps_per_action: int = DEFAULT_STEPS_PER_ACTION,
) -> tuple[np.ndarray, InitializationLibrary, dict]:
    """Plan GCOPTER once and create the cascaded reset-state library."""
    trajectory_path = run_dir / TRAJECTORY_FILENAME
    library_path = run_dir / INITIALIZATION_LIBRARY_FILENAME
    simulation = MujocoSimulation(DEFAULT_SCENE_PATH, record_actual_trajectory=False)
    if not np.isclose(simulation.quad.dt, 0.001):
        raise ValueError("RL baseline requires a 1 kHz MuJoCo model timestep")
    trajectory_dt = simulation.quad.dt * steps_per_action
    if trajectory_path.is_file():
        trajectory = np.load(trajectory_path)
    else:
        _, flight_config = utils.get_config()
        trajectory = _plan_trajectory(
            "gcopter",
            simulation,
            flight_config.getfloat("velocity"),
            trajectory_dt,
            visualize=False,
        )
        np.save(trajectory_path, trajectory)

    if library_path.is_file():
        library = InitializationLibrary.load(library_path)
    else:
        library = generate_initialization_library(
            trajectory,
            steps_per_reference=steps_per_action,
            model_path=DEFAULT_SCENE_PATH,
        )
        library.save(library_path)
    if len(library.states) != len(trajectory):
        raise ValueError("cached initialization library does not match the trajectory")
    return trajectory, library, quad_parameters(simulation.quad)


def make_environment_factory(
        trajectory_path: Path,
        library_path: Path,
        monitor_path: Path,
    *,
    rank: int,
    steps_per_action: int = DEFAULT_STEPS_PER_ACTION,
) -> Callable[[], gym.Env]:
    """Create a picklable per-process environment factory."""
    def factory() -> gym.Env:
        environment = MujocoTrajectoryTrackingEnv(
            np.load(trajectory_path),
            InitializationLibrary.load(library_path),
            model_path=DEFAULT_SCENE_PATH,
            steps_per_action=steps_per_action,
            random_start=True,
            perturb_initial_state=True,
            curriculum_progress=0.0,
        )
        return Monitor(
            environment,
            filename=str(monitor_path / f"worker_{rank}.csv"),
            info_keywords=(
                "success", "position_error", "collision", "failure_reason",
                "position_rmse", "start_index",
            ),
        )
    return factory


def train(
        run_dir: str | Path,
        *,
        total_timesteps: int | None = None,
        n_envs: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        resume: str | Path | None = None,
        config_path: str | Path | None = None,
) -> Path:
    """Prepare assets, train or resume PPO, and return the run directory."""
    settings = load_training_config(config_path)
    total_timesteps = int(settings["total_timesteps"] if total_timesteps is None else total_timesteps)
    n_envs = int(settings["n_envs"] if n_envs is None else n_envs)
    seed = int(settings["seed"] if seed is None else seed)
    device = str(settings["device"] if device is None else device)
    steps_per_action = int(settings["steps_per_action"])
    ppo = settings["ppo"]
    if total_timesteps < 1 or n_envs < 1:
        raise ValueError("total_timesteps and n_envs must be positive")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = run_dir / "monitor"
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    monitor_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    trajectory, library, physical_parameters = prepare_assets(
        run_dir, steps_per_action=steps_per_action)
    environment_check = MujocoTrajectoryTrackingEnv(
        trajectory,
        library,
        steps_per_action=steps_per_action,
        random_start=False,
        perturb_initial_state=False,
    )
    check_env(environment_check, warn=True)
    environment_check.close()

    configuration = {
        "schema_version": RL_CONFIG_VERSION,
        "planner": "gcopter",
        "scene": "lab_course.xml",
        "observation_dim": OBSERVATION_SIZE,
        "action_shape": [ACTION_SIZE],
        "action_low": [-1.0] * ACTION_SIZE,
        "action_high": [1.0] * ACTION_SIZE,
        "physics_dt": physical_parameters["physics_dt"],
        "control_dt": physical_parameters["physics_dt"] * steps_per_action,
        "steps_per_action": steps_per_action,
        "quad_parameters": physical_parameters,
        "trajectory_file": TRAJECTORY_FILENAME,
        "initialization_library_file": INITIALIZATION_LIBRARY_FILENAME,
        "seed": int(seed),
        "device": device,
        "n_envs": int(n_envs),
        "total_timesteps": int(total_timesteps),
        "ppo": {
            "net_arch": ppo["net_arch"],
            "activation_fn": "ReLU",
            "n_steps": int(ppo["n_steps"]),
            "batch_size": int(ppo["batch_size"]),
            "n_epochs": int(ppo["n_epochs"]),
            "gamma": float(ppo["gamma"]),
            "gae_lambda": float(ppo["gae_lambda"]),
            "learning_rate": [
                float(ppo["learning_rate_start"]),
                float(ppo["learning_rate_end"]),
            ],
            "clip_range": float(ppo["clip_range"]),
            "ent_coef": float(ppo["ent_coef"]),
            "max_grad_norm": float(ppo["max_grad_norm"]),
        },
        "evaluation_interval": int(settings["evaluation_interval"]),
        "checkpoint_interval": int(settings["checkpoint_interval"]),
    }
    with (run_dir / RL_CONFIG_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)
        file.write("\n")

    factories = [
        make_environment_factory(
            run_dir / TRAJECTORY_FILENAME,
            run_dir / INITIALIZATION_LIBRARY_FILENAME,
            monitor_dir,
            rank=rank,
            steps_per_action=steps_per_action,
        )
        for rank in range(n_envs)
    ]
    vector_environment = (
        DummyVecEnv(factories)
        if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    )
    evaluation_environment = Monitor(MujocoTrajectoryTrackingEnv(
        trajectory,
        library,
        random_start=False,
        perturb_initial_state=False,
        curriculum_progress=0.0,
        steps_per_action=steps_per_action,
    ))

    if resume is None:
        model = PPO(
            "MlpPolicy",
            vector_environment,
            policy_kwargs={
                "activation_fn": torch.nn.ReLU,
                "net_arch": {
                    "pi": [int(width) for width in ppo["net_arch"]["pi"]],
                    "vf": [int(width) for width in ppo["net_arch"]["vf"]],
                },
            },
            n_steps=int(ppo["n_steps"]),
            batch_size=int(ppo["batch_size"]),
            n_epochs=int(ppo["n_epochs"]),
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
            learning_rate=lambda progress: linear_learning_rate(
                progress,
                float(ppo["learning_rate_start"]),
                float(ppo["learning_rate_end"]),
            ),
            clip_range=float(ppo["clip_range"]),
            ent_coef=float(ppo["ent_coef"]),
            max_grad_norm=float(ppo["max_grad_norm"]),
            tensorboard_log=str(tensorboard_dir),
            seed=seed,
            device=device,
            verbose=1,
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(resume, env=vector_environment, device=device)
        reset_num_timesteps = False

    callbacks = CallbackList([
        CurriculumCallback(),
        EpisodeMetricsCallback(),
        EvalCallback(
            evaluation_environment,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "evaluation"),
            eval_freq=max(int(settings["evaluation_interval"]) // n_envs, 1),
            n_eval_episodes=1,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(int(settings["checkpoint_interval"]) // n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_trajectory",
        ),
    ])
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model.save(run_dir / "final_model")
        if not (run_dir / "best_model.zip").is_file():
            model.save(run_dir / "best_model")
    finally:
        vector_environment.close()
        evaluation_environment.close()
    return run_dir


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
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    output = train(
        arguments.run_dir,
        total_timesteps=arguments.total_timesteps,
        n_envs=arguments.n_envs,
        seed=arguments.seed,
        device=arguments.device,
        resume=arguments.resume,
        config_path=arguments.config,
    )
    print(f"RL training artifacts: {output}")


if __name__ == "__main__":
    main()
