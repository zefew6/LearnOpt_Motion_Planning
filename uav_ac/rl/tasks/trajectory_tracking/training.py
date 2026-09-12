"""PPO training orchestration for trajectory tracking."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from uav_ac.control.rl_controller import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    RL_CONFIG_FILENAME,
    RL_CONFIG_VERSION,
    quad_parameters,
)
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH
from uav_ac.simulation.wind_disturb import RandomWindConfig
from ...common.environment import MujocoTrajectoryTrackingEnv
from ...common.trajectory_bank import (
    TRAJECTORY_BANK_DIRECTORY,
    TRAJECTORY_BANK_SCHEMA_VERSION,
    TrajectoryBank,
)
from .assets import prepare_assets
from .config import DEFAULT_CONFIG_PATH, load_training_config, validate_training_config
from .environment import make_environment_factory

_validate_training_config = validate_training_config

class CurriculumCallback(BaseCallback):
    """Broadcast state-perturbation and wind curriculum progress to workers."""

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
    """Expose recent multi-trajectory and wind diagnostics in SB3 logs."""

    def __init__(self, window_size: int = 100):
        super().__init__(verbose=0)
        self._recent_episodes: deque[dict[str, Any]] = deque(maxlen=int(window_size))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        mpc = getattr(self.model.policy, "mpc", None)
        if mpc is not None:
            for name in ("failures", "retries", "max_residual", "seconds"):
                self.logger.record_mean(f"mpc/{name}", mpc.last_diagnostics.get(name, 0.))
            if infos:
                self.logger.record_mean("mpc/allocation_scale", float(np.mean([info.get("allocation_scale", 1.) for info in infos])))
        dones = self.locals.get("dones", ())
        for index, info in enumerate(infos):
            if index >= len(dones) or not dones[index]:
                continue
            episode_info = info.get("episode", {})
            self._recent_episodes.append({
                "success": bool(info.get("success", episode_info.get("success", False))),
                "collision": bool(info.get("collision", episode_info.get("collision", False))),
                "failure_reason": info.get("failure_reason", episode_info.get("failure_reason")),
                "position_rmse": float(info.get(
                    "position_rmse", episode_info.get("position_rmse", np.nan))),
                "wind_enabled": bool(info.get(
                    "wind_enabled", episode_info.get("wind_enabled", False))),
                "maximum_wind_force": float(info.get(
                    "maximum_wind_force", episode_info.get("maximum_wind_force", 0.0))),
            })
        if not self._recent_episodes:
            return True
        episodes = list(self._recent_episodes)
        successful = [episode for episode in episodes if episode["success"]]
        windy = [episode for episode in episodes if episode["wind_enabled"]]
        self.logger.record("rollout/success_rate", np.mean([
            episode["success"] for episode in episodes]))
        self.logger.record("rollout/collision_rate", np.mean([
            episode["collision"] for episode in episodes]))
        self.logger.record("rollout/windy_episode_rate", len(windy) / len(episodes))
        self.logger.record("rollout/windy_success_rate", (
            np.mean([episode["success"] for episode in windy]) if windy else 0.0))
        self.logger.record("rollout/mean_maximum_wind_force", np.mean([
            episode["maximum_wind_force"] for episode in episodes]))
        successful_rmses = [
            episode["position_rmse"] for episode in successful
            if np.isfinite(episode["position_rmse"])
        ]
        self.logger.record(
            "rollout/successful_mean_position_rmse",
            np.mean(successful_rmses) if successful_rmses else 0.0,
        )
        failed_count = sum(not episode["success"] for episode in episodes)
        for reason in sorted({
            str(episode["failure_reason"]) for episode in episodes
            if episode["failure_reason"] is not None
        }):
            self.logger.record(
                f"rollout/failure_{reason}_rate",
                sum(episode["failure_reason"] == reason for episode in episodes)
                / max(failed_count, 1),
            )
        return True


class TrajectoryBankEvaluationCallback(BaseCallback):
    """Evaluate a fixed validation panel without SB3 VecEnv type warnings."""

    def __init__(
            self,
            bank: TrajectoryBank,
            run_dir: Path,
            *,
            steps_per_action: int,
            wind_config: RandomWindConfig,
            eval_freq: int,
            panel_size: int,
            perturb_initial_state: bool,
            seed: int,
            observation_mode: str = "mlp",
            mpc_horizon_steps: int = 20,
            model_path: str | Path = OPEN_FIELD_SCENE_PATH,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.eval_freq = max(int(eval_freq), 1)
        self.panel_ids = bank.indices("validation")[:int(panel_size)]
        self.perturb_initial_state = bool(perturb_initial_state)
        self.seed = int(seed)
        self.model_path = Path(model_path)
        self.best_rank: tuple[float, float, float] | None = None
        self.environment = MujocoTrajectoryTrackingEnv(
            bank,
            model_path=self.model_path,
            steps_per_action=steps_per_action,
            random_start=False,
            perturb_initial_state=self.perturb_initial_state,
            curriculum_progress=1.0,
            split="validation",
            observation_mode=observation_mode,
            mpc_horizon_steps=mpc_horizon_steps,
            wind_config=wind_config,
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        episodes = []
        if self.environment.observation_mode == "acmpc":
            from ...acmpc.benchmark import collect_panel
            cases = [dict(trajectory_id=int(trajectory_id), seed=self.seed+2*index+int(windy),
                wind_enabled=windy, perturbation_scale=1. if self.perturb_initial_state else 0.)
                for index,trajectory_id in enumerate(self.panel_ids) for windy in (False,True)]
            envs = [MujocoTrajectoryTrackingEnv(self.environment.trajectory_bank,
                model_path=self.model_path,steps_per_action=self.environment.steps_per_action,
                split="validation",observation_mode="acmpc",mpc_horizon_steps=self.environment.mpc_horizon_steps,
                curriculum_progress=1., wind_config=self.environment.wind_config) for _ in cases]
            try:
                episodes = collect_panel(self.model, envs, cases)
            finally:
                for env in envs:
                    env.close()
        else:
            for panel_index, trajectory_id in enumerate(self.panel_ids):
                for windy in (False, True):
                    episodes.append(self._evaluate_episode(
                        int(trajectory_id), self.seed + 2 * panel_index + int(windy), windy))
        success_rate = float(np.mean([episode["success"] for episode in episodes]))
        rmses = [episode["position_rmse"] for episode in episodes if episode["success"]]
        mean_rmse = float(np.mean(rmses)) if rmses else float("inf")
        mean_return = float(np.mean([episode["return"] for episode in episodes]))
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/successful_mean_position_rmse", (
            mean_rmse if np.isfinite(mean_rmse) else 0.0))
        self.logger.record("eval/mean_return", mean_return)
        result = {
            "timesteps": int(self.num_timesteps),
            "success_rate": success_rate,
            "successful_mean_position_rmse": (
                mean_rmse if np.isfinite(mean_rmse) else None),
            "mean_return": mean_return,
            "episodes": episodes,
        }
        with (self.run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        rank = (success_rate, -mean_rmse, mean_return)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank = rank
            self.model.save(self.run_dir / "best_model")
        return True

    def _evaluate_episode(
            self, trajectory_id: int, seed: int, windy: bool,
    ) -> dict[str, Any]:
        observation, _ = self.environment.reset(
            seed=seed,
            options={
                "trajectory_id": trajectory_id,
                "start_index": 0,
                "perturbation_scale": 1.0 if self.perturb_initial_state else 0.0,
                "wind_enabled": windy,
                "wind_scale": 1.0,
            },
        )
        maximum_steps = len(self.environment.trajectory) + int(round(
            1.0 / self.environment.control_dt)) + 1
        episode_return = 0.0
        info: dict[str, Any] = {}
        for _ in range(maximum_steps):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = self.environment.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        return {
            "trajectory_id": trajectory_id,
            "seed": seed,
            "wind_enabled": windy,
            "success": bool(info.get("success", False)),
            "failure_reason": info.get("failure_reason"),
            "position_rmse": float(info.get("position_rmse", np.inf)),
            "return": episode_return,
        }

    def _on_training_end(self) -> None:
        self.environment.close()


def linear_learning_rate(
        progress_remaining: float,
        start: float = 3.0e-4,
        end: float = 3.0e-5,
) -> float:
    return float(end + progress_remaining * (start - end))




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
    """Train or resume PPO, preserving the legacy run artifact layout.

    ``settings`` accepts the validated legacy-shaped mapping supplied by the
    unified config adapter and requires an existing trajectory bank. It is
    mutually exclusive with ``config_path``. Legacy config calls may generate
    assets when no bank is configured.
    """
    if settings is not None and config_path is not None:
        raise ValueError("settings and config_path are mutually exclusive")
    supplied_settings = settings is not None
    if settings is None:
        settings = load_training_config(config_path)
    if supplied_settings:
        settings = deepcopy(settings)
        if not settings.get("trajectory_bank_path"):
            raise ValueError("settings requires an existing trajectory_bank_path; implicit generation is disabled")
    _validate_training_config(settings)
    model_path = Path(model_path)
    total_timesteps = int(settings["total_timesteps"] if total_timesteps is None else total_timesteps)
    n_envs = int(settings["n_envs"] if n_envs is None else n_envs)
    seed = int(settings["seed"] if seed is None else seed)
    device = str(settings["device"] if device is None else device)
    steps_per_action = int(settings["steps_per_action"])
    ppo = settings["ppo"]
    policy_type = settings["policy_type"]
    mpc_settings = {}
    if policy_type == "acmpc":
        from ...acmpc.solver import MPCSettings
        mpc_settings = asdict(MPCSettings(**settings.get("mpc", {})))
        torch.set_num_threads(int(settings.get("torch_threads", 1)))
    if total_timesteps < 1 or n_envs < 1:
        raise ValueError("total_timesteps and n_envs must be positive")

    run_dir = Path(run_dir)
    if policy_type == "acmpc" and (run_dir / RL_CONFIG_FILENAME).exists() and resume is None:
        raise ValueError("ACMPC run already exists; use --resume or a new run directory")
    resumed_model = PPO.load(resume, device=device) if resume is not None else None
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = run_dir / "monitor"
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    for directory in (monitor_dir, checkpoint_dir, tensorboard_dir):
        directory.mkdir(exist_ok=True)

    bank, physical_parameters = prepare_assets(
        run_dir, settings=settings, steps_per_action=steps_per_action, seed=seed,
        model_path=model_path)
    wind_config = RandomWindConfig(**settings["wind"])
    environment_check = MujocoTrajectoryTrackingEnv(
        bank,
        model_path=model_path,
        steps_per_action=steps_per_action,
        random_start=False,
        perturb_initial_state=False,
        split="train",
        wind_config=wind_config,
        observation_mode=policy_type,
        mpc_horizon_steps=mpc_settings.get("horizon_steps", 20),
    )
    check_env(environment_check, warn=True)
    environment_check.close()

    configuration = {
        "task": "trajectory_tracking",
        "schema_version": 2 if policy_type == "acmpc" else RL_CONFIG_VERSION,
        "policy_type": policy_type,
        "asset_schema_version": TRAJECTORY_BANK_SCHEMA_VERSION,
        "planner": "gcopter",
        "scene": model_path.name,
        "observation_dim": OBSERVATION_SIZE,
        "action_shape": [ACTION_SIZE],
        "action_low": [-1.0] * ACTION_SIZE,
        "action_high": [1.0] * ACTION_SIZE,
        "physics_dt": physical_parameters["physics_dt"],
        "control_dt": physical_parameters["physics_dt"] * steps_per_action,
        "steps_per_action": steps_per_action,
        "quad_parameters": physical_parameters,
        "trajectory_bank_directory": str(Path(settings["trajectory_bank_path"]).resolve()) if settings.get("trajectory_bank_path") else TRAJECTORY_BANK_DIRECTORY,
        "trajectory_splits": settings["trajectory_bank"]["splits"],
        "wind": asdict(wind_config),
        "seed": seed,
        "device": device,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "ppo": {
            "net_arch": ppo["net_arch"],
            "activation_fn": "ReLU",
            "n_steps": int(ppo["n_steps"]),
            "batch_size": int(ppo["batch_size"]),
            "n_epochs": int(ppo["n_epochs"]),
            "gamma": float(ppo["gamma"]),
            "gae_lambda": float(ppo["gae_lambda"]),
            "learning_rate": [
                float(ppo["learning_rate_start"]), float(ppo["learning_rate_end"])],
            "clip_range": float(ppo["clip_range"]),
            "ent_coef": float(ppo["ent_coef"]),
            "max_grad_norm": float(ppo["max_grad_norm"]),
        },
        "evaluation_interval": int(settings["evaluation_interval"]),
        "checkpoint_interval": int(settings["checkpoint_interval"]),
    }
    if policy_type == "acmpc":
        from ...acmpc.solver import SOLVER_VERSION
        if not np.isclose(mpc_settings["dt"], configuration["control_dt"]):
            raise ValueError("MPC dt must equal environment control_dt")
        configuration.update(mpc=mpc_settings, observation_version=1, solver_version=SOLVER_VERSION)
        if resumed_model is not None:
            if (getattr(resumed_model.policy, "mpc_settings", None) != mpc_settings
                    or getattr(resumed_model.policy, "quad_parameters", None) != physical_parameters):
                raise ValueError("resume checkpoint is incompatible with ACMPC configuration")
            for name in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "max_grad_norm"):
                if not np.isclose(getattr(resumed_model, name), ppo[name]):
                    raise ValueError(f"resume checkpoint has incompatible PPO {name}")
    with (run_dir / RL_CONFIG_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)
        file.write("\n")

    factories = [
        make_environment_factory(
            Path(configuration["trajectory_bank_directory"]) if settings.get("trajectory_bank_path") else run_dir / TRAJECTORY_BANK_DIRECTORY,
            monitor_dir,
            rank=rank,
            model_path=model_path,
            steps_per_action=steps_per_action,
            wind_settings=settings["wind"],
            observation_mode=policy_type,
            mpc_horizon_steps=mpc_settings.get("horizon_steps", 20),
        )
        for rank in range(n_envs)
    ]
    vector_environment = (
        DummyVecEnv(factories)
        if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    )
    if resume is None:
        policy_class = "MlpPolicy"
        extra_policy_kwargs = {}
        if policy_type == "acmpc":
            from ...acmpc.policy import ACMPCPolicy
            policy_class = ACMPCPolicy
            extra_policy_kwargs = {"quad_parameters": physical_parameters, "mpc_settings": mpc_settings, "log_std_init": -2.0}
        model = PPO(
            policy_class,
            vector_environment,
            policy_kwargs={
                **extra_policy_kwargs,
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
        model = resumed_model
        model.set_env(vector_environment)
        reset_num_timesteps = False
    if policy_type == "acmpc":
        model.policy.solver_strict = True

    callbacks = CallbackList([
        CurriculumCallback(),
        EpisodeMetricsCallback(),
        TrajectoryBankEvaluationCallback(
            bank,
            run_dir,
            model_path=model_path,
            steps_per_action=steps_per_action,
            wind_config=wind_config,
            eval_freq=max(int(settings["evaluation_interval"]) // n_envs, 1),
            panel_size=int(settings["evaluation"]["panel_size"]),
            perturb_initial_state=bool(settings["evaluation"]["perturb_initial_state"]),
            seed=seed + 10_000,
            observation_mode=policy_type,
            mpc_horizon_steps=mpc_settings.get("horizon_steps", 20),
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
    except Exception as error:
        if policy_type == "acmpc":
            model.save(run_dir / "interrupted_model")
            diagnostic = dict(model.policy.mpc.last_diagnostics, error=str(error), timesteps=model.num_timesteps)
            (run_dir / "solver_failure.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        raise
    finally:
        vector_environment.close()
    return run_dir
