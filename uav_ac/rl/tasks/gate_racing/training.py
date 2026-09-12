"""PPO training orchestration for gate racing."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ...acmpc.racing import RacingACMPCPolicy
from .config import contract, read_run, settings_from
from .environment import make_environment
from .evaluation import evaluate_model


class RacingEvaluationCallback(BaseCallback):
    def __init__(self, settings, run_dir):
        super().__init__()
        self.settings, self.run_dir = settings, Path(run_dir)
        self.frequency = max(1, settings["evaluation_interval"] // settings["n_envs"])
        self.best = (-1., -float("inf"), -float("inf"))
        path = self.run_dir / "best_evaluation.json"
        if path.exists():
            self.best = self.rank(json.loads(path.read_text()))

    @staticmethod
    def rank(result):
        return (result["success_rate"], -(result["mean_success_seconds"] or float("inf")),
                result["mean_gates_passed"])

    def evaluate(self):
        result = evaluate_model(self.model, self.settings, tuple(range(10000, 10000+self.settings["evaluation_episodes"])))
        result["timesteps"] = self.model.num_timesteps
        (self.run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        for key in ("success_rate", "mean_gates_passed", "collision_rate", "out_of_bounds_rate", "solver_failure_rate"):
            self.logger.record(f"racing/{key}", result[key])
        self.logger.dump(step=self.model.num_timesteps)
        if self.rank(result) > self.best:
            self.best = self.rank(result)
            self.model.save(self.run_dir / "best_model.zip")
            (self.run_dir / "best_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.logger.record("racing/episode_gates", info["gates_passed"])
        if self.n_calls % self.frequency == 0 and self.num_timesteps < self.model._total_timesteps:
            self.evaluate()
        return True


def train(run_dir, settings, *, resume=None, **overrides):
    settings = deepcopy(settings)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    settings = settings_from(settings)
    torch.set_num_threads(settings["torch_threads"])
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise ValueError("use a new run directory or explicit --resume")
    probe = make_environment(settings)
    try:
        check_env(probe, warn=True)
        signature = contract(settings, probe)
    finally:
        probe.close()
    old = None
    if resume is not None:
        resume = Path(resume).resolve(strict=True)
        config_dir = resume.parent if (resume.parent / "rl_config.json").exists() else resume.parent.parent
        old, _ = read_run(config_dir)
        if old["contract"] != signature or old["settings"]["ppo"] != settings["ppo"]:
            raise ValueError("resume has incompatible task, physics, MPC or PPO settings")
        if (run_dir / "rl_config.json").exists():
            current, _ = read_run(run_dir)
            if current["contract"] != signature:
                raise ValueError("destination run has incompatible contract")
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("resume destination contains unrelated files")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rl_config.json").write_text(json.dumps({"task": "gate_racing", "contract": signature,
                                                       "settings": settings}, indent=2) + "\n")
    suffix = f"_resume_{time.time_ns()}" if resume is not None else ""

    def factory(rank):
        return Monitor(make_environment(settings), str(run_dir / f"monitor_{rank}{suffix}.csv"))

    vector = DummyVecEnv([partial(factory, i) for i in range(settings["n_envs"])])
    ppo = settings["ppo"]
    model = None
    try:
        if resume is not None:
            model = PPO.load(resume, env=vector, device=settings["device"])
            model.set_random_seed(settings["seed"])
        else:
            extra = {"quad_parameters": signature["physical_parameters"], "mpc_settings": settings["mpc"]} if settings["policy_type"] == "acmpc" else {}
            model = PPO(RacingACMPCPolicy if extra else "MlpPolicy", vector,
                        policy_kwargs={**extra, "net_arch": ppo["net_arch"], "activation_fn": torch.nn.ReLU,
                                       "log_std_init": -2.},
                        n_steps=ppo["n_steps"], batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
                        gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
                        ent_coef=ppo["ent_coef"], max_grad_norm=ppo["max_grad_norm"],
                        learning_rate=lambda p: ppo["learning_rate_end"] + p*(ppo["learning_rate_start"]-ppo["learning_rate_end"]),
                        seed=settings["seed"], device=settings["device"], verbose=1,
                        tensorboard_log=str(run_dir / "tensorboard"))
        if settings["policy_type"] == "acmpc":
            model.policy.solver_strict = True
        evaluation = RacingEvaluationCallback(settings, run_dir)
        checkpoint = CheckpointCallback(max(1, settings["checkpoint_interval"]//settings["n_envs"]),
                                        str(run_dir / "checkpoints"), name_prefix="ppo")
        model.learn(settings["total_timesteps"], callback=[evaluation, checkpoint], reset_num_timesteps=resume is None)
        model.save(run_dir / "final_model.zip")
        evaluation.evaluate()
    except (Exception, KeyboardInterrupt) as error:
        if model is not None:
            model.save(run_dir / "interrupted_model.zip")
            diagnostics = getattr(getattr(model.policy, "mpc", None), "last_diagnostics", {})
            (run_dir / "solver_failure.json").write_text(json.dumps({"error": repr(error), "diagnostics": diagnostics}, indent=2) + "\n")
        raise
    finally:
        vector.close()
    return run_dir
