"""Gate-racing RL workflow registration."""

from copy import deepcopy
from pathlib import Path

from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH
from ...common.config import load_yaml_mapping
from ...common.registry import TaskWorkflow
from .config import settings_from
from .evaluation import evaluate_metrics as _evaluate_metrics, replay
from .training import train as _train


def dispatch_train(run_dir, *, total_timesteps=None, n_envs=None, seed=None, device=None,
          resume=None, config_path=None, settings=None,
          model_path=OPEN_FIELD_SCENE_PATH):
    if settings is not None and config_path is not None:
        raise ValueError("settings and config_path are mutually exclusive")
    if settings is None and config_path is None:
        raise ValueError("gate-racing training requires a config path or settings")
    values = deepcopy(settings) if settings is not None else load_yaml_mapping(config_path)
    validated = settings_from(values)
    if Path(model_path) != OPEN_FIELD_SCENE_PATH:
        raise ValueError("gate racing selects its scene through the training config")
    return _train(run_dir, validated, resume=resume, total_timesteps=total_timesteps,
                  n_envs=n_envs, seed=seed, device=device)


def evaluate_metrics(run_dir, *, seeds=tuple(range(20)),
                     perturb_initial_state=True, device="cpu", split="test",
                     trajectory_id=None, wind="both"):
    if wind == "random" or trajectory_id is not None:
        raise ValueError("gate racing does not use wind or trajectory IDs")
    return _evaluate_metrics(run_dir, seeds=seeds,
                             perturb_initial_state=perturb_initial_state,
                             device=device)


def run_interactive(run_dir, *, device="cpu", split="test",
                    trajectory_id=None, windy=False, seed=0):
    if windy or trajectory_id is not None:
        raise ValueError("gate racing does not use wind or trajectory IDs")
    replay(run_dir, device=device, seed=seed)


def record_run(run_dir, output_path, *, device="cpu", fps=30., width=1280,
               height=720, split="test", trajectory_id=None, windy=False,
               seed=0):
    if windy or trajectory_id is not None:
        raise ValueError("gate racing does not use wind or trajectory IDs")
    replay(run_dir, device=device, seed=seed, output=output_path, fps=fps,
           width=width, height=height)
    return Path(output_path)


WORKFLOW = TaskWorkflow(
    name="gate_racing",
    settings_from=settings_from,
    train=dispatch_train,
    prepare=None,
    evaluate_metrics=evaluate_metrics,
    run_interactive=run_interactive,
    record_run=record_run,
    metric_wind_modes=("nominal", "both"),
    replay_wind_modes=("nominal", "both"),
)
