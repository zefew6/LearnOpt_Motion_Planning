"""Configuration and checkpoint compatibility for gate racing."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from uav_ac.control.rl_controller import quad_parameters
from uav_ac.tasks.gate_racing import FEATURE_SCALES, SCENE_PATH
from uav_ac.tasks.gate_course import COURSE_DEFAULTS, COURSE_VERSION, course_settings
from ...acmpc.solver import MPCSettings, SOLVER_VERSION
from ...acmpc.racing_targets import RacingSettings
from ...acmpc.racing import COST_VERSION
from uav_ac.tasks.gate_course import vehicle_diameter
from ...common.config import COMMON_TRAINING_DEFAULTS, deep_merge


def settings_from(values):
    defaults = deepcopy(COMMON_TRAINING_DEFAULTS)
    defaults.update(task="gate_racing", scene="gate_racing", device="cpu", n_envs=4,
                    total_timesteps=8192, evaluation_interval=8192, checkpoint_interval=8192,
                    episode_seconds=30., evaluation_episodes=20, torch_threads=1,
                    mpc=asdict(MPCSettings(horizon_steps=50, dt=.02)),
                    racing={k:v for k,v in asdict(RacingSettings()).items() if k != "vehicle_diameter"},
                    course=deepcopy(COURSE_DEFAULTS))
    defaults["ppo"].update(n_steps=128, batch_size=64, n_epochs=3)
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"unknown gate-racing settings: {sorted(unknown)}")
    for key in ("ppo", "mpc", "racing"):
        if key in values and (not isinstance(values[key], dict) or set(values[key])-set(defaults[key])):
            raise ValueError(f"invalid gate-racing {key} settings")
    if "course" in values and not isinstance(values["course"], dict):
        raise ValueError("invalid course settings")
    course = course_settings(values.get("course", {}))
    settings = deep_merge(defaults, values)
    settings["course"] = course
    if settings["task"] != "gate_racing" or settings["policy_type"] not in {"mlp", "acmpc"}:
        raise ValueError("gate racing requires policy_type mlp or acmpc")
    scene = Path(settings["scene"])
    if scene.name != str(scene) or scene.suffix not in {"", ".xml"}:
        raise ValueError("scene must name an XML under simulation/models")
    if not scene_path(settings).is_file():
        raise FileNotFoundError(scene_path(settings))
    for key in ("seed", "n_envs", "total_timesteps", "steps_per_action", "evaluation_interval",
                "checkpoint_interval", "evaluation_episodes", "torch_threads"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"invalid gate-racing {key}")
    if not np.isfinite(settings["episode_seconds"]) or settings["episode_seconds"] <= 0:
        raise ValueError("episode_seconds must be positive and finite")
    ppo = settings["ppo"]
    for key in ("n_steps", "batch_size", "n_epochs"):
        if isinstance(ppo[key], bool) or not isinstance(ppo[key], int) or ppo[key] < 1:
            raise ValueError(f"invalid ppo.{key}")
    if not 1 < ppo["batch_size"] <= ppo["n_steps"] * settings["n_envs"]:
        raise ValueError("ppo.batch_size must be >1 and <= rollout size")
    for key in ("gamma", "gae_lambda", "clip_range", "ent_coef", "max_grad_norm",
                "learning_rate_start", "learning_rate_end"):
        if not np.isfinite(ppo[key]) or ppo[key] < 0:
            raise ValueError(f"invalid ppo.{key}")
    if not 0 < ppo["gamma"] <= 1 or not 0 <= ppo["gae_lambda"] <= 1:
        raise ValueError("invalid PPO discount parameters")
    if ppo["activation"] != "relu" or set(ppo["net_arch"]) != {"pi", "vf"}:
        raise ValueError("PPO requires relu and pi/vf architectures")
    if any(not widths or any(type(w) is not int or w < 1 for w in widths)
           for widths in ppo["net_arch"].values()):
        raise ValueError("invalid PPO network widths")
    for key in ("horizon_steps", "iterations", "retry_iterations", "chunk_size"):
        if type(settings["mpc"][key]) is not int:
            raise ValueError(f"mpc.{key} must be an integer")
    MPCSettings(**settings["mpc"])
    RacingSettings(**settings["racing"])
    return settings


def scene_path(settings):
    return SCENE_PATH.parent / (Path(settings["scene"]).stem + ".xml")


def contract(settings, env):
    return {"schema_version": 3, "task": "gate_racing", "task_version": 3,
            "racing_cost_version": COST_VERSION, "racing": settings["racing"],
            "vehicle_diameter": vehicle_diameter(env.unwrapped.simulation),
            "diameter_definition": "origin_centered_collision_bounding_sphere",
            "course_version": COURSE_VERSION, "course": settings["course"],
            "policy_type": settings["policy_type"], "scene": settings["scene"],
            "scene_sha256": hashlib.sha256(scene_path(settings).read_bytes()).hexdigest(),
            "feature_layout": "q_wxyz,v_body,omega_body,previous_action,two_gate_corners,two_valid",
            "feature_scales": FEATURE_SCALES.tolist(), "feature_clip": 10.,
            "action": "normalized_collective_thrust_body_moments_hover_zero",
            "physical_parameters": quad_parameters(env.unwrapped.quad),
            "control_dt": env.unwrapped.control_dt,
            "mpc": settings["mpc"] if settings["policy_type"] == "acmpc" else None,
            "solver_version": SOLVER_VERSION, "episode_seconds": settings["episode_seconds"]}


def read_run(run_dir):
    from .environment import make_environment

    metadata = json.loads((Path(run_dir) / "rl_config.json").read_text())
    if metadata["contract"].get("task_version") != 3:
        raise ValueError("incompatible gate-racing task version; train a new run with geometric MPC and diameter-relative gates")
    settings = settings_from(metadata["settings"])
    env = make_environment(settings)
    try:
        if metadata["contract"] != contract(settings, env):
            raise ValueError("incompatible gate-racing checkpoint contract or changed scene")
    finally:
        env.close()
    return metadata, settings
