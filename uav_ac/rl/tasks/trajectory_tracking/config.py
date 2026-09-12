"""Trajectory-tracking training configuration."""

from pathlib import Path
from typing import Any

import numpy as np

from uav_ac.simulation.wind_disturb import RandomWindConfig
from ...acmpc.solver import MPCSettings
from ...common.config import COMMON_TRAINING_DEFAULTS, deep_merge, load_yaml_mapping

DEFAULT_TOTAL_TIMESTEPS = 10_000_000
DEFAULT_ENVIRONMENTS = 24
DEFAULT_SEED = 42
DEFAULT_STEPS_PER_ACTION = 10
EVALUATION_INTERVAL = 100_000
CHECKPOINT_INTERVAL = 250_000
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "ppo_trajectory.yaml"


DEFAULT_TRAINING_CONFIG: dict[str, Any] = deep_merge(COMMON_TRAINING_DEFAULTS, {
    "task": "trajectory_tracking",
    "trajectory_bank_path": None,
    "trajectory_bank": {
        "splits": {"train": 200, "validation": 20, "test": 20},
        "navigation_waypoints": [5, 10],
        "horizontal_bounds": [[-90.0, -90.0], [90.0, 90.0]],
        "altitude": [0.8, 4.5],
        "takeoff_altitude": 1.2,
        "segment_length": [12.0, 50.0],
        "path_length": [60.0, 300.0],
        "average_speed_bins": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "maximum_speed": 8.0,
        "length_per_piece": 6.0,
        "gcopter_max_acceleration": 6.0,
        "takeoff_max_speed": 3.0,
        "takeoff_max_acceleration": 3.0,
        "max_attempts": 20_000,
        "mpc_validation": {
            "horizon_steps": 10,
            "terminal_hold_seconds": 1.0,
            "position_rmse_max": 0.5,
            "position_error_max": 2.0,
            "final_position_error_max": 0.5,
            "final_velocity_error_max": 0.5,
            "maximum_tilt_degrees": 60.0,
        },
    },
    "wind": {
        "probability": 0.5,
        "steady_horizontal_max": 1.0,
        "gust_horizontal_max": 0.5,
        "gust_vertical_max": 0.1,
        "angular_frequency_min": 0.3,
        "angular_frequency_max": 1.5,
        "curriculum_fraction": 0.3,
    },
    "evaluation": {"panel_size": 5, "perturb_initial_state": True},
})


def settings_from(values: dict[str, Any]) -> dict[str, Any]:
    if values.get("task", "trajectory_tracking") != "trajectory_tracking":
        raise ValueError("trajectory-tracking workflow received another task")
    settings = deep_merge(DEFAULT_TRAINING_CONFIG, values)
    validate_training_config(settings)
    return settings


def load_training_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    values = load_yaml_mapping(path, missing_ok=config_path is None)
    return settings_from(values)


def validate_training_config(settings: dict[str, Any]) -> None:
    if settings["policy_type"] not in {"mlp", "acmpc"}:
        raise ValueError("policy_type must be mlp or acmpc")
    if settings["policy_type"] == "acmpc":
        MPCSettings(**settings.get("mpc", {}))
    ppo = settings["ppo"]
    required_positive = (
        "total_timesteps", "n_envs", "steps_per_action",
        "evaluation_interval", "checkpoint_interval",
    )
    if any(int(settings[key]) < 1 for key in required_positive):
        raise ValueError("training steps, environments, and intervals must be positive")
    for key in ("n_steps", "batch_size", "n_epochs"):
        if int(ppo[key]) < 1:
            raise ValueError(f"ppo.{key} must be positive")
    if int(ppo["batch_size"]) > int(ppo["n_steps"]) * int(settings["n_envs"]):
        raise ValueError("ppo.batch_size cannot exceed n_steps * n_envs")
    for key in (
        "gamma", "gae_lambda", "clip_range", "ent_coef", "max_grad_norm",
        "learning_rate_start", "learning_rate_end",
    ):
        if float(ppo[key]) < 0.0:
            raise ValueError(f"ppo.{key} must be non-negative")
    if str(ppo["activation"]).lower() != "relu":
        raise ValueError("only ReLU policies are supported")
    for branch in ("pi", "vf"):
        architecture = ppo["net_arch"].get(branch)
        if not architecture or any(int(width) < 1 for width in architecture):
            raise ValueError(f"ppo.net_arch.{branch} must contain positive layer widths")

    bank = settings["trajectory_bank"]
    if set(bank["splits"]) != {"train", "validation", "test"}:
        raise ValueError("trajectory_bank.splits must define train, validation, and test")
    if any(int(count) < 1 for count in bank["splits"].values()):
        raise ValueError("all trajectory-bank splits must be non-empty")
    waypoint_limits = np.asarray(bank["navigation_waypoints"], dtype=int)
    if waypoint_limits.shape != (2,) or waypoint_limits[0] < 1 or waypoint_limits[1] < waypoint_limits[0]:
        raise ValueError("trajectory_bank.navigation_waypoints must be increasing positive bounds")
    horizontal_bounds = np.asarray(bank["horizontal_bounds"], dtype=float)
    if horizontal_bounds.shape != (2, 2) or np.any(horizontal_bounds[1] <= horizontal_bounds[0]):
        raise ValueError("trajectory_bank.horizontal_bounds must have increasing 2D bounds")
    for name in ("altitude", "segment_length", "path_length"):
        bounds = np.asarray(bank[name], dtype=float)
        if bounds.shape != (2,) or bounds[0] <= 0.0 or bounds[1] <= bounds[0]:
            raise ValueError(f"trajectory_bank.{name} must have increasing positive bounds")
    speed_edges = np.asarray(bank["average_speed_bins"], dtype=float)
    if len(speed_edges) < 2 or speed_edges[0] <= 0.0 or np.any(np.diff(speed_edges) <= 0.0):
        raise ValueError("trajectory_bank.average_speed_bins must be increasing")
    if float(bank["maximum_speed"]) < speed_edges[-1]:
        raise ValueError("trajectory_bank.maximum_speed must cover all average-speed bins")
    if float(bank["length_per_piece"]) <= 0.0:
        raise ValueError("trajectory_bank.length_per_piece must be positive")
    if float(bank["gcopter_max_acceleration"]) <= 0.0:
        raise ValueError("trajectory_bank.gcopter_max_acceleration must be positive")
    if float(bank["takeoff_max_speed"]) <= 0.0 or float(bank["takeoff_max_acceleration"]) <= 0.0:
        raise ValueError("trajectory-bank takeoff limits must be positive")
    if int(bank["max_attempts"]) < sum(map(int, bank["splits"].values())):
        raise ValueError("trajectory_bank.max_attempts cannot be below the requested bank size")
    if int(settings["evaluation"]["panel_size"]) < 1:
        raise ValueError("evaluation.panel_size must be positive")
    RandomWindConfig(**settings["wind"])
