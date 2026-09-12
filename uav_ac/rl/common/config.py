"""Configuration primitives shared by RL task workflows."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PPO_CONFIG: dict[str, Any] = {
    "n_steps": 4096,
    "batch_size": 4096,
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
}

COMMON_TRAINING_DEFAULTS: dict[str, Any] = {
    "policy_type": "mlp",
    "seed": 42,
    "device": "cuda",
    "n_envs": 24,
    "total_timesteps": 10_000_000,
    "steps_per_action": 10,
    "evaluation_interval": 100_000,
    "checkpoint_interval": 250_000,
    "ppo": DEFAULT_PPO_CONFIG,
}


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_mapping(path: str | Path, *, missing_ok: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        if missing_ok:
            return {}
        raise FileNotFoundError(f"training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError(f"training config must contain a YAML mapping: {path}")
    return values
