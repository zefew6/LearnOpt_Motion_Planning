import pytest

from uav_ac.rl.mlp_baseline.training import load_training_config


def test_training_config_should_load_editable_yaml_values(tmp_path):
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
n_envs: 2
total_timesteps: 1000
ppo:
  n_steps: 128
  batch_size: 64
  learning_rate_start: 0.001
""",
        encoding="utf-8",
    )

    settings = load_training_config(config_path)

    assert settings["n_envs"] == 2
    assert settings["total_timesteps"] == 1000
    assert settings["ppo"]["n_steps"] == 128
    assert settings["ppo"]["batch_size"] == 64
    assert settings["ppo"]["learning_rate_start"] == pytest.approx(0.001)
    assert settings["ppo"]["net_arch"] == {"pi": [512, 512], "vf": [512, 512]}


def test_training_config_should_reject_batch_larger_than_rollout(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "n_envs: 1\nppo:\n  n_steps: 16\n  batch_size: 32\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size"):
        load_training_config(config_path)


def test_training_config_should_report_missing_explicit_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_config(tmp_path / "missing.yaml")
