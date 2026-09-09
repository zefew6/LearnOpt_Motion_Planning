"""Compatibility and scene routing at the shared training boundary."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from uav_ac.rl import training
from uav_ac.rl.common.trajectory_bank import trajectory_bank_fingerprint
from uav_ac.rl.mlp_baseline import training as legacy
from uav_ac.rl.mlp_baseline import train as legacy_cli


def test_legacy_training_exports_shared_implementation():
    assert legacy.train is training.train
    assert legacy_cli.train is training.train
    assert legacy.prepare_assets is training.prepare_assets
    assert legacy._validate_training_config is training._validate_training_config
    assert legacy.DEFAULT_CONFIG_PATH == training.DEFAULT_CONFIG_PATH
    assert training.DEFAULT_CONFIG_PATH.name == "ppo_trajectory.yaml"
    assert training.DEFAULT_CONFIG_PATH.parent.name == "configs"
    assert training.DEFAULT_CONFIG_PATH == Path(legacy.__file__).resolve().parents[3] / "configs" / "ppo_trajectory.yaml"


def test_settings_reject_config_path_before_creating_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="mutually exclusive"):
        training.train(run_dir, settings={}, config_path=tmp_path / "absent.yaml")
    assert not run_dir.exists()


def test_settings_cannot_implicitly_generate_bank(tmp_path):
    with pytest.raises(ValueError, match="existing trajectory_bank_path"):
        training.train(tmp_path / "run", settings=deepcopy(training.DEFAULT_TRAINING_CONFIG))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("matching_scene", [True, False])
def test_external_bank_checks_selected_scene(tmp_path, monkeypatch, matching_scene):
    scene = tmp_path / "scene.xml"
    scene.write_text("selected scene", encoding="utf-8")
    settings = deepcopy(training.DEFAULT_TRAINING_CONFIG)
    settings["trajectory_bank_path"] = str(tmp_path / "bank")
    fingerprint_scene = scene if matching_scene else training.OPEN_FIELD_SCENE_PATH
    bank = SimpleNamespace(metadata={"fingerprint": trajectory_bank_fingerprint(
        settings["trajectory_bank"], scene_path=fingerprint_scene, steps_per_action=10)})
    monkeypatch.setattr(training.TrajectoryBank, "load", Mock(return_value=bank))
    generate = Mock(side_effect=AssertionError("must not generate an external bank"))
    monkeypatch.setattr(training, "generate_trajectory_bank", generate)
    simulation = Mock(return_value=SimpleNamespace(quad=object()))
    monkeypatch.setattr(training, "MujocoSimulation", simulation)
    monkeypatch.setattr(training, "quad_parameters", Mock(return_value={"physics_dt": 0.001}))
    if matching_scene:
        assert training.prepare_assets(tmp_path, settings=settings, model_path=scene)[0] is bank
        simulation.assert_called_once_with(scene, record_actual_trajectory=False)
    else:
        with pytest.raises(ValueError, match="fingerprint"):
            training.prepare_assets(tmp_path, settings=settings, model_path=scene)
        simulation.assert_not_called()
    generate.assert_not_called()


def test_legacy_asset_generation_passes_selected_scene(tmp_path, monkeypatch):
    scene = tmp_path / "scene.xml"
    generate = Mock(return_value=object())
    monkeypatch.setattr(training, "generate_trajectory_bank", generate)
    simulation = Mock(return_value=SimpleNamespace(quad=object()))
    monkeypatch.setattr(training, "MujocoSimulation", simulation)
    monkeypatch.setattr(training, "quad_parameters", Mock(return_value={}))
    training.prepare_assets(tmp_path, model_path=scene)
    assert generate.call_args.kwargs["scene_path"] == scene
    simulation.assert_called_once_with(scene, record_actual_trajectory=False)


def test_worker_factory_passes_selected_scene(tmp_path, monkeypatch):
    scene = tmp_path / "scene.xml"
    bank = object()
    monkeypatch.setattr(training.TrajectoryBank, "load", Mock(return_value=bank))
    environment = Mock()
    monkeypatch.setattr(training, "MujocoTrajectoryTrackingEnv", environment)
    monitor = Mock()
    monkeypatch.setattr(training, "Monitor", monitor)
    factory = training.make_environment_factory(
        tmp_path / "bank", tmp_path / "monitor", rank=2, steps_per_action=10,
        wind_settings=training.DEFAULT_TRAINING_CONFIG["wind"], model_path=scene)
    assert factory() is monitor.return_value
    assert environment.call_args.args == (bank,)
    assert environment.call_args.kwargs["model_path"] == scene


@pytest.mark.parametrize("policy_type", ["mlp", "acmpc"])
def test_training_routes_scene_and_preserves_artifacts(tmp_path, monkeypatch, policy_type):
    settings = deepcopy(training.DEFAULT_TRAINING_CONFIG)
    settings.update(policy_type=policy_type, n_envs=1, total_timesteps=8, device="cpu",
                    trajectory_bank_path=str(tmp_path / "bank"))
    settings["ppo"].update(n_steps=8, batch_size=8)
    original = deepcopy(settings)
    scene = tmp_path / "selected.xml"
    bank = SimpleNamespace(indices=lambda split: [0])
    prepare = Mock(return_value=(bank, {"physics_dt": 0.001}))
    monkeypatch.setattr(training, "prepare_assets", prepare)
    monkeypatch.setattr(training, "load_training_config", Mock(side_effect=AssertionError("must use settings")))
    environment = Mock()
    monkeypatch.setattr(training, "MujocoTrajectoryTrackingEnv", environment)
    monkeypatch.setattr(training, "check_env", Mock())
    factory = Mock()
    monkeypatch.setattr(training, "make_environment_factory", factory)
    vector = Mock()
    monkeypatch.setattr(training, "DummyVecEnv", Mock(return_value=vector))
    model = Mock(policy=SimpleNamespace())
    ppo = Mock(return_value=model)
    monkeypatch.setattr(training, "PPO", ppo)
    run_dir = tmp_path / "run"
    assert training.train(run_dir, settings=settings, model_path=scene) == run_dir
    assert settings == original
    assert prepare.call_args.kwargs["model_path"] == scene
    assert factory.call_args.kwargs["model_path"] == scene
    assert len(environment.call_args_list) == 2  # check and evaluation environments
    assert all(call.kwargs["model_path"] == scene for call in environment.call_args_list)
    configuration = json.loads((run_dir / training.RL_CONFIG_FILENAME).read_text())
    assert configuration["scene"] == scene.name
    assert configuration["policy_type"] == policy_type
    assert configuration["trajectory_bank_directory"] == settings["trajectory_bank_path"]
    assert configuration["control_dt"] == pytest.approx(0.01)
    assert all((run_dir / name).is_dir() for name in ("monitor", "checkpoints", "tensorboard"))
    assert [call.args[0].name for call in model.save.call_args_list] == ["final_model", "best_model"]
    checkpoint = model.learn.call_args.kwargs["callback"].callbacks[-1]
    assert checkpoint.name_prefix == "ppo_trajectory"
    vector.close.assert_called_once()
    if policy_type == "acmpc":
        from uav_ac.rl.acmpc.policy import ACMPCPolicy
        assert ppo.call_args.args[0] is ACMPCPolicy
        assert ACMPCPolicy.__module__ == "uav_ac.rl.acmpc.policy"
        assert model.policy.solver_strict is True
    else:
        assert ppo.call_args.args[0] == "MlpPolicy"


def test_acmpc_evaluation_panel_uses_selected_scene(tmp_path, monkeypatch):
    from uav_ac.rl.acmpc import benchmark

    scene = tmp_path / "selected.xml"
    bank = SimpleNamespace(indices=lambda split: [0, 1])
    environments = []

    def make_env(bank, **kwargs):
        env = SimpleNamespace(trajectory_bank=bank, close=Mock(), **kwargs)
        environments.append(env)
        return env

    monkeypatch.setattr(training, "MujocoTrajectoryTrackingEnv", make_env)
    collect = Mock(return_value=[{"success": True, "position_rmse": 0.1, "return": 1.0}])
    monkeypatch.setattr(benchmark, "collect_panel", collect)
    callback = training.TrajectoryBankEvaluationCallback(
        bank, tmp_path, model_path=scene, steps_per_action=10,
        wind_config=training.RandomWindConfig(), eval_freq=1, panel_size=2,
        perturb_initial_state=True, seed=42, observation_mode="acmpc")
    callback.model = Mock()
    assert callback._on_step()
    assert len(environments) == 5
    assert all(env.model_path == scene for env in environments)
    for env in environments[1:]:
        env.close.assert_called_once()
    assert (tmp_path / "validation_history.jsonl").is_file()
    callback.model.save.assert_called_once_with(tmp_path / "best_model")
    callback._on_training_end()
    environments[0].close.assert_called_once()
