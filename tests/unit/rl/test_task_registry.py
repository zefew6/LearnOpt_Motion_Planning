"""Explicit RL task registration and compatibility routing."""

import json

import pytest

from uav_ac.rl import evaluate, training
from uav_ac.rl.common import registry
from uav_ac.rl.mlp_baseline import evaluate as legacy_evaluate


def test_registry_contains_unique_supported_tasks():
    tasks = registry.registered_tasks()
    assert tasks == ("trajectory_tracking", "gate_racing")
    assert len(tasks) == len(set(tasks))
    assert tuple(registry.get_workflow(name).name for name in tasks) == tasks


def test_duplicate_registry_entry_is_rejected():
    with pytest.raises(RuntimeError, match="duplicate RL task workflow"):
        registry._build_registry((("same", "first"), ("same", "second")))


def test_missing_task_metadata_defaults_to_trajectory_tracking(tmp_path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "rl_config.json").write_text(json.dumps({"schema_version": 1}))
    assert registry.task_name({}) == "trajectory_tracking"
    assert registry.task_name_from_run(run_dir) == "trajectory_tracking"


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown training task"):
        registry.get_workflow("unknown")
    with pytest.raises(ValueError, match="unknown training task"):
        registry.task_name({"task": "unknown"})


def test_invalid_registry_target_is_rejected(monkeypatch):
    monkeypatch.setitem(registry._TASK_MODULES, "broken", "uav_ac.rl")
    with pytest.raises(RuntimeError, match="invalid RL task workflow"):
        registry.get_workflow("broken")


def test_shipped_configs_select_explicit_tasks():
    assert training.load_training_config("configs/ppo_trajectory.yaml")["task"] == "trajectory_tracking"
    assert training.load_training_config("configs/ppo_gate_racing.yaml")["task"] == "gate_racing"


def test_legacy_evaluation_exports_unified_entrypoint():
    assert legacy_evaluate.evaluate_metrics is evaluate.evaluate_metrics
    assert legacy_evaluate.run_interactive is evaluate.run_interactive
    assert legacy_evaluate.record_run is evaluate.record_run
