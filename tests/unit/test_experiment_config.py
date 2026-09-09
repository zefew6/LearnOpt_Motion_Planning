from pathlib import Path

import pytest

from uav_ac.experiments.config import load_config, read_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_example_configs_have_one_explicit_scene_and_no_overrides():
    for path in (ROOT / "configs" / "experiments").glob("*.yaml"):
        config = load_config(path)
        assert Path(config["scene"]["xml"]).is_file()
        assert "controller" in config["agent"]["type"] or config["agent"]["type"] == "rl"


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("mode: deploy\nmode: train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        read_yaml(path)


def test_controller_and_rl_branches_cannot_be_combined(tmp_path):
    path = tmp_path / "invalid.yaml"
    content = """mode: deploy
scene:
  config: PLACEHOLDER_SCENE
task:
  name: trajectory_tracking
  reference: {source: saved_trajectory, path: trajectory.npz}
planner: {name: gcopter, limits: {velocity: 3.0}, options: {}}
agent: {type: controller, name: cascaded, options: {}}
disturbance: {wind: {type: none}}
execution: {action_dt: 0.01}
output: {directory: runs/test}
""".replace("PLACEHOLDER_SCENE", str(ROOT / "configs/scenes/open_field.yaml"))
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="planner: inactive"):
        load_config(path)


def test_training_rejects_rendering_flags(tmp_path):
    path = ROOT / "configs" / "training" / "acmpc_open_field.yaml"
    values = path.read_text(encoding="utf-8")
    values = values.replace("../scenes/open_field.yaml", str(ROOT / "configs/scenes/open_field.yaml"))
    values = values.replace("viewer: false", "viewer: true")
    changed = tmp_path / "train.yaml"
    changed.write_text(values, encoding="utf-8")
    with pytest.raises(ValueError, match="rendering"):
        load_config(changed)
