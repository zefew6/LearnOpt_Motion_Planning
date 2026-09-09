from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import numpy as np
import pytest

from uav_ac import main


def write_config(tmp_path, text):
    path = tmp_path / "flight.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_flight_config_is_small_and_resolves_scene():
    config = main.load_config()
    assert Path(config["scene"]).is_file()
    assert config["planner"] in main.PLANNERS
    assert config["controller"] in main.CONTROLLERS
    assert config["wind"] == "none"


def test_minimal_config_receives_runtime_defaults(tmp_path):
    config = main.load_config(write_config(
        tmp_path, "scene: lab_course\nplanner: mini_snap\ncontroller: cascaded\n"))
    assert config["speed"] == 3.0
    assert config["control_dt"] == 0.01
    assert config["visualize"] is False
    assert config["follow_camera"] is False
    assert config["gcopter"] == {}


def test_follow_camera_is_loaded_from_flight_yaml(tmp_path):
    config = main.load_config(write_config(
        tmp_path, "scene: lab_course\nplanner: mini_snap\ncontroller: cascaded\nfollow_camera: true\n"))
    assert config["follow_camera"] is True


@pytest.mark.parametrize("text,match", [
    ("scene: lab_course\nscene: open_field\nplanner: gcopter\ncontroller: cascaded\n",
     "duplicate YAML key"),
    ("scene: lab_course\nplanner: wrong\ncontroller: cascaded\n", "planner must"),
    ("scene: lab_course\nplanner: gcopter\ncontroller: wrong\n", "controller must"),
    ("scene: open_field\nplanner: gcopter\ncontroller: rl\nrl: {checkpoint: model.zip}\ncontrol_dt: 0.01\n",
     "control_dt is owned"),
    ("scene: ../lab_course\nplanner: gcopter\ncontroller: cascaded\n", "scene must"),
])
def test_invalid_compact_config_fails_locally(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        main.load_config(write_config(tmp_path, text))


def test_inactive_component_blocks_are_retained_but_ignored(tmp_path):
    config = main.load_config(write_config(tmp_path, """\
scene: lab_course
planner: gcopter
controller: cascaded
wind: none
bmtp: {segments: 8}
mpc: {horizon_steps: 10}
rl: {checkpoint: missing.zip, device: cuda}
wind_options: {steady_force: [3.0, 0.0, 0.0]}
"""))
    assert config["bmtp"]["segments"] == 8
    assert config["rl"]["checkpoint"] == "missing.zip"


def test_rl_requires_explicit_zip_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="rl.checkpoint is required"):
        main.load_config(write_config(
            tmp_path, "scene: open_field\nplanner: gcopter\ncontroller: rl\n"))


def test_cascaded_controller_uses_selected_period():
    simulation = SimpleNamespace(quad=SimpleNamespace(g=9.81, dt=0.001))
    controller, dt = main.build_controller({
        "controller": "cascaded", "control_dt": 0.02, "mpc": {}}, simulation)
    assert dt == 0.02
    assert controller.dt == 0.02


def test_rl_controller_owns_control_period(monkeypatch):
    loaded = SimpleNamespace(control_dt=0.025)
    loader = Mock(return_value=loaded)
    monkeypatch.setattr(main.RLController, "from_checkpoint", loader)
    simulation = SimpleNamespace(quad=object())
    controller, dt = main.build_controller({
        "controller": "rl", "rl": {"checkpoint": "model.zip", "device": "cuda"}}, simulation)
    assert controller is loaded
    assert dt == 0.025
    loader.assert_called_once_with("model.zip", simulation.quad, device="cuda")


def test_run_does_not_require_or_write_an_output_directory(monkeypatch):
    trajectory = np.zeros((2, 10))
    trajectory[1, 0] = 1.0
    tracker = Mock()
    simulation = SimpleNamespace(
        quad=SimpleNamespace(dt=0.001, position=np.zeros(3)),
        goal_position=np.zeros(3), collision_detected=False,
        set_trajectory_visualization=Mock(), run_interactive=Mock(),
        set_external_force_world=Mock(),
    )
    monkeypatch.setattr(main, "MujocoSimulation", Mock(return_value=simulation))
    monkeypatch.setattr(main, "build_controller", Mock(return_value=(object(), 0.01)))
    monkeypatch.setattr(main, "plan_trajectory", Mock(return_value=trajectory))
    monkeypatch.setattr(main, "TrajectoryController", Mock(return_value=tracker))
    config = {"scene": str(main.MODEL_DIRECTORY / "lab_course.xml"),
              "planner": "gcopter", "controller": "cascaded", "visualize": False,
              "wind": "none", "wind_options": {}}
    assert main.run(config) is trajectory
    simulation.run_interactive.assert_called_once_with(
        ANY, ANY, chase_camera=False)
