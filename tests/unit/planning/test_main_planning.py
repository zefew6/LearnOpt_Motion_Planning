from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from uav_ac import main


def config(planner, **options):
    return {
        "planner": planner, "speed": 2.0, "visualize": False, "seed": 7,
        "bmtp": {}, "gcopter": {}, "gcs": {}, **options,
    }


def simulation():
    return SimpleNamespace(
        start_position=np.array([0.0, 0.0, 0.0]),
        goal_position=np.array([2.0, 0.0, -1.0]),
        mission_waypoints=np.array([
            [9.0, 0.0, 0.0], [0.0, 0.0, -1.0], [2.0, 0.0, -1.0]]),
        obstacles=np.empty((0, 6)),
        quad=SimpleNamespace(
            m=1.0, g=9.81, dt=0.001, min_thrust=0.1,
            max_thrust=5.0, max_tilt_angle=0.5),
    )


def test_minimum_snap_receives_scene_waypoints_without_mutating_them(monkeypatch):
    sim = simulation()
    original = sim.mission_waypoints.copy()
    expected = np.zeros((2, 10))
    generate = Mock(return_value=expected)
    monkeypatch.setattr(main, "generate_minimum_snap_mission", generate)
    assert main.plan_trajectory(config("mini_snap"), sim, 0.01) is expected
    np.testing.assert_array_equal(generate.call_args.args[0][0], sim.start_position)
    np.testing.assert_array_equal(sim.mission_waypoints, original)


def test_gcopter_uses_vehicle_physics_and_optional_settings(monkeypatch):
    sim = simulation()
    corridor = SimpleNamespace(regions=[object()], fixed_boundaries=[])
    build_corridor = Mock(return_value=corridor)
    monkeypatch.setattr(main, "build_mission_corridor", build_corridor)
    planner = Mock()
    constructor = Mock(return_value=planner)
    monkeypatch.setattr("uav_ac.planning.trajectory.gcopter.GCOPTER", constructor)
    sampled = np.zeros((2, 10))
    monkeypatch.setattr(main, "gcopter_controller_trajectory", Mock(return_value=sampled))
    settings = config("gcopter", gcopter={"max_iterations": 17})
    assert main.plan_trajectory(settings, sim, 0.01) is sampled
    native = constructor.call_args.args[0]
    assert native.max_velocity == 2.0
    assert native.mass == sim.quad.m
    assert native.max_iterations == 17
    np.testing.assert_array_equal(
        build_corridor.call_args.args[1],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [2.0, 0.0, -1.0]]),
    )


def test_gcs_applies_native_options(monkeypatch):
    sim = simulation()
    corridor, geometric = object(), object()
    monkeypatch.setattr(main, "build_gcs_corridor", Mock(return_value=corridor))
    generate = Mock(return_value=geometric)
    monkeypatch.setattr(main, "generate_gcs_trajectory", generate)
    sampled = np.zeros((2, 10))
    monkeypatch.setattr(main, "gcs_controller_trajectory", Mock(return_value=sampled))
    assert main.plan_trajectory(
        config("gcs", gcs={"max_iterations": 17}), sim, 0.02) is sampled
    assert generate.call_args.args[2].max_iterations == 17
    main.gcs_controller_trajectory.assert_called_once_with(geometric, 2.0, 0.02)


def test_bmtp_builds_explicit_settings_without_loading_another_yaml(monkeypatch):
    sim = simulation()
    sampled = np.zeros((2, 10))
    generate = Mock(return_value=sampled)
    monkeypatch.setattr(
        "uav_ac.planning.pipeline.bmtp_mission.generate_bmtp_mission", generate)
    settings = config("bmtp", bmtp={
        "initial_route": 2, "segments": 9, "clearance": 0.2,
        "acceleration": 4.0, "max_iterations": 12,
    })
    assert main.plan_trajectory(settings, sim, 0.01) is sampled
    passed = generate.call_args.kwargs["settings"]
    assert generate.call_args.kwargs["visualize"] is True
    assert passed["scene"] == {
        "initial_route": 2, "segments": 9, "clearance": 0.2}
    assert passed["limits"]["velocity"] == 2.0
    assert passed["limits"]["acceleration"] == 4.0
    assert passed["planner"]["max_iterations"] == 12
