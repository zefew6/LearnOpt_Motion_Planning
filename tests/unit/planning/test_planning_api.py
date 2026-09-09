from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from uav_ac import runtime
from uav_ac.planning import plan
from uav_ac.planning import pipeline
from uav_ac.planning.pipeline import bmtp_mission


@pytest.fixture
def sim():
    return SimpleNamespace(
        start_position=np.array([0., 0., 0.]), goal_position=np.array([2., 0., -1.]),
        mission_waypoints=np.array([[9., 0., 0.], [0., 0., -1.], [2., 0., -1.]]),
        obstacles=np.empty((0, 6)), space_limits=np.array([[-5.] * 3, [5.] * 3]),
        gcs_guide_paths=[], get_planning_obstacle_points=Mock(),
        quad=SimpleNamespace(m=1., g=9.81, min_thrust=.1, max_thrust=5., max_tilt_angle=.5),
        model=SimpleNamespace(), data=SimpleNamespace(), _body_id=1,
        set_planning_paths=Mock(),
    )


@pytest.mark.parametrize("name", ["mini_snap", "gcopter", "gcs"])
def test_default_dispatch_preserves_inputs_and_layout(sim, monkeypatch, name):
    result = np.arange(40.).reshape(4, 10)
    dispatch = Mock(return_value=result)
    monkeypatch.setattr(runtime, "plan_trajectory", dispatch)
    config = {"name": name, "limits": {"velocity": 2.},
              "initial_path": {"source": "scene_waypoints"}}
    before = deepcopy(config)
    original = sim.mission_waypoints.copy()
    assert plan(config, sim, .02, visualize=True) is result
    args, kwargs = dispatch.call_args
    assert args == (name, sim, 2., .02)
    assert kwargs["visualize"] is True
    np.testing.assert_array_equal(kwargs["waypoints"][0], sim.start_position)
    np.testing.assert_array_equal(sim.mission_waypoints, original)
    assert config == before


def test_random_source_is_seeded(sim, monkeypatch):
    dispatch = Mock(return_value=np.zeros((2, 10)))
    monkeypatch.setattr(runtime, "plan_trajectory", dispatch)
    config = {"name": "mini_snap", "initial_path": {"source": "random_open_field"}}
    paths = []
    for seed in (42, 42, 43):
        plan(config, sim, .01, seed=seed)
        paths.append(dispatch.call_args.kwargs["waypoints"])
    np.testing.assert_array_equal(paths[0], paths[1])
    assert paths[0].shape != paths[2].shape or not np.allclose(paths[0], paths[2])


@pytest.mark.parametrize("name", ["mini_snap", "gcopter", "gcs", "bmtp"])
def test_unknown_native_option_fails_before_dispatch(sim, name):
    with pytest.raises(ValueError, match="unknown .* planner options.*typo"):
        plan({"name": name, "options": {"typo": 1}}, sim, .01)


@pytest.mark.parametrize("config,match", [
    ({"name": "unknown"}, "unsupported planner"),
    ({"name": "mini_snap", "limits": {"acceleration": 2}}, "unknown.*limits"),
    ({"name": "gcopter", "options": {"max_velocity": 4}}, "limits.velocity"),
    ({"name": "mini_snap", "initial_path": {"source": "scene_route"}}, "source"),
    ({"name": "bmtp", "initial_path": {"source": "scene_waypoints"}}, "scene_route"),
    ({"name": "gcs", "initial_path": {"index": 1}}, "unknown initial_path"),
    ({"name": "gcs", "options": None}, "options must be a dict"),
    ({"name": "mini_snap", "limits": {"velocity": float("nan")}}, "velocity"),
])
def test_invalid_config(sim, config, match):
    with pytest.raises(ValueError, match=match):
        plan(config, sim, .01)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_sample_period(sim, dt):
    with pytest.raises(ValueError, match="dt"):
        plan({"name": "mini_snap"}, sim, dt)


@pytest.mark.parametrize("name,attribute", [
    ("mini_snap", "mission_waypoints"), ("gcopter", "quad"),
    ("gcs", "gcs_guide_paths"), ("bmtp", "model"),
])
def test_scene_requirements(sim, name, attribute):
    delattr(sim, attribute)
    with pytest.raises(ValueError, match=f"scene requires:.*{attribute}"):
        plan({"name": name}, sim, .01)


def test_random_source_rejects_obstacles(sim):
    sim.obstacles = np.zeros((1, 6))
    with pytest.raises(ValueError, match="obstacle-free"):
        plan({"name": "mini_snap", "initial_path": {"source": "random_open_field"}}, sim, .01)


def test_gcs_native_options_are_applied(sim, monkeypatch):
    corridor, geometric = object(), object()
    monkeypatch.setattr(pipeline, "build_gcs_corridor", Mock(return_value=corridor))
    generate = Mock(return_value=geometric)
    monkeypatch.setattr(pipeline, "generate_gcs_trajectory", generate)
    sample = Mock(return_value=np.zeros((2, 10)))
    monkeypatch.setattr(pipeline, "gcs_controller_trajectory", sample)
    plan({"name": "gcs", "options": {"max_iterations": 17}}, sim, .02)
    assert generate.call_args.args[2].max_iterations == 17
    sample.assert_called_once_with(geometric, 3., .02)


def test_gcopter_native_options_are_applied(sim, monkeypatch):
    from uav_ac.planning.trajectory import gcopter

    corridor = SimpleNamespace(regions=[object()], fixed_boundaries=[(0, sim.start_position)])
    monkeypatch.setattr(pipeline, "build_mission_corridor", Mock(return_value=corridor))
    planner = Mock()
    constructor = Mock(return_value=planner)
    monkeypatch.setattr(gcopter, "GCOPTER", constructor)
    sample = Mock(return_value=np.zeros((2, 10)))
    monkeypatch.setattr(pipeline, "gcopter_controller_trajectory", sample)
    plan({"name": "gcopter", "limits": {"velocity": 2.},
          "options": {"max_iterations": 17}}, sim, .02)
    native = constructor.call_args.args[0]
    assert (native.max_iterations, native.max_velocity, native.mass) == (17, 2., sim.quad.m)
    assert native.length_per_piece == 1.5
    assert planner.plan.call_args.kwargs["fixed_corridor_boundaries"] is corridor.fixed_boundaries
    sample.assert_called_once_with(planner.plan.return_value, .02)


@pytest.fixture
def bmtp_mocks(sim, monkeypatch):
    samples = np.arange(27.).reshape(3, 9)
    trajectory = SimpleNamespace(sample=Mock(return_value=samples),
                                 control_points=np.zeros((1, 9, 3)), duration=1.234)
    result = SimpleNamespace(success=True, trajectory=trajectory, status="converged",
                             timings={"total_seconds": .1})
    planner = Mock(return_value=SimpleNamespace(plan=Mock(return_value=result)))
    monkeypatch.setattr(bmtp_mission, "BMTPPlanner", planner)
    monkeypatch.setattr(bmtp_mission, "scene_problem", Mock(return_value=("domain", [], .2)))
    monkeypatch.setattr(bmtp_mission, "scene_seed_paths", Mock(return_value=[
        np.array([[0., 0., 0.], [1., 0., -1.]]),
        np.array([[0., 0., 0.], [2., 0., -1.]]),
    ]))
    monkeypatch.setattr(bmtp_mission, "collisions", Mock(return_value=[]))
    return planner, result, samples


def test_bmtp_explicit_settings_no_yaml_or_retiming(sim, monkeypatch, bmtp_mocks):
    planner, result, samples = bmtp_mocks
    loader = Mock(side_effect=AssertionError("must not load YAML"))
    monkeypatch.setattr(bmtp_mission, "load_bmtp_settings", loader)
    config = {"name": "bmtp", "options": {"max_iterations": 7},
              "limits": {"velocity": 2., "acceleration": 4., "jerk": None, "snap": None},
              "initial_path": {"source": "scene_route", "index": 1, "segments": 3, "clearance": .25}}
    before = deepcopy(config)
    output = plan(config, sim, .07)
    assert output.shape == (3, 10)
    np.testing.assert_array_equal(output[:, :9], samples)
    result.trajectory.sample.assert_called_once_with(.07)
    assert result.trajectory.duration == 1.234
    assert sim.bmtp_result is result
    assert planner.call_args.args[0].max_iterations == 7
    path, _, _, limits = planner.return_value.plan.call_args.args
    assert path.shape == (4, 3)
    np.testing.assert_array_equal(path[-1], [2., 0., -1.])
    assert limits.values == (2., 4.)
    assert config == before
    loader.assert_not_called()


def test_legacy_bmtp_positional_config_path(sim, monkeypatch, bmtp_mocks):
    loader = Mock(return_value={"planner": {}, "limits_preset": "flight",
        "limits": {"flight": {"velocity": 2.}},
        "scene": {"initial_route": 0, "segments": 2, "clearance": .15}})
    monkeypatch.setattr(bmtp_mission, "load_bmtp_settings", loader)
    assert bmtp_mission.generate_bmtp_mission(sim, .01, "demo.yaml").shape == (3, 10)
    loader.assert_called_once_with("demo.yaml")


@pytest.mark.parametrize("initial,match", [
    ({"index": 8}, "outside scene routes"), ({"index": -1}, "initial_route"),
    ({"segments": 0}, "segments"), ({"segments": 2.5}, "segments"),
    ({"clearance": -.1}, "clearance"), ({"clearance": float("nan")}, "clearance"),
])
def test_bmtp_bad_route_settings(sim, bmtp_mocks, initial, match):
    with pytest.raises(ValueError, match=match):
        plan({"name": "bmtp", "initial_path": initial}, sim, .01)


def test_bmtp_rechecks_certification(sim, monkeypatch, bmtp_mocks):
    monkeypatch.setattr(bmtp_mission, "collisions", Mock(return_value=[object()]))
    with pytest.raises(RuntimeError, match="collision certification failed"):
        plan({"name": "bmtp"}, sim, .01)
    bmtp_mocks[1].trajectory.sample.assert_not_called()


def test_bmtp_failed_plan_is_not_sampled(sim, bmtp_mocks):
    result = bmtp_mocks[1]
    result.success = False
    result.message = "infeasible seed"
    with pytest.raises(RuntimeError, match="no certified executable trajectory"):
        plan({"name": "bmtp"}, sim, .01)
    result.trajectory.sample.assert_not_called()


def test_bmtp_missing_scene_routes(sim):
    sim.model.nsite = 0
    with pytest.raises(ValueError, match="scene requires.*bmtp_route"):
        bmtp_mission.scene_seed_paths(sim)


def test_bmtp_missing_vehicle_geometry(sim):
    sim.model = SimpleNamespace(geom_bodyid=np.array([1]),
                                geom_contype=np.array([0]), geom_conaffinity=np.array([0]))
    with pytest.raises(ValueError, match="collidable vehicle geometry"):
        bmtp_mission.scene_problem(sim, {"scene": {"clearance": .15}})


@pytest.mark.parametrize("waypoints", [[], [[0, 0, 0]], [[0, 0], [1, 1]],
                                       [[0, 0, 0], [0, 0, -1], [float("nan"), 0, -1]]])
def test_malformed_scene_waypoints(sim, waypoints):
    sim.mission_waypoints = waypoints
    with pytest.raises(ValueError, match="finite 3D waypoints"):
        plan({"name": "mini_snap"}, sim, .01)
