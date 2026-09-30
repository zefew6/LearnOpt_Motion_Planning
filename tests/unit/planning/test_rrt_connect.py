import numpy as np
import pytest

from uav_ac.planning.search.RRT_connect import (
    RRTConnectPlanningError, _AerialStateSpace, _write_state, plan_rrt_connect,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.planner import (
    _resample_preserving_corners, _unwrap_path_yaw,
)


def test_ompl_rrt_connect_returns_exact_valid_8d_route_around_wall():
    lower = np.r_[-1., -1., -1., -np.pi, np.full(4, -1.)]
    upper = np.r_[1., 1., 1., np.pi, np.full(4, 1.)]
    scale = np.r_[np.full(3, .5), .7, np.full(4, .8)]
    start, goal = np.zeros(8), np.zeros(8)
    start[:3] = [-.8, -.6, 0.]
    goal[:3] = [.8, -.6, 0.]
    def valid(state):
        x, y, z = state[:3]
        return not (-.15 < x < .15 and -1. < y < .2 and -.8 < z < .8)

    def edge_valid(first, second):
        delta = second-first
        delta[3] = (delta[3]+np.pi)%(2*np.pi)-np.pi
        count = max(1, int(np.ceil(np.linalg.norm(delta[:3])/.04)),
                    int(np.ceil(abs(delta[3])/.08)),
                    int(np.ceil(np.max(np.abs(delta[4:]))/.08)))
        for alpha in np.arange(1, count+1)/count:
            state = first+alpha*delta
            state[3] = (state[3]+np.pi)%(2*np.pi)-np.pi
            if not valid(state):
                return False
        return True

    path, metrics = plan_rrt_connect(
        start, goal, lower, upper, scale,
        state_valid=valid, edge_valid=edge_valid,
        sampling_regions=[
            {"name": "restricted", "lower": [-1., -.8, -.2],
             "upper": [1., -.4, .2], "joint_lower": [-.2]*4,
             "joint_upper": [.2]*4, "fraction": .25},
            {"name": "global", "lower": lower[:3],
             "upper": upper[:3], "fraction": .75},
        ],
        range_size=.25, timeout_s=3.)
    assert path.shape[1] == 8
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_array_equal(path[-1], goal)
    assert all(edge_valid(a, b) for a, b in zip(path[:-1], path[1:], strict=True))
    assert metrics["rrt_nodes"] > 2
    assert metrics["rrt_motion_checks"] > 0
    assert set(metrics["rrt_sampling_stage_seconds"]) == {"restricted", "global"}


def test_yaw_unwrap_and_distance_resampling_preserve_every_corner():
    path = np.array([[0., 0., 0., 3.1, 0., 0., 0., 0.],
                     [.1, 0., 0., -3.1, .1, 0., 0., 0.],
                     [.1, .2, 0., -3.0, .1, .1, 0., 0.]])
    unwrapped = _unwrap_path_yaw(path)
    assert np.max(np.abs(np.diff(unwrapped[:, 3]))) < .2
    knots = _resample_preserving_corners(unwrapped, spacing=.25)
    assert all(any(np.array_equal(corner, knot) for knot in knots)
               for corner in unwrapped)
    assert knots[0, 3] == unwrapped[0, 3]
    np.testing.assert_array_equal(knots[-1], unwrapped[-1])
    assert len(knots) >= 3


def test_ompl_compound_metric_normalizes_heterogeneous_dimensions():
    lower = np.r_[np.full(3, -2.), -np.pi, [-1., -2., -3., -4.]]
    upper = np.r_[np.full(3, 2.), np.pi, [1., 2., 3., 4.]]
    scales = np.r_[np.full(3, .5), .7, [.2, .5, 1., 2.]]
    space = _AerialStateSpace(lower, upper, scales)
    origin, state = space.allocState(), space.allocState()
    _write_state(origin, np.zeros(8), scales[4:8])
    distances = []
    for index, offset in ((0, .5), (3, .7), (4, .2), (5, .5), (6, 1.), (7, 2.)):
        value = np.zeros(8)
        value[index] = offset
        _write_state(state, value, scales[4:8])
        distances.append(space.distance(origin, state))
    np.testing.assert_allclose(distances, np.ones(6), atol=1e-12)


def test_ompl_callback_timeout_preserves_search_diagnostics():
    lower = np.r_[np.full(3, -1.), -np.pi, np.full(4, -1.)]
    upper = np.r_[np.full(3, 1.), np.pi, np.full(4, 1.)]
    scales = np.r_[np.full(3, .5), .7, np.full(4, .8)]
    calls = 0

    def valid(_state):
        nonlocal calls
        calls += 1
        if calls > 6:
            raise TimeoutError("test deadline")
        return True

    with pytest.raises(RRTConnectPlanningError, match="test deadline") as caught:
        plan_rrt_connect(
            np.zeros(8), np.r_[.8, np.zeros(7)], lower, upper, scales,
            state_valid=valid, edge_valid=lambda *_: True,
            range_size=.2, timeout_s=1.)
    assert caught.value.metrics["rrt_nodes"] >= 2
    assert caught.value.metrics["rrt_sampling_stage_seconds"]


def test_distance_resampling_counts_joint_and_in_place_yaw_motion():
    path = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1.2, .9, 0., 0., 0.]])
    knots = _resample_preserving_corners(path, spacing=.25)
    assert len(knots) > 2
    np.testing.assert_array_equal(knots[0], path[0])
    np.testing.assert_array_equal(knots[-1], path[-1])
