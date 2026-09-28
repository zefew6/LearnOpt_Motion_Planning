import numpy as np
import pytest

from uav_ac.planning.search.rrt_connect import RRTConnect
from uav_ac.planning.trajectory.gcopter.aerial_manipulator.planner import (
    _resample_preserving_corners, _unwrap_path_yaw,
)


def _planner(obstacle=True, seed=3):
    def interpolate(a, b, alpha):
        delta = b-a
        delta[1] = (delta[1]+np.pi) % (2*np.pi)-np.pi
        result = a+alpha*delta
        result[1] = (result[1]+np.pi) % (2*np.pi)-np.pi
        return result

    def metric(a, b):
        delta = b-a
        delta[1] = (delta[1]+np.pi) % (2*np.pi)-np.pi
        return np.linalg.norm(delta)

    valid = lambda x: not obstacle or not (abs(x[0]) < .2 and abs(x[2]) < .35)

    def edge(a, b):
        length = metric(a, b)
        count = max(1, int(np.ceil(length/.025)))
        return all(valid(interpolate(a, b, t)) for t in np.linspace(0, 1, count+1))

    return RRTConnect(np.array([-1., -np.pi, -1.]), np.array([1., np.pi, 1.]),
                      metric, interpolate, valid, edge, .2,
                      max_iterations=2500, rng=np.random.default_rng(seed))


def test_rrt_connect_returns_valid_start_to_goal_path_and_simplifies():
    planner = _planner()
    start, goal = np.array([-.8, 3.1, 0.]), np.array([.8, -3.1, 0.])
    path = planner.plan(start, goal)
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert all(planner.edge_valid_fn(a, b) for a, b in zip(path[:-1], path[1:]))
    short = planner.simplify(path)
    assert len(short) <= len(path)


def test_rrt_connect_rejects_invalid_endpoints():
    planner = _planner()
    with pytest.raises(ValueError, match="valid states"):
        planner.plan(np.array([0., 0., 0.]), np.array([.8, 0., 0.]))


def test_yaw_unwrap_and_resampling_preserve_path_turns():
    path = np.array([[0., 0., 0., 3.1, 0., 0., 0., 0.],
                     [.1, 0., 0., -3.1, .1, 0., 0., 0.],
                     [.1, .2, 0., -3.0, .1, .1, 0., 0.]])
    unwrapped = _unwrap_path_yaw(path)
    assert np.max(np.abs(np.diff(unwrapped[:, 3]))) < .2
    knots = _resample_preserving_corners(unwrapped, pieces=5)
    assert any(np.allclose(corner, knot) for corner in unwrapped[1:-1]
               for knot in knots)
