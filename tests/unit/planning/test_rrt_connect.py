import numpy as np
import pytest

from uav_ac.planning.search.rrt_connect import (
    RRTConnectPlanningError, plan_rrt_connect,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.search import (
    AerialManipulatorStateSpaceAdapter,
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
        AerialManipulatorStateSpaceAdapter(), start, goal, lower, upper, scale,
        state_valid=valid, edge_valid=edge_valid,
        sampling_regions=[
            {"name": "restricted", "lower": np.r_[-1., -.8, -.2, -np.pi, [-.2]*4],
             "upper": np.r_[1., -.4, .2, np.pi, [.2]*4], "fraction": .25},
            {"name": "global", "lower": lower, "upper": upper, "fraction": .75},
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
    adapter = AerialManipulatorStateSpaceAdapter()
    space = adapter.create_space(lower, upper, scales)
    origin, state = space.allocState(), space.allocState()
    adapter.write_state(origin, np.zeros(8))
    distances = []
    for index, offset in ((0, .5), (3, .7), (4, .2), (5, .5), (6, 1.), (7, 2.)):
        value = np.zeros(8)
        value[index] = offset
        adapter.write_state(state, value)
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
            AerialManipulatorStateSpaceAdapter(),
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


def test_shortcut_minimizes_route_cost_instead_of_farthest_node_index():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.search import _shortest_shortcut

    path = np.zeros((4, 8))
    path[:, :2] = [[0., 0.], [1., 0.], [0., 4.], [2., 0.]]
    allowed = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}

    def edge_valid(a, b):
        first = next(i for i, point in enumerate(path) if np.array_equal(point, a))
        last = next(i for i, point in enumerate(path) if np.array_equal(point, b))
        return (first, last) in allowed

    result, _ = _shortest_shortcut(path, edge_valid)
    np.testing.assert_array_equal(result, path[[0, 1, 3]])


def test_shortcut_uses_shortest_yaw_difference_when_comparing_routes():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.search import _shortest_shortcut

    path = np.zeros((4, 8))
    path[:, 0] = [0., .5, 1., 2.]
    path[:, 3] = [3.1, -3.1, 0., -3.0]
    result, _ = _shortest_shortcut(path, lambda a, b: not (a[0] == 0. and b[0] == 2.))
    np.testing.assert_array_equal(result[:, 0], [0., .5, 2.])
    assert np.sum(np.abs(np.diff(result[:, 3]))) < .2


def test_shortcut_counts_joint_motion_and_preserves_required_connections():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.search import _shortest_shortcut

    path = np.zeros((4, 8))
    path[:, 0] = [0., .5, 1., 2.]
    path[2, 4] = 2.
    result, _ = _shortest_shortcut(path, lambda a, b: not (a[0] == 0. and b[0] == 2.))
    np.testing.assert_array_equal(result, path[[0, 1, 3]])
    unchanged, _ = _shortest_shortcut(path, lambda a, b: (a[0], b[0]) in
                                    {(0., .5), (.5, 1.), (1., 2.)})
    np.testing.assert_array_equal(unchanged, path)


def test_partial_shortcut_removes_arm_detours_without_cutting_base_route():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.search import _partial_shortcut

    path = np.zeros((3, 8))
    path[:, :2] = [[0., 0.], [1., 1.], [2., 0.]]
    path[1, 3:] = [.8, 1., -.5, .3, .2]
    result, _ = _partial_shortcut(path, lambda a, b: True)
    np.testing.assert_array_equal(result[:, :3], path[:, :3])
    np.testing.assert_array_equal(result[:, 3:], np.zeros((3, 5)))
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])


def test_partial_shortcut_retains_joint_pose_needed_to_pass_obstacle():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.search import _partial_shortcut

    path = np.zeros((3, 8))
    path[:, 0] = [0., 1., 2.]
    path[1, 4] = 1.

    def edge_valid(a, b):
        # At the middle of the passage the arm must remain folded.
        return (a[0] != 1. or a[4] >= .9) and (b[0] != 1. or b[4] >= .9)

    result, _ = _partial_shortcut(path, edge_valid)
    np.testing.assert_array_equal(result, path)


def test_experimental_shortcut_requires_explicit_selection():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig

    assert AerialManipulatorMINCOConfig().shortcut_method == 'legacy'
    config = AerialManipulatorMINCOConfig.from_mapping({'shortcut_method': 'cost_partial'})
    assert config.shortcut_method == 'cost_partial'
    with pytest.raises(ValueError, match='shortcut_method'):
        AerialManipulatorMINCOConfig.from_mapping({'shortcut_method': 'unknown'})
