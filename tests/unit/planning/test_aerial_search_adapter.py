import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.trajectory.aerial_manipulator_minco.search_adapter import (
    AerialManipulatorStateSpaceAdapter,
    plan_aerial_astar_guide,
)


def _grid_with_wall_opening():
    occupied = np.zeros((21, 21, 21), dtype=bool)
    occupied[10, :, :] = True
    occupied[10, 5:16, 5:16] = False
    occupied[2, 2, 2] = True
    grid_map = GridMap(occupied, np.zeros(3), .05)
    return grid_map, ESDF.from_occupancy(grid_map)


def test_aerial_adapter_builds_guide_from_grid_map_and_esdf():
    grid_map, esdf = _grid_with_wall_opening()

    guide, metrics = plan_aerial_astar_guide(
        grid_map, np.array([.2, .25, .25]), np.array([.8, .25, .25]), esdf,
        proxy_radius=.02, margin=0., grid_resolution=.1,
        fallback_resolution=.05, sample_spacing_m=.1)

    assert guide is not None, metrics
    assert guide.path.shape[1] == 3
    assert len(guide.samples) >= 2
    assert metrics["astar_expansions"] > 0
    assert metrics["astar_route_length_m"] > .6
    for first, second in zip(guide.path[:-1], guide.path[1:], strict=True):
        count = max(2, int(np.ceil(np.linalg.norm(second-first)/.025)))
        points = first+np.arange(count+1)[:, None]*(second-first)/count
        hits, outside = grid_map.collision_mask(
            points, np.full(len(points), .02), 0.)
        assert not np.any(hits | outside)


def test_aerial_adapter_preserves_no_route_and_endpoint_diagnostics():
    occupied = np.zeros((9, 9, 9), dtype=bool)
    occupied[4, :, :] = True
    grid_map = GridMap(occupied, np.zeros(3), .1)
    esdf = ESDF.from_occupancy(grid_map)

    guide, metrics = plan_aerial_astar_guide(
        grid_map, np.array([.1, .2, .2]), np.array([.7, .2, .2]), esdf,
        proxy_radius=.01, margin=0., grid_resolution=.1,
        fallback_resolution=.05)

    assert guide is None
    assert metrics["astar_failure_reason"] == "no_proxy_route"

    guide, metrics = plan_aerial_astar_guide(
        grid_map, np.array([-1., .2, .2]), np.array([.7, .2, .2]), esdf,
        proxy_radius=.01, margin=0., grid_resolution=.1)
    assert guide is None
    assert metrics["astar_failure_reason"] == "endpoint_outside_workspace"


def test_aerial_state_space_adapter_round_trips_scaled_compound_state():
    adapter = AerialManipulatorStateSpaceAdapter()
    lower = np.r_[np.full(3, -2.), -np.pi, [-1., -2., -3., -4.]]
    upper = np.r_[np.full(3, 2.), np.pi, [1., 2., 3., 4.]]
    scales = np.r_[np.full(3, .5), .7, [.2, .5, 1., 2.]]
    space = adapter.create_space(lower, upper, scales)
    state = space.allocState()
    values = np.array([.5, -.4, .3, .7, .2, -1., 2., -3.])

    adapter.write_state(state, values)

    np.testing.assert_allclose(adapter.read_state(state), values, atol=1e-12)
    adapter.set_sampling_bounds(
        np.r_[-1., -1., -1., -.5, [-.5]*4],
        np.r_[1., 1., 1., .5, [.5]*4],
    )


def test_aerial_state_space_adapter_rejects_invalid_sampling_bounds():
    adapter = AerialManipulatorStateSpaceAdapter()
    lower = np.r_[np.full(3, -1.), -np.pi, np.full(4, -1.)]
    upper = np.r_[np.full(3, 1.), np.pi, np.full(4, 1.)]
    scales = np.r_[np.full(3, .5), .7, np.full(4, .8)]
    adapter.create_space(lower, upper, scales)

    with pytest.raises(ValueError, match="sampling bounds"):
        adapter.set_sampling_bounds(lower, np.r_[upper[:7], -1.])
