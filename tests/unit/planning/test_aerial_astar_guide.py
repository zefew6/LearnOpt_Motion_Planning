import numpy as np

from uav_ac.planning.geometry.esdf import ESDF, InflatedOccupancyGrid
from uav_ac.planning.search.aerial_astar_guide import plan_aerial_astar_guide


def _grid_with_wall_opening():
    occupied = np.zeros((21, 21, 21), dtype=bool)
    occupied[10, :, :] = True
    occupied[10, 5:16, 5:16] = False
    # Add another occupied voxel away from the route so the SDF has both signs.
    occupied[2, 2, 2] = True
    occupancy = InflatedOccupancyGrid(occupied, np.zeros(3), .05)
    return occupancy, ESDF.from_occupancy(occupancy)


def test_a_star_guidance_routes_through_wall_opening_and_checks_simplified_edges():
    occupancy, esdf = _grid_with_wall_opening()
    guide, metrics = plan_aerial_astar_guide(
        np.array([.2, .25, .25]), np.array([.8, .25, .25]),
        np.array([[0., 0., 0.], [1., 1., 1.]]), occupancy, esdf,
        proxy_radius=.02, margin=0., grid_resolution=.1,
        fallback_resolution=.05, budget_s=1., sample_spacing_m=.1)
    assert guide is not None, metrics
    assert guide.path.shape[1] == 3
    assert len(guide.samples) >= 2
    assert metrics["astar_expansions"] > 0
    assert metrics["astar_route_length_m"] > .6
    for first, second in zip(guide.path[:-1], guide.path[1:], strict=True):
        count = max(2, int(np.ceil(np.linalg.norm(second-first)/.025)))
        points = first+np.arange(count+1)[:, None]*(second-first)/count
        hits, outside = occupancy.collision_mask(
            points, np.full(len(points), .02), 0.)
        assert not np.any(hits | outside)


def test_a_star_endpoint_inside_proxy_obstacle_returns_no_guide():
    occupancy, esdf = _grid_with_wall_opening()
    guide, metrics = plan_aerial_astar_guide(
        np.array([.5, .25, .25]), np.array([.8, .25, .25]),
        np.array([[0., 0., 0.], [1., 1., 1.]]), occupancy, esdf,
        proxy_radius=.02, margin=0., grid_resolution=.1,
        fallback_resolution=.05, budget_s=.1)
    assert guide is None
    assert metrics["astar_failure_reason"] in {"no_proxy_route", "budget_exceeded"}


def test_a_star_zero_budget_falls_back_without_throwing():
    occupancy, esdf = _grid_with_wall_opening()
    guide, metrics = plan_aerial_astar_guide(
        np.array([.2, .25, .25]), np.array([.8, .25, .25]),
        np.array([[0., 0., 0.], [1., 1., 1.]]), occupancy, esdf,
        proxy_radius=.02, margin=0., grid_resolution=.1,
        fallback_resolution=.05, budget_s=0.)
    assert guide is None
    assert metrics["astar_failure_reason"] == "budget_exceeded"
