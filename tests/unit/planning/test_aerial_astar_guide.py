import numpy as np

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.trajectory.aerial_manipulator_minco.search import (
    AerialAStarMaps, plan_aerial_astar_guide,
)


def _grid_with_wall_opening():
    occupied = np.zeros((21, 21, 21), dtype=bool)
    occupied[10, :, :] = True
    occupied[10, 5:16, 5:16] = False
    # Add another occupied voxel away from the route so the SDF has both signs.
    occupied[2, 2, 2] = True
    occupancy = GridMap(occupied, np.zeros(3), .05)
    return occupancy, ESDF.from_occupancy(occupancy)


def test_a_star_guidance_finishes_and_checks_simplified_edges():
    occupancy, esdf = _grid_with_wall_opening()
    maps = AerialAStarMaps(occupancy, esdf, proxy_radius=.02, margin=0.,
                          grid_resolution=.1, fallback_resolution=.05)
    guide, metrics = plan_aerial_astar_guide(
        maps, np.array([.2, .25, .25]), np.array([.8, .25, .25]), sample_spacing_m=.1)
    assert guide is not None, metrics
    assert guide.path.shape[1] == 3
    assert len(guide.samples) >= 2
    assert metrics["astar_expansions"] > 0
    assert metrics["astar_seconds"] >= 0.
    assert metrics["astar_route_length_m"] > .6
    for first, second in zip(guide.path[:-1], guide.path[1:], strict=True):
        count = max(2, int(np.ceil(np.linalg.norm(second-first)/.025)))
        points = first+np.arange(count+1)[:, None]*(second-first)/count
        hits, outside = occupancy.collision_mask(
            points, np.full(len(points), .02), 0.)
        assert not np.any(hits | outside)


def test_a_star_endpoint_inside_proxy_obstacle_returns_no_guide():
    occupancy, esdf = _grid_with_wall_opening()
    maps = AerialAStarMaps(occupancy, esdf, proxy_radius=.02, margin=0.,
                          grid_resolution=.1, fallback_resolution=.05)
    guide, metrics = plan_aerial_astar_guide(
        maps, np.array([.5, .25, .25]), np.array([.8, .25, .25]))
    assert guide is None
    assert metrics["astar_failure_reason"] == "no_proxy_route"


def test_initialized_a_star_maps_reused_without_map_construction(monkeypatch):
    from uav_ac.planning.trajectory.aerial_manipulator_minco import search as aerial_search

    occupancy, esdf = _grid_with_wall_opening()
    maps = aerial_search.AerialAStarMaps(
        occupancy, esdf, proxy_radius=.02, margin=0., grid_resolution=.1,
        fallback_resolution=.05)
    snapshots = [grid.occupancy.occupied.copy() for grid in maps.grids]

    def unexpected(*args, **kwargs):
        raise AssertionError('search rebuilt a static map')

    monkeypatch.setattr(GridMap, '__post_init__', unexpected)
    monkeypatch.setattr(GridMap, 'to_pathfinding3d_matrix', unexpected)
    monkeypatch.setattr(ESDF, 'from_occupancy', unexpected)
    for start, goal in (([.2, .5, .5], [.8, .5, .5]),
                        ([.8, .5, .5], [.2, .5, .5])):
        guide, metrics = plan_aerial_astar_guide(maps, start, goal)
        assert guide is not None, metrics
        np.testing.assert_array_equal(guide.path[0], start)
        np.testing.assert_array_equal(guide.path[-1], goal)
        for first, second in zip(guide.path[:-1], guide.path[1:], strict=True):
            points = np.linspace(first, second, 101)
            hits, outside = occupancy.collision_mask(points, np.full(101, .02), 0.)
            assert not np.any(hits | outside)
    for grid, snapshot in zip(maps.grids, snapshots, strict=True):
        np.testing.assert_array_equal(grid.occupancy.occupied, snapshot)
        assert not grid.traversal_cost.flags.writeable


def test_fallback_search_uses_preinitialized_fine_grid(monkeypatch):
    from types import SimpleNamespace
    from uav_ac.planning.trajectory.aerial_manipulator_minco import search as aerial_search

    occupancy, esdf = _grid_with_wall_opening()
    maps = aerial_search.AerialAStarMaps(
        occupancy, esdf, proxy_radius=.02, margin=0., grid_resolution=.1,
        fallback_resolution=.05)
    original_search = aerial_search.astar_search
    seen = []

    def search(grid, *args, **kwargs):
        seen.append(grid)
        if grid.resolution == .1:
            return SimpleNamespace(found=False, path=None, expansions=3)
        return original_search(grid, *args, **kwargs)

    monkeypatch.setattr(aerial_search, 'astar_search', search)
    monkeypatch.setattr(GridMap, '__post_init__', lambda *_: (_ for _ in ()).throw(
        AssertionError('fallback rebuilt a map')))
    guide, metrics = plan_aerial_astar_guide(maps, [.2, .5, .5], [.8, .5, .5])
    assert guide is not None, metrics
    assert metrics['astar_fallback_used']
    assert metrics['astar_grid_resolution'] == .05
    assert len(seen) == 2
    assert all(actual is expected.occupancy for actual, expected in zip(
        seen, maps.grids, strict=True))
