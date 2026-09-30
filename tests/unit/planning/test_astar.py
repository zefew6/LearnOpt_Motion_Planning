import numpy as np
import pytest

from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.search.astar import AStarResult, astar_search


def _map_with_open_wall():
    occupied = np.zeros((11, 11, 5), dtype=bool)
    occupied[5, :, :] = True
    occupied[5, 5, :] = False
    return GridMap(occupied, np.zeros(3), 1.0)


def test_astar_search_uses_grid_map_and_returns_world_path_through_opening():
    result = astar_search(_map_with_open_wall(), [1., 2., 2.], [9., 2., 2.])

    assert isinstance(result, AStarResult)
    assert result.found
    assert result.path[0] == pytest.approx([1., 2., 2.])
    assert result.path[-1] == pytest.approx([9., 2., 2.])
    assert result.expansions > 0


def test_astar_search_reports_unreachable_grid_without_aerial_inputs():
    occupied = np.zeros((7, 7, 3), dtype=bool)
    occupied[3, :, :] = True
    result = astar_search(GridMap(occupied, np.zeros(3), 1.0), [1., 1., 1.], [5., 1., 1.])

    assert not result.found
    assert result.path is None
    assert result.metrics["failure_reason"] == "no_path"


@pytest.mark.parametrize("point", [[-1., 1., 1.], [1., 1., 3.]])
def test_astar_search_reports_endpoint_outside_grid(point):
    grid_map = GridMap(np.zeros((3, 3, 3), dtype=bool), np.zeros(3), 1.0)

    result = astar_search(grid_map, point, [1., 1., 1.])

    assert not result.found
    assert result.metrics["failure_reason"] == "endpoint_outside_grid"


def test_astar_search_rejects_endpoint_inside_occupied_voxel():
    occupied = np.zeros((3, 3, 3), dtype=bool)
    occupied[1, 1, 1] = True

    result = astar_search(GridMap(occupied, np.zeros(3), 1.0),
                          [1., 1., 1.], [2., 2., 2.])

    assert not result.found
    assert result.metrics["failure_reason"] == "endpoint_occupied"


def test_astar_search_rejects_traversal_cost_with_wrong_shape():
    grid_map = GridMap(np.zeros((3, 3, 3), dtype=bool), np.zeros(3), 1.0)

    with pytest.raises(ValueError, match="traversal_cost"):
        astar_search(grid_map, [0., 0., 0.], [2., 2., 2.],
                     traversal_cost=np.ones((3, 3, 2)))
