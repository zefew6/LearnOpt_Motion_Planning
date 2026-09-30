"""GridMap-based A* search."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.core.heuristic import octile
from pathfinding3d.finder.a_star import AStarFinder

from ..geometry.grid_map import GridMap


@dataclass(frozen=True)
class AStarResult:
    """Result of a voxel-grid search in world coordinates."""

    path: np.ndarray | None
    found: bool
    expansions: int
    metrics: dict


def astar_search(
        grid_map: GridMap,
        start_world: np.ndarray,
        goal_world: np.ndarray,
        *,
        traversal_cost: np.ndarray | None = None,
        heuristic_weight: float = 1.0,
) -> AStarResult:
    """Search a ``GridMap`` and return a world-coordinate path.

    Occupied voxels are blocked.  ``traversal_cost`` may assign positive
    integer-like costs to free voxels while retaining zero for blocked voxels.
    The search backend is deliberately unaware of the source of either map or
    cost array.
    """
    started = time.perf_counter()
    if not isinstance(grid_map, GridMap):
        raise TypeError("grid_map must be a GridMap")
    if (not np.isfinite(heuristic_weight) or heuristic_weight < 1.0):
        raise ValueError("heuristic_weight must be finite and at least one")

    cost = _validate_traversal_cost(traversal_cost, grid_map)
    start = _validate_point(start_world, "start_world")
    goal = _validate_point(goal_world, "goal_world")
    metrics = {
        "search_seconds": 0.0,
        "expansions": 0,
        "failure_reason": None,
        "heuristic_weight": float(heuristic_weight),
    }

    start_status, start_candidates = _connectable_indices(start, grid_map)
    goal_status, goal_candidates = _connectable_indices(goal, grid_map)
    if start_status == "outside" or goal_status == "outside":
        metrics["failure_reason"] = "endpoint_outside_grid"
        return _result(None, metrics, started)
    if start_status == "occupied" or goal_status == "occupied":
        metrics["failure_reason"] = "endpoint_occupied"
        return _result(None, metrics, started)

    if np.array_equal(start, goal):
        path = np.vstack((start,))
        return _result(path, metrics, started, found=True)

    matrix = (~grid_map.occupied).astype(np.int32)
    if cost is not None:
        matrix = cost.copy()
        matrix[grid_map.occupied] = 0
    grid = Grid(matrix=matrix)
    resolution_scale = 1000.0*grid_map.resolution
    heuristic = lambda dx, dy, dz: (  # noqa: E731
        resolution_scale*heuristic_weight*octile(dx, dy, dz))

    searched = False
    for start_index, start_center in start_candidates:
        for goal_index, goal_center in goal_candidates:
            if searched:
                grid.cleanup()
            finder = AStarFinder(
                heuristic=heuristic,
                diagonal_movement=DiagonalMovement.only_when_no_obstacle,
                time_limit=float("inf"),
            )
            searched = True
            nodes, runs = finder.find_path(
                grid.node(*start_index), grid.node(*goal_index), grid)
            metrics["expansions"] += int(runs)
            if not nodes:
                continue
            voxel_path = np.asarray([
                grid_map.index_to_world(np.asarray(node.identifier, dtype=int))
                for node in nodes
            ])
            path = np.vstack((start, voxel_path, goal))
            path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0.0, axis=1)]]
            return _result(path, metrics, started, found=True)

    metrics["failure_reason"] = "no_path"
    return _result(None, metrics, started)


def _validate_traversal_cost(cost, grid_map: GridMap):
    if cost is None:
        return None
    cost = np.asarray(cost)
    if cost.shape != grid_map.shape:
        raise ValueError("traversal_cost must have the same shape as grid_map")
    if not np.all(np.isfinite(cost)):
        raise ValueError("traversal_cost must be finite")
    if np.any(cost[~grid_map.occupied] <= 0.0):
        raise ValueError("traversal_cost must be positive on free voxels")
    if np.any(cost[~grid_map.occupied] != np.rint(cost[~grid_map.occupied])):
        raise ValueError("traversal_cost must contain integer-like values")
    return np.rint(cost).astype(np.int32)


def _validate_point(point, name):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return point


def _connectable_indices(point: np.ndarray, grid_map: GridMap):
    center = grid_map.world_to_index(point)
    shape = np.asarray(grid_map.shape)
    if np.any(center < 0) or np.any(center >= shape):
        return "outside", []
    center_tuple = tuple(int(value) for value in center)
    if grid_map.occupied[center_tuple]:
        return "occupied", []
    offsets = np.asarray([
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ], dtype=int)
    indices = center[None, :] + offsets
    inside = np.all((indices >= 0) & (indices < shape), axis=1)
    candidates = []
    for index in indices[inside]:
        index_tuple = tuple(int(value) for value in index)
        if grid_map.occupied[index_tuple]:
            continue
        world = grid_map.index_to_world(index)
        candidates.append((index_tuple, world))
    candidates.sort(key=lambda item: float(np.linalg.norm(item[1]-point)))
    return "ok", candidates


def _result(path, metrics, started, *, found=False):
    metrics["search_seconds"] = time.perf_counter()-started
    return AStarResult(path, bool(found), int(metrics["expansions"]), dict(metrics))


__all__ = ["AStarResult", "astar_search"]
