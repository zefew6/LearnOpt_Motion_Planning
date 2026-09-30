"""3-D A* guide using the ``pathfinding3d`` search implementation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.core.heuristic import octile
from pathfinding3d.finder.a_star import AStarFinder


@dataclass(frozen=True)
class AStarGuide:
    path: np.ndarray
    samples: np.ndarray
    low_clearance_samples: np.ndarray
    metrics: dict


def plan_aerial_astar_guide(
        start, goal, bounds, occupancy, esdf, *, proxy_radius, margin,
        grid_resolution=.08, fallback_resolution=.04,
        clearance_weight_m=.10, clearance_offset_m=.05,
        sample_spacing_m=.10, clearance_error_m=0., heuristic_weight=2.):
    """Plan a collision-checked base-center route to guide whole-body sampling."""
    started = time.perf_counter()
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    bounds = np.asarray(bounds, float)
    metrics = {
        "astar_seconds": 0.0,
        "astar_expansions": 0,
        "astar_grid_resolution": float(grid_resolution),
        "astar_route_length_m": None,
        "astar_minimum_clearance_m": None,
        "astar_fallback_used": False,
        "astar_heuristic_weight": float(heuristic_weight),
        "astar_failure_reason": None,
    }
    if (start.shape != (3,) or goal.shape != (3,) or bounds.shape != (2, 3)
            or not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal))
            or not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0])
            or np.any(goal < bounds[0]) or np.any(goal > bounds[1])
            or np.any(start < bounds[0]) or np.any(start > bounds[1])):
        metrics["astar_failure_reason"] = "endpoint_outside_workspace"
        metrics["astar_seconds"] = time.perf_counter()-started
        return None, metrics
    if not np.isfinite(heuristic_weight) or heuristic_weight < 1.:
        raise ValueError("heuristic_weight must be finite and at least one")
    if (not np.isfinite(grid_resolution) or grid_resolution <= 0.
            or not np.isfinite(fallback_resolution) or fallback_resolution <= 0.
            or not np.isfinite(proxy_radius) or proxy_radius < 0.
            or not np.isfinite(margin) or margin < 0.
            or not np.isfinite(clearance_weight_m) or clearance_weight_m < 0.
            or not np.isfinite(clearance_offset_m) or clearance_offset_m <= 0.
            or not np.isfinite(sample_spacing_m) or sample_spacing_m <= 0.
            or not np.isfinite(clearance_error_m) or clearance_error_m < 0.):
        raise ValueError("A* resolutions, collision values, and clearance settings are invalid")

    resolutions = [float(grid_resolution)]
    while resolutions:
        resolution = resolutions.pop(0)
        metrics["astar_grid_resolution"] = resolution
        try:
            route, expansions = _search_grid(
                start, goal, bounds, occupancy, esdf, proxy_radius, margin,
                resolution, clearance_weight_m, clearance_offset_m,
                clearance_error_m, heuristic_weight)
            metrics["astar_expansions"] += expansions
            if route is not None:
                route = _simplify(route, occupancy, proxy_radius, margin,
                                  occupancy.resolution*.5)
                samples, clearances = _resample(
                    route, esdf, proxy_radius+margin+clearance_error_m,
                    sample_spacing_m)
                metrics["astar_seconds"] = time.perf_counter()-started
                metrics["astar_route_length_m"] = float(np.sum(
                    np.linalg.norm(np.diff(route, axis=0), axis=1)))
                metrics["astar_minimum_clearance_m"] = float(np.min(clearances))
                metrics["astar_nodes"] = int(len(route))
                metrics["astar_samples"] = int(len(samples))
                order = np.argsort(clearances, kind="stable")
                low_count = max(1, int(np.ceil(len(samples)*.25)))
                guide = AStarGuide(route, samples, samples[order[:low_count]], metrics)
                return guide, metrics
        except (ValueError, IndexError, FloatingPointError) as error:
            metrics["astar_failure_reason"] = f"guide_error: {error}"
            break

        metrics["astar_failure_reason"] = "no_proxy_route"
        if (resolution == float(grid_resolution)
                and fallback_resolution < grid_resolution):
            metrics["astar_fallback_used"] = True
            resolutions.insert(0, float(fallback_resolution))

    metrics["astar_seconds"] = time.perf_counter()-started
    return None, metrics


def _search_grid(start, goal, bounds, occupancy, esdf, proxy_radius, margin,
                 resolution, clearance_weight, clearance_offset,
                 clearance_error, heuristic_weight):
    shape = np.floor((bounds[1]-bounds[0])/resolution+1e-9).astype(int)+1
    shape = tuple(int(value) for value in shape)
    matrix = occupancy.to_pathfinding3d_matrix(
        bounds[0], shape, resolution, radius=proxy_radius, margin=margin)
    matrix = _clearance_weighted_matrix(
        matrix, bounds[0], resolution, esdf, proxy_radius, margin,
        clearance_error, clearance_weight, clearance_offset)

    # Grid construction creates the library's node graph before the search.
    grid = Grid(matrix=matrix)
    starts = _connectable_nodes(start, bounds[0], resolution, shape,
                                matrix, occupancy, proxy_radius, margin)
    goals = _connectable_nodes(goal, bounds[0], resolution, shape,
                               matrix, occupancy, proxy_radius, margin)
    if not starts or not goals:
        return None, 0

    resolution_scale = 1000.*resolution
    heuristic = lambda dx, dy, dz: (  # noqa: E731
        resolution_scale*heuristic_weight*octile(dx, dy, dz))
    expansions = 0
    searched = False
    for start_index, start_node_position in starts:
        for goal_index, goal_node_position in goals:
            if searched:
                grid.cleanup()
            finder = AStarFinder(
                heuristic=heuristic,
                diagonal_movement=DiagonalMovement.only_when_no_obstacle,
                time_limit=float("inf"))
            searched = True
            nodes, runs = finder.find_path(
                grid.node(*start_index), grid.node(*goal_index), grid)
            expansions += int(runs)
            if not nodes:
                continue
            coordinates = np.asarray([
                bounds[0]+resolution*np.asarray(node.identifier, dtype=float)
                for node in nodes], dtype=float)
            route = np.vstack((start, coordinates, goal))
            route = route[np.r_[True, np.any(np.diff(route, axis=0) != 0., axis=1)]]
            if _route_edges_clear(route, occupancy, proxy_radius, margin):
                return route, expansions
    return None, expansions


def _clearance_weighted_matrix(matrix, lower, resolution, esdf, proxy_radius,
                               margin, clearance_error, clearance_weight,
                               clearance_offset):
    weighted = matrix.astype(np.int32)
    shape = matrix.shape
    yz_cells = int(shape[1]*shape[2])
    block_x = max(1, 2_000_000//yz_cells)
    y = lower[1]+np.arange(shape[1])*resolution
    z = lower[2]+np.arange(shape[2])*resolution
    scale = 1000.*resolution
    for first in range(0, shape[0], block_x):
        last = min(shape[0], first+block_x)
        free = matrix[first:last] != 0
        if not np.any(free):
            continue
        x = lower[0]+np.arange(first, last)*resolution
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        esdf_inside = np.all((points >= esdf.origin) & (points <= esdf.upper), axis=1)
        clearance = np.zeros(len(points), dtype=float)
        if np.any(esdf_inside):
            clearance[esdf_inside] = np.maximum(
                esdf.distance(points[esdf_inside])-proxy_radius-margin-clearance_error,
                0.)
        penalty = 1.+clearance_weight/(clearance+clearance_offset)
        costs = np.maximum(1, np.rint(scale*penalty)).astype(np.int32)
        costs[~free.ravel()] = 0
        weighted[first:last] = costs.reshape(last-first, shape[1], shape[2])
    return weighted


def _connectable_nodes(point, lower, resolution, shape, matrix, occupancy,
                       proxy_radius, margin):
    indices = _endpoint_candidates(point, lower, resolution, shape)
    candidates = []
    for index in indices:
        if matrix[index] <= 0:
            continue
        center = lower+resolution*np.asarray(index, dtype=float)
        if np.linalg.norm(center-point) <= 2.7*resolution+1e-9 and _segment_clear(
                point, center, occupancy, proxy_radius, margin):
            candidates.append((index, center))
    candidates.sort(key=lambda item: np.linalg.norm(item[1]-point))
    return candidates


def _endpoint_candidates(point, lower, resolution, shape):
    center = np.rint((point-lower)/resolution).astype(int)
    offsets = np.asarray([
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ], dtype=int)
    indices = center[None, :]+offsets
    inside = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    return [tuple(int(value) for value in row) for row in indices[inside]]


def _route_edges_clear(route, occupancy, proxy_radius, margin):
    for first, second in zip(route[:-1], route[1:], strict=True):
        if not _segment_clear(first, second, occupancy, proxy_radius, margin):
            return False
    return True


def _segment_clear(first, second, occupancy, proxy_radius, margin):
    first, second = np.asarray(first, float), np.asarray(second, float)
    midpoint = .5*(first+second)
    radius = proxy_radius+.5*np.linalg.norm(second-first)
    hits, outside = occupancy.collision_mask(
        midpoint.reshape(1, 3), np.asarray([radius]), margin)
    return not bool(hits[0] or outside[0])


def _simplify(route, occupancy, proxy_radius, margin, step):
    if len(route) <= 2:
        return route
    result = [route[0]]
    anchor = 0
    while anchor < len(route)-1:
        target = len(route)-1
        while target > anchor+1:
            delta = route[target]-route[anchor]
            count = max(1, int(np.ceil(np.linalg.norm(delta)/max(step, 1e-6))))
            points = route[anchor]+np.arange(1, count, dtype=float)[:, None]*delta/count
            if len(points):
                hits, outside = occupancy.collision_mask(
                    points, np.full(len(points), proxy_radius), margin)
                clear = not np.any(hits | outside)
            else:
                clear = True
            if clear:
                break
            target -= 1
        result.append(route[target])
        anchor = target
    return np.asarray(result)


def _resample(route, esdf, proxy_radius, spacing):
    deltas = np.diff(route, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] <= 1e-12:
        samples = route[:1].copy()
    else:
        distances = np.r_[np.arange(0., cumulative[-1], spacing), cumulative[-1]]
        segment_ids = np.minimum(np.searchsorted(cumulative[1:], distances, side="right"),
                                 len(lengths)-1)
        alpha = ((distances-cumulative[segment_ids])/
                 np.maximum(lengths[segment_ids], 1e-12))
        samples = route[segment_ids]+alpha[:, None]*deltas[segment_ids]
    clearances = esdf.distance(samples)-proxy_radius
    return samples, clearances


__all__ = ["AStarGuide", "plan_aerial_astar_guide"]
