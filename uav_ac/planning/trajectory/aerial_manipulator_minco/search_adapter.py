"""Aerial-manipulator conversions for the generic search algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from ompl import base as ob

from ...geometry.grid_map import GridMap
from ...search.astar import astar_search


@dataclass(frozen=True)
class AStarGuide:
    path: np.ndarray
    samples: np.ndarray
    low_clearance_samples: np.ndarray
    metrics: dict


def plan_aerial_astar_guide(
        grid_map, start, goal, esdf, *, proxy_radius, margin,
        grid_resolution=.08, fallback_resolution=.04,
        clearance_weight_m=.10, clearance_offset_m=.05,
        sample_spacing_m=.10, clearance_error_m=0., heuristic_weight=2.):
    """Build an aerial guide by adapting a scene ``GridMap`` to generic A*."""
    started = time.perf_counter()
    if not isinstance(grid_map, GridMap):
        raise TypeError("grid_map must be a GridMap")
    start, goal = _point(start), _point(goal)
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
    if (start is None or goal is None
            or np.any(start < grid_map.origin) or np.any(start > grid_map.upper)
            or np.any(goal < grid_map.origin) or np.any(goal > grid_map.upper)):
        metrics["astar_failure_reason"] = "endpoint_outside_workspace"
        metrics["astar_seconds"] = time.perf_counter()-started
        return None, metrics
    if (not np.isfinite(heuristic_weight) or heuristic_weight < 1.
            or not np.isfinite(grid_resolution) or grid_resolution <= 0.
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
                start, goal, grid_map, esdf, proxy_radius, margin,
                resolution, clearance_weight_m, clearance_offset_m,
                clearance_error_m, heuristic_weight)
            metrics["astar_expansions"] += expansions
            if route is not None:
                route = _simplify(route, grid_map, proxy_radius, margin,
                                  grid_map.resolution*.5)
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


def _search_grid(start, goal, grid_map, esdf, proxy_radius, margin,
                 resolution, clearance_weight, clearance_offset,
                 clearance_error, heuristic_weight):
    shape = np.floor((grid_map.upper-grid_map.origin)/resolution+1e-9).astype(int)+1
    shape = tuple(int(value) for value in shape)
    matrix = grid_map.to_pathfinding3d_matrix(
        grid_map.origin, shape, resolution, radius=proxy_radius, margin=margin)
    inflated_map = GridMap(matrix == 0, grid_map.origin, resolution)
    prepared = _prepare_endpoint_map(
        inflated_map, grid_map, (start, goal), proxy_radius, margin)
    if prepared is None:
        return None, 0
    inflated_map, search_start, search_goal = prepared
    cost = _clearance_weighted_matrix(
        matrix, grid_map.origin, resolution, esdf, proxy_radius, margin,
        clearance_error, clearance_weight, clearance_offset)
    result = astar_search(
        inflated_map, search_start, search_goal, traversal_cost=cost,
        heuristic_weight=heuristic_weight)
    if not result.found or result.path is None:
        return None, result.expansions
    route = np.vstack((start, result.path, goal))
    route = route[np.r_[True, np.any(np.diff(route, axis=0) != 0., axis=1)]]
    if not _route_edges_clear(route, grid_map, proxy_radius, margin):
        return None, result.expansions
    return route, result.expansions


def _prepare_endpoint_map(inflated_map, source_map, endpoints, radius, margin):
    """Choose coarse free cells that conservatively connect to exact endpoints."""
    points = np.asarray(endpoints, dtype=float)
    occupied = inflated_map.occupied.copy()
    search_points = []
    shape = np.asarray(inflated_map.shape)
    offsets = np.asarray([
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ], dtype=int)
    for point in points:
        center = inflated_map.world_to_index(point)
        indices = center[None, :]+offsets
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        candidates = []
        for index in indices[inside]:
            index_tuple = tuple(int(value) for value in index)
            if occupied[index_tuple]:
                continue
            world = inflated_map.index_to_world(index)
            if _segment_clear(point, world, source_map, radius, margin):
                candidates.append((float(np.linalg.norm(world-point)), index_tuple, world))
        if not candidates:
            return None
        _, index_tuple, world = min(candidates, key=lambda item: item[0])
        search_points.append(world)
    return (GridMap(occupied, inflated_map.origin, inflated_map.resolution),
            search_points[0], search_points[1])


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
        inside = np.all((points >= esdf.origin) & (points <= esdf.upper), axis=1)
        clearance = np.zeros(len(points), dtype=float)
        if np.any(inside):
            clearance[inside] = np.maximum(
                esdf.distance(points[inside])-proxy_radius-margin-clearance_error,
                0.)
        penalty = 1.+clearance_weight/(clearance+clearance_offset)
        costs = np.maximum(1, np.rint(scale*penalty)).astype(np.int32)
        costs[~free.ravel()] = 0
        weighted[first:last] = costs.reshape(last-first, shape[1], shape[2])
    return weighted


def _route_edges_clear(route, grid_map, proxy_radius, margin):
    return all(
        _segment_clear(first, second, grid_map, proxy_radius, margin)
        for first, second in zip(route[:-1], route[1:], strict=True))


def _segment_clear(first, second, grid_map, proxy_radius, margin):
    first, second = np.asarray(first, float), np.asarray(second, float)
    midpoint = .5*(first+second)
    radius = proxy_radius+.5*np.linalg.norm(second-first)
    hits, outside = grid_map.collision_mask(
        midpoint.reshape(1, 3), np.asarray([radius]), margin)
    return not bool(hits[0] or outside[0])


def _simplify(route, grid_map, proxy_radius, margin, step):
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
                hits, outside = grid_map.collision_mask(
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
        segment_ids = np.minimum(
            np.searchsorted(cumulative[1:], distances, side="right"),
            len(lengths)-1)
        alpha = ((distances-cumulative[segment_ids])/
                 np.maximum(lengths[segment_ids], 1e-12))
        samples = route[segment_ids]+alpha[:, None]*deltas[segment_ids]
    clearances = esdf.distance(samples)-proxy_radius
    return samples, clearances


def _point(point):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point


class AerialManipulatorStateSpaceAdapter:
    """OMPL adapter for the aerial manipulator's compound state space."""

    def __init__(self):
        self._joint_scales = None
        self._position_space = None
        self._joint_space = None
        self.space = None

    def create_space(self, lower, upper, metric_scale):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        metric_scale = np.asarray(metric_scale, float)
        if (lower.shape != (8,) or upper.shape != (8,)
                or metric_scale.shape != (8,) or np.any(upper <= lower)
                or np.any(metric_scale <= 0.) or not np.all(np.isfinite(metric_scale))):
            raise ValueError("aerial state-space bounds and scales must be finite ordered 8-vectors")
        self._joint_scales = metric_scale[4:8].copy()
        self.space = ob.CompoundStateSpace()
        position = ob.RealVectorStateSpace(3)
        self._position_space = position
        self._set_real_bounds(position, lower[:3], upper[:3])
        yaw = ob.SO2StateSpace()
        joints = ob.RealVectorStateSpace(4)
        self._joint_space = joints
        self._set_real_bounds(joints, lower[4:8]/self._joint_scales,
                              upper[4:8]/self._joint_scales)
        self.space.addSubspace(position, float(1./metric_scale[0]))
        self.space.addSubspace(yaw, float(1./metric_scale[3]))
        self.space.addSubspace(joints, 1.0)
        return self.space

    def read_state(self, state):
        return np.r_[np.asarray([state[0][index] for index in range(3)], dtype=float),
                     float(state[1].value),
                     np.asarray([state[2][index] for index in range(4)], dtype=float)
                     *self._joint_scales]

    def write_state(self, state, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError("aerial OMPL state must be a finite 8-vector")
        for index in range(3):
            state[0][index] = float(values[index])
        state[1].value = float(values[3])
        for index in range(4):
            state[2][index] = float(values[4+index]/self._joint_scales[index])

    def set_sampling_bounds(self, lower, upper):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (lower.shape != (8,) or upper.shape != (8,)
                or np.any(upper <= lower) or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))):
            raise ValueError("aerial sampling bounds must be ordered finite 8-vectors")
        self._set_real_bounds(self._position_space, lower[:3], upper[:3])
        self._set_real_bounds(self._joint_space,
                              lower[4:8]/self._joint_scales,
                              upper[4:8]/self._joint_scales)

    @staticmethod
    def _set_real_bounds(space, lower, upper):
        bounds = ob.RealVectorBounds(len(lower))
        for index, (low, high) in enumerate(zip(lower, upper, strict=True)):
            bounds.setLow(index, float(low))
            bounds.setHigh(index, float(high))
        space.setBounds(bounds)


__all__ = [
    "AStarGuide",
    "AerialManipulatorStateSpaceAdapter",
    "plan_aerial_astar_guide",
]
