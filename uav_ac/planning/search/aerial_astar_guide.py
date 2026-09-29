"""Fast 3-D, optimistic A* routes used only to guide whole-body sampling."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np


@dataclass(frozen=True)
class AStarGuide:
    path: np.ndarray
    samples: np.ndarray
    low_clearance_samples: np.ndarray
    metrics: dict


class _AStarTimeout(TimeoutError):
    def __init__(self, expansions):
        super().__init__("A* guide exceeded its time budget")
        self.expansions = int(expansions)


_NEIGHBORS = np.asarray([
    (x, y, z)
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
    if (x, y, z) != (0, 0, 0)
], dtype=int)


def plan_aerial_astar_guide(
        start, goal, bounds, occupancy, esdf, *, proxy_radius, margin,
        grid_resolution=.08, fallback_resolution=.04, budget_s=.10,
        clearance_weight_m=.10, clearance_offset_m=.05,
        sample_spacing_m=.10, clearance_error_m=0., heuristic_weight=2.):
    """Plan and simplify an optimistic base-center route within a time budget.

    The route is a sampling hint only. All returned 8-D motion still passes
    the aerial manipulator's complete state and edge collision validators.
    """
    started = time.perf_counter()
    deadline = started+float(budget_s)
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
            or np.any(goal < bounds[0]) or np.any(goal > bounds[1])
            or np.any(start < bounds[0]) or np.any(start > bounds[1])):
        metrics["astar_failure_reason"] = "endpoint_outside_workspace"
        metrics["astar_seconds"] = time.perf_counter()-started
        return None, metrics
    if not np.isfinite(heuristic_weight) or heuristic_weight < 1.:
        raise ValueError("heuristic_weight must be finite and at least one")

    resolutions = [float(grid_resolution)]
    while resolutions and time.perf_counter() < deadline:
        resolution = resolutions.pop(0)
        metrics["astar_grid_resolution"] = resolution
        attempt_started = time.perf_counter()
        try:
            route, expansions = _search_grid(
                start, goal, bounds, occupancy, esdf, proxy_radius, margin,
                resolution, deadline, clearance_weight_m, clearance_offset_m,
                clearance_error_m, heuristic_weight)
            metrics["astar_expansions"] += expansions
            if route is not None:
                route = _simplify(route, occupancy, proxy_radius, margin,
                                  occupancy.resolution*.5, deadline)
                samples, clearances = _resample(
                    route, esdf, proxy_radius+margin+clearance_error_m,
                    sample_spacing_m)
                if time.perf_counter() > deadline:
                    metrics["astar_failure_reason"] = "budget_exceeded"
                    break
                metrics["astar_seconds"] = time.perf_counter()-started
                metrics["astar_route_length_m"] = float(np.sum(
                    np.linalg.norm(np.diff(route, axis=0), axis=1)))
                metrics["astar_minimum_clearance_m"] = float(np.min(clearances))
                metrics["astar_nodes"] = int(len(route))
                metrics["astar_samples"] = int(len(samples))
                order = np.argsort(clearances, kind="stable")
                low_count = max(1, int(np.ceil(len(samples)*.25)))
                return AStarGuide(route, samples, samples[order[:low_count]], metrics), metrics
        except _AStarTimeout as error:
            metrics["astar_expansions"] += error.expansions
            metrics["astar_failure_reason"] = "budget_exceeded"
            break
        except TimeoutError:
            metrics["astar_failure_reason"] = "budget_exceeded"
            break
        except (ValueError, IndexError, FloatingPointError) as error:
            metrics["astar_failure_reason"] = f"guide_error: {error}"
            break
        metrics["astar_failure_reason"] = "no_proxy_route"
        if (resolution == float(grid_resolution)
                and fallback_resolution < grid_resolution
                and time.perf_counter() < deadline):
            metrics["astar_fallback_used"] = True
            resolutions.insert(0, float(fallback_resolution))
        elif time.perf_counter() >= deadline:
            metrics["astar_failure_reason"] = "budget_exceeded"
            break
        if time.perf_counter() <= attempt_started:
            break
    metrics["astar_seconds"] = time.perf_counter()-started
    if metrics["astar_failure_reason"] is None:
        metrics["astar_failure_reason"] = "budget_exceeded"
    return None, metrics


def _search_grid(start, goal, bounds, occupancy, esdf, proxy_radius, margin,
                 resolution, deadline, clearance_weight, clearance_offset,
                 clearance_error, heuristic_weight):
    shape = np.floor((bounds[1]-bounds[0])/resolution+1e-9).astype(int)+1
    axes = [bounds[0, axis]+np.arange(shape[axis])*resolution for axis in range(3)]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    blocked, outside = occupancy.collision_mask(
        coordinates, np.full(len(coordinates), proxy_radius), margin)
    free = ~(blocked | outside)
    esdf_upper = esdf.upper
    esdf_inside = np.all((coordinates >= esdf.origin) & (coordinates <= esdf_upper), axis=1)
    clearance = np.zeros(len(coordinates), dtype=float)
    if np.any(esdf_inside):
        esdf_indices = np.rint(
            (coordinates[esdf_inside]-esdf.origin)/esdf.resolution).astype(int)
        nearest_distances = esdf.values[
            esdf_indices[:, 0], esdf_indices[:, 1], esdf_indices[:, 2]]
        # Nearest stored voxels can overestimate the true distance. The
        # configured discretization margin covers voxelization and query
        # quantization error before this clearance affects edge costs.
        clearance[esdf_inside] = np.maximum(
            nearest_distances-proxy_radius-margin-clearance_error, 0.)
    if time.perf_counter() >= deadline:
        raise _AStarTimeout(0)

    endpoint_collision, endpoint_outside = occupancy.collision_mask(
        np.vstack((start, goal)), np.full(2, proxy_radius), margin)
    if np.any(endpoint_collision | endpoint_outside):
        return None, 0

    start_candidates = _endpoint_candidates(start, bounds[0], resolution, shape)
    goal_candidates = _endpoint_candidates(goal, bounds[0], resolution, shape)
    edge_cache = {}

    def edge_clear(first_id, second_id, first, second):
        key = (min(first_id, second_id), max(first_id, second_id))
        value = edge_cache.get(key)
        if value is None:
            delta = second-first
            count = max(1, int(np.ceil(np.linalg.norm(delta)/
                                      max(occupancy.resolution*.5, 1e-6))))
            points = first+np.arange(1, count, dtype=float)[:, None]*delta/count
            if len(points):
                hits, out = occupancy.collision_mask(
                    points, np.full(len(points), proxy_radius), margin)
                value = not np.any(hits | out)
            else:
                value = True
            edge_cache[key] = value
        return value

    def connectable(point, candidates, endpoint_id):
        connected = []
        for node_id in candidates:
            if not free[node_id]:
                continue
            node = coordinates[node_id]
            if np.linalg.norm(node-point) <= 2.7*resolution+1e-9 and edge_clear(
                    endpoint_id, int(node_id), point, node):
                connected.append(int(node_id))
        return connected

    starts = connectable(start, start_candidates, -1)
    goals = set(connectable(goal, goal_candidates, -2))
    if not starts or not goals:
        return None, 0
    flat_strides = np.asarray([shape[1]*shape[2], shape[2], 1], dtype=int)
    heap, costs, parents = [], {}, {}
    for node_id in starts:
        initial_cost = float(np.linalg.norm(coordinates[node_id]-start))
        costs[node_id] = initial_cost
        parents[node_id] = -1
        heuristic = float(np.linalg.norm(coordinates[node_id]-goal))
        heapq.heappush(heap, (initial_cost+heuristic_weight*heuristic,
                              initial_cost, node_id))
    expansions = 0
    reached = None
    while heap:
        if expansions % 64 == 0 and time.perf_counter() >= deadline:
            raise _AStarTimeout(expansions)
        _, cost, node_id = heapq.heappop(heap)
        if cost != costs.get(node_id):
            continue
        if node_id in goals:
            reached = node_id
            break
        expansions += 1
        current = coordinates[node_id]
        ijk = np.asarray([node_id//flat_strides[0],
                          (node_id//flat_strides[1]) % shape[1],
                          node_id % shape[2]], dtype=int)
        next_indices = ijk[None, :]+_NEIGHBORS
        inside = np.all((next_indices >= 0) & (next_indices < shape), axis=1)
        offsets = _NEIGHBORS[inside]
        next_ids = next_indices[inside] @ flat_strides
        free_neighbors = free[next_ids]
        offsets, next_ids = offsets[free_neighbors], next_ids[free_neighbors]
        if not len(next_ids):
            continue
        edge_ok = np.ones(len(next_ids), dtype=bool)
        pending_indices, pending_ids = [], []
        for index, next_id in enumerate(next_ids):
            key = (min(node_id, int(next_id)), max(node_id, int(next_id)))
            cached = edge_cache.get(key)
            if cached is False:
                edge_ok[index] = False
            elif cached is None:
                pending_indices.append(index)
                pending_ids.append(int(next_id))
        if pending_ids:
            endpoints = coordinates[pending_ids]
            segment_lengths = np.linalg.norm(endpoints-current[None, :], axis=1)
            midpoints = .5*(endpoints+current[None, :])
            # A sphere centered at the segment midpoint with radius half the
            # edge length plus the proxy radius contains the entire swept
            # proxy. This is conservative and prevents diagonal corner cuts.
            hits, out = occupancy.collision_mask(
                midpoints, proxy_radius+.5*segment_lengths, margin)
            valid_edges = ~(hits | out)
            for index, next_id, valid_edge in zip(
                    pending_indices, pending_ids, valid_edges, strict=True):
                key = (min(node_id, next_id), max(node_id, next_id))
                edge_cache[key] = bool(valid_edge)
                edge_ok[index] = bool(valid_edge)
        for offset, next_id, valid_edge in zip(offsets, next_ids, edge_ok, strict=True):
            if not valid_edge:
                continue
            next_clearance = min(clearance[node_id], clearance[next_id])
            length = float(np.linalg.norm(offset)*resolution)
            step_cost = length*(1.+clearance_weight/
                                 (next_clearance+clearance_offset))
            candidate_cost = cost+step_cost
            if candidate_cost >= costs.get(next_id, np.inf):
                continue
            costs[next_id] = candidate_cost
            parents[next_id] = node_id
            estimate = float(np.linalg.norm(coordinates[next_id]-goal))
            heapq.heappush(heap, (candidate_cost+heuristic_weight*estimate,
                                  candidate_cost, next_id))
    if reached is None:
        return None, expansions
    ids = []
    current_id = reached
    while current_id >= 0:
        ids.append(current_id)
        current_id = parents[current_id]
    ids.reverse()
    route = np.vstack((start, coordinates[ids], goal))
    keep = np.r_[True, np.any(np.diff(route, axis=0) != 0., axis=1)]
    return route[keep], expansions


def _endpoint_candidates(point, lower, resolution, shape):
    center = np.rint((point-lower)/resolution).astype(int)
    indices = [
        center+offset
        for offset in _NEIGHBORS
    ]+[center]
    indices = [index for index in indices
               if np.all(index >= 0) and np.all(index < shape)]
    strides = np.asarray([shape[1]*shape[2], shape[2], 1], dtype=int)
    return [int(index @ strides) for index in indices]


def _simplify(route, occupancy, proxy_radius, margin, step, deadline):
    if len(route) <= 2:
        return route
    result = [route[0]]
    anchor = 0
    while anchor < len(route)-1:
        if time.perf_counter() >= deadline:
            raise TimeoutError
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
