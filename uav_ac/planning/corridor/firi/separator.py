"""Separating-plane construction in the current FIRI ellipsoid metric."""

from itertools import combinations

import numpy as np


def separating_halfspaces(
        obstacles: np.ndarray,
        seed_points: np.ndarray,
        shape: np.ndarray,
        center: np.ndarray,
        bbox_A: np.ndarray,
        bbox_b: np.ndarray,
        max_planes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct GCOPTER-style tangent planes and greedily cull obstacles."""
    forward = np.linalg.inv(shape)
    boundary_normals = bbox_A @ shape
    boundary_offsets = bbox_A @ center - bbox_b
    boundary_distances = (
        np.abs(boundary_offsets) / np.linalg.norm(boundary_normals, axis=1))
    obstacles_bar = (obstacles - center) @ forward.T
    seeds_bar = (seed_points - center) @ forward.T
    normals, offsets, distances = tangent_planes(obstacles_bar, seeds_bar)
    candidates = [
        (float(distance), 0, index)
        for index, distance in enumerate(boundary_distances)
    ] + [
        (float(distance), 1, index)
        for index, distance in enumerate(distances)
        if np.isfinite(distance)
    ]
    candidates.sort(key=lambda item: item[0])

    selected_normals = []
    selected_offsets = []
    selected_boundaries = np.zeros(len(bbox_A), dtype=bool)
    separated = np.zeros(len(obstacles_bar), dtype=bool)
    for _distance, kind, index in candidates:
        if kind == 0:
            normal = boundary_normals[index]
            offset = boundary_offsets[index]
            selected_boundaries[index] = True
        else:
            if separated[index]:
                continue
            normal = normals[index]
            offset = offsets[index]
        selected_normals.append(normal)
        selected_offsets.append(float(offset))
        if len(obstacles_bar):
            separated |= obstacles_bar @ normal + offset >= -1.0e-6
        if np.all(selected_boundaries) and np.all(separated):
            break
        if len(selected_normals) >= max_planes:
            raise RuntimeError("FIRI reached max_planes before separating all obstacles")

    if not np.all(selected_boundaries) or not np.all(separated):
        raise RuntimeError("FIRI could not construct a bounded separating polytope")
    region_A = np.asarray(selected_normals) @ forward
    region_b = -np.asarray(selected_offsets) + region_A @ center
    lengths = np.linalg.norm(region_A, axis=1)
    return region_A / lengths[:, None], region_b / lengths


def tangent_planes(
        obstacles_bar: np.ndarray,
        seeds_bar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct obstacle tangent planes in unit-ellipsoid coordinates."""
    count = len(obstacles_bar)
    if count == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    distances = np.linalg.norm(obstacles_bar, axis=1)
    valid = distances > 1.0e-12
    normals = np.zeros_like(obstacles_bar)
    normals[valid] = obstacles_bar[valid] / distances[valid, None]
    offsets = -distances.copy()
    distances[~valid] = np.inf

    if len(seeds_bar) != 2:
        for index, obstacle in enumerate(obstacles_bar):
            constraints = np.vstack((seeds_bar, -obstacle))
            bounds = np.concatenate((np.ones(len(seeds_bar)), [-1.0]))
            solution, feasible = minimum_norm_point(constraints, bounds)
            norm = float(np.linalg.norm(solution))
            if feasible and norm > 1.0e-12:
                normals[index] = solution / norm
                offsets[index] = -1.0 / norm
                distances[index] = 1.0 / norm
            else:
                distances[index] = np.inf
        return normals, offsets, distances

    first, second = seeds_bar
    move_tangents_behind_seed(normals, offsets, distances, obstacles_bar, first)
    move_tangents_behind_seed(normals, offsets, distances, obstacles_bar, second)
    violation = normals @ first + offsets > 1.0e-6
    if np.any(violation):
        cross = np.cross(
            first - obstacles_bar[violation], second - obstacles_bar[violation])
        cross_norm = np.linalg.norm(cross, axis=1)
        usable = cross_norm > 1.0e-12
        indices = np.flatnonzero(violation)
        usable_indices = indices[usable]
        normals[usable_indices] = cross[usable] / cross_norm[usable, None]
        offsets[usable_indices] = -(normals[usable_indices] @ first)
        flip = offsets[usable_indices] > 0.0
        normals[usable_indices[flip]] *= -1.0
        offsets[usable_indices[flip]] *= -1.0
        distances[usable_indices] = np.abs(offsets[usable_indices])
        distances[indices[~usable]] = np.inf
    return normals, offsets, distances


def move_tangents_behind_seed(
        normals: np.ndarray,
        offsets: np.ndarray,
        distances: np.ndarray,
        obstacles: np.ndarray,
        seed: np.ndarray,
) -> None:
    violation = normals @ seed + offsets > 1.0e-6
    if not np.any(violation):
        return
    indices = np.flatnonzero(violation)
    delta = obstacles[indices] - seed
    denominator = np.sum(delta * delta, axis=1)
    usable = denominator > 1.0e-15
    indices = indices[usable]
    delta = delta[usable]
    projection = seed - (
        np.sum(delta * seed, axis=1) / denominator[usable])[:, None] * delta
    norms = np.linalg.norm(projection, axis=1)
    usable = norms > 1.0e-12
    indices = indices[usable]
    normals[indices] = projection[usable] / norms[usable, None]
    offsets[indices] = -norms[usable]
    distances[indices] = norms[usable]


def minimum_norm_point(
        normals: np.ndarray,
        bounds: np.ndarray,
        tolerance: float = 1.0e-9,
) -> tuple[np.ndarray, bool]:
    """Solve ``min ||y||²`` subject to low-dimensional inequalities."""
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    if len(normals) != len(bounds):
        raise ValueError("normals and bounds must have matching rows")
    if np.all(normals @ np.zeros(3) <= bounds + tolerance):
        return np.zeros(3), True
    best = None
    best_norm_squared = np.inf
    for active_count in range(1, min(3, len(normals)) + 1):
        for indices in combinations(range(len(normals)), active_count):
            active_normals = normals[list(indices)]
            gram = active_normals @ active_normals.T
            if np.linalg.matrix_rank(gram, tol=1.0e-11) < active_count:
                continue
            coefficients = np.linalg.solve(gram, bounds[list(indices)])
            candidate = active_normals.T @ coefficients
            if np.all(normals @ candidate <= bounds + tolerance):
                norm_squared = float(candidate @ candidate)
                if norm_squared < best_norm_squared:
                    best, best_norm_squared = candidate, norm_squared
    return (np.zeros(3), False) if best is None else (best, True)
