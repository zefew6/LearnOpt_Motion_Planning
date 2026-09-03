"""Arc-length sampling utilities for geometric paths."""

import numpy as np


def _validated_path(path: np.ndarray, spacing: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 3 or len(path) < 2:
        raise ValueError("path must have shape (n, >=3) with n >= 2")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return path[:, :3]


def sample_path_preserving_vertices(path: np.ndarray, spacing: float) -> np.ndarray:
    """Sample every edge while retaining all original path corners."""
    path = _validated_path(path, spacing)
    samples = [path[0]]
    for start, goal in zip(path[:-1], path[1:], strict=True):
        length = float(np.linalg.norm(goal - start))
        if length <= 1.0e-10:
            continue
        subdivisions = max(1, int(np.ceil(length / spacing)))
        samples.extend(
            start + (goal - start) * fraction
            for fraction in np.arange(1, subdivisions + 1) / subdivisions
        )
    if len(samples) < 2:
        raise ValueError("path must have non-zero length")
    return np.asarray(samples)


def sample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at approximately uniform global arc length."""
    path = _validated_path(path, spacing)
    keep = np.concatenate((
        [True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1.0e-10))
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path must have non-zero length")
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.append(np.arange(0.0, cumulative[-1], spacing), cumulative[-1])
    sampled = []
    for target in targets:
        segment = min(
            np.searchsorted(cumulative, target, side="right") - 1,
            len(lengths) - 1,
        )
        fraction = (target - cumulative[segment]) / lengths[segment]
        sampled.append(path[segment] + fraction * (path[segment + 1] - path[segment]))
    return np.asarray(sampled)
