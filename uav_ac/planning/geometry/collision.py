"""Collision predicates independent of any search algorithm."""

import numpy as np


def segment_intersects_aabb(
        start: np.ndarray,
        end: np.ndarray,
        bounds: np.ndarray,
) -> bool:
    """Return whether a closed segment intersects an axis-aligned box."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    bounds = np.asarray(bounds, dtype=float).reshape(6)
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        low, high = bounds[2 * axis:2 * axis + 2]
        if abs(direction[axis]) < 1.0e-12:
            if start[axis] < low or start[axis] > high:
                return False
            continue
        t_low = (low - start[axis]) / direction[axis]
        t_high = (high - start[axis]) / direction[axis]
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        t_min = max(t_min, t_low)
        t_max = min(t_max, t_high)
        if t_min > t_max:
            return False
    return True
