"""Conservative continuous Bézier/polytope collision certificates (paper Alg. 2)."""

import numpy as np

from ...geometry.polytope import ConvexPolytope
from .bezier import split


def normalized(polytope: ConvexPolytope) -> ConvexPolytope:
    norms = np.linalg.norm(polytope.A, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("polytope normals must be nonzero")
    return ConvexPolytope(polytope.A / norms[:, None], polytope.b / norms)


def offset(polytope: ConvexPolytope, distance: float) -> ConvexPolytope:
    """Positive distance gives a conservative outer offset; negative shrinks."""
    polytope = normalized(polytope)
    return ConvexPolytope(polytope.A, polytope.b + distance)


def box(lower: np.ndarray, upper: np.ndarray) -> ConvexPolytope:
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
        raise ValueError("box requires ordered three-dimensional bounds")
    return ConvexPolytope(np.vstack((np.eye(3), -np.eye(3))), np.r_[upper, -lower])


def collision_free(points: np.ndarray, obstacle: ConvexPolytope,
                   tolerance: float = 1e-4, max_depth: int = 32) -> bool:
    """False means collision OR unresolved. Boundary contact is not certified free."""
    stack = [(np.asarray(points, float), 0)]
    while stack:
        control, depth = stack.pop()
        residuals = obstacle.A @ control.T - obstacle.b[:, None]
        if np.any(np.max(residuals[:, [0, -1]], axis=0) <= 1e-10):
            return False
        if np.any(np.min(residuals, axis=1) > 1e-10):
            continue
        radius = np.max(np.linalg.norm(control - control.mean(axis=0), axis=1))
        if radius <= tolerance or depth >= max_depth:
            return False
        left, right = split(control)
        stack.extend(((right, depth+1), (left, depth+1)))
    return True


def collisions(control_points: np.ndarray, obstacles: list[ConvexPolytope],
               tolerance: float = 1e-4, max_depth: int = 32) -> set[tuple[int, int]]:
    return {(i, j) for i, segment in enumerate(control_points)
            for j, obstacle in enumerate(obstacles)
            if not collision_free(segment, obstacle, tolerance, max_depth)}
