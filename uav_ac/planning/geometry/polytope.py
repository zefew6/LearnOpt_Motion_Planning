"""Half-space convex polytope primitives."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def enumerate_vertices(
        A: np.ndarray,
        b: np.ndarray,
        tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Enumerate 3D vertices from feasible half-space triplet intersections."""
    A = np.asarray(A, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1)
    triples = np.asarray(list(combinations(range(len(A)), 3)), dtype=int)
    if len(triples) == 0:
        return np.zeros((0, 3))
    matrices = A[triples]
    nonsingular = np.abs(np.linalg.det(matrices)) >= 1.0e-10
    if not np.any(nonsingular):
        return np.zeros((0, 3))
    points = np.linalg.solve(
        matrices[nonsingular], b[triples[nonsingular], None])[:, :, 0]
    feasible = np.all(points @ A.T <= b[None, :] + tolerance, axis=1)
    return np.unique(np.round(points[feasible], decimals=9), axis=0)


@dataclass(frozen=True)
class ConvexPolytope:
    """Bounded convex region represented by ``A @ x <= b``."""

    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float).reshape(-1)
        if A.ndim != 2 or A.shape[1] != 3 or len(A) != len(b) or len(A) < 4:
            raise ValueError("polytope half-spaces require A=(m, 3), b=(m,), m >= 4")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("polytope half-spaces must be finite")
        object.__setattr__(self, "A", A.copy())
        object.__setattr__(self, "b", b.copy())

    def contains(self, points: np.ndarray, tolerance: float = 1.0e-8) -> np.ndarray | bool:
        points = np.asarray(points, dtype=float)
        single = points.ndim == 1
        points = points.reshape(-1, 3)
        result = np.all(self.A @ points.T <= self.b[:, None] + tolerance, axis=0)
        return bool(result[0]) if single else result

    def vertices(self, tolerance: float = 1.0e-7) -> np.ndarray:
        return enumerate_vertices(self.A, self.b, tolerance)

    def edges(self, tolerance: float = 1.0e-6) -> list[tuple[np.ndarray, np.ndarray]]:
        vertices = self.vertices()
        active_planes = [
            set(np.flatnonzero(np.abs(self.A @ vertex - self.b) <= tolerance))
            for vertex in vertices
        ]
        return [
            (vertices[first], vertices[second])
            for first, second in combinations(range(len(vertices)), 2)
            if len(active_planes[first].intersection(active_planes[second])) >= 2
        ]

    def intersection(self, other: "ConvexPolytope") -> "ConvexPolytope":
        return ConvexPolytope(
            np.vstack((self.A, other.A)), np.concatenate((self.b, other.b)))
