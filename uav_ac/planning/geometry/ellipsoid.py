"""Ellipsoid value type used by convex-region inflation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ellipsoid:
    """Affine image ``center + shape @ unit_ball`` in three dimensions."""

    center: np.ndarray
    shape: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        shape = np.asarray(self.shape, dtype=float)
        if center.shape != (3,) or shape.shape != (3, 3):
            raise ValueError("ellipsoid center/shape must have shapes (3,) and (3, 3)")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(shape)):
            raise ValueError("ellipsoid parameters must be finite")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "shape", shape.copy())

    @property
    def volume(self) -> float:
        return 4.0 * np.pi * abs(float(np.linalg.det(self.shape))) / 3.0

    def support_radius(self, normals: np.ndarray) -> np.ndarray:
        """Return the support radius along one or more row normals."""
        normals = np.asarray(normals, dtype=float).reshape(-1, 3)
        return np.linalg.norm(normals @ self.shape, axis=1)
