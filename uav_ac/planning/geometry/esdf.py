"""Signed Euclidean distance queries backed by the open-source ``edt`` package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import NdBSpline

import edt

from .grid_map import GridMap


@dataclass(frozen=True)
class ESDF:
    """Axis-aligned, uniformly sampled distance grid in NED coordinates."""

    values: np.ndarray
    origin: np.ndarray
    resolution: float

    def __post_init__(self):
        raw_values = np.asarray(self.values)
        # ``edt.sdf`` returns float32. Keep that storage through interpolation
        # instead of doubling a multi-million-voxel field to float64. Integer
        # inputs still become float64 so the public constructor remains useful
        # for hand-built fields.
        value_dtype = (raw_values.dtype if np.issubdtype(raw_values.dtype, np.floating)
                       else np.dtype(float))
        values = np.array(raw_values, dtype=value_dtype, copy=True, order="C")
        origin = np.asarray(self.origin, dtype=float).copy()
        if values.ndim != 3 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("ESDF values must be a finite 3-D grid with at least 2 cells per axis")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("ESDF origin must be a finite 3-vector")
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("ESDF resolution must be positive and finite")
        values.setflags(write=False)
        origin.setflags(write=False)
        # Linear tensor-product B-splines reproduce grid values exactly and
        # provide analytic, piecewise-constant derivatives.  Repeated end
        # knots make the interpolation domain coincide with the map bounds;
        # extrapolation remains disabled and is checked explicitly below.
        knots = tuple(np.r_[origin[i], origin[i],
                            origin[i] + self.resolution*np.arange(1, values.shape[i]-1),
                            origin[i] + self.resolution*(values.shape[i]-1),
                            origin[i] + self.resolution*(values.shape[i]-1)]
                      for i in range(3))
        spline = NdBSpline(knots, values, k=(1, 1, 1), extrapolate=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "_spline", spline)

    @property
    def upper(self) -> np.ndarray:
        return self.origin + self.resolution * (np.asarray(self.values.shape) - 1)

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite (N, 3) array")
        if np.any(points < self.origin) or np.any(points > self.upper):
            raise ValueError("ESDF query lies outside the known grid")
        return np.asarray(self._spline(points, extrapolate=False))

    def distance_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite (N, 3) array")
        if np.any(points < self.origin) or np.any(points > self.upper):
            raise ValueError("ESDF query lies outside the known grid")
        spline = self._spline
        distance = spline(points, extrapolate=False)
        # NdBSpline.__call__(nu=...) evaluates the analytic derivative of the
        # same tensor-product spline. Constructing ``spline.derivative(...)``
        # here is prohibitively expensive on dense 3-D fields: SciPy builds
        # each derivative coefficient array with a Python loop over every
        # coefficient in the other two dimensions.
        gradient = np.column_stack(tuple(
            spline(points, nu=tuple(1 if axis == index else 0 for axis in range(3)),
                   extrapolate=False)
            for index in range(3)))
        return np.asarray(distance), gradient

    @classmethod
    def from_occupancy(cls, occupancy: GridMap) -> "ESDF":
        """Build the signed field with the open-source ``edt`` implementation."""
        occupied = np.asarray(occupancy.occupied, dtype=bool)
        if not np.any(occupied) or np.all(occupied):
            raise ValueError("ESDF occupancy must contain both free and occupied voxels")
        # ``edt.sdf`` assigns positive distance to foreground and negative
        # distance to zero-valued voxels. Here free space is foreground.
        values = edt.sdf(~occupied, anisotropy=(occupancy.resolution,)*3,
                         black_border=False, parallel=1)
        return cls(values, occupancy.origin, occupancy.resolution)

    @classmethod
    def from_axis_aligned_boxes(
            cls, boxes: np.ndarray, lower: np.ndarray, upper: np.ndarray,
            resolution: float, *, ground_height: float | None = None,
            block_cells: int = 2_000_000) -> "ESDF":
        """Rasterize boxes through the reusable grid map and build the signed field."""
        occupancy = GridMap.from_axis_aligned_boxes(
            boxes, lower, upper, resolution, ground_height=ground_height,
            block_cells=block_cells)
        return cls.from_occupancy(occupancy)

__all__ = ["ESDF"]
