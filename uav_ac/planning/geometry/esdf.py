"""Batched trilinear Euclidean signed-distance field queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ESDF:
    """Axis-aligned, uniformly sampled distance grid in NED coordinates."""

    values: np.ndarray
    origin: np.ndarray
    resolution: float

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float).copy()
        origin = np.asarray(self.origin, dtype=float).copy()
        if values.ndim != 3 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("ESDF values must be a finite 3-D grid with at least 2 cells per axis")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("ESDF origin must be a finite 3-vector")
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("ESDF resolution must be positive and finite")
        values.setflags(write=False)
        origin.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "resolution", float(self.resolution))

    @property
    def upper(self) -> np.ndarray:
        return self.origin + self.resolution * (np.asarray(self.values.shape) - 1)

    def distance(self, points: np.ndarray) -> np.ndarray:
        return self.distance_and_gradient(points)[0]

    def distance_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite (N, 3) array")
        coordinates = (points - self.origin) / self.resolution
        maximum = np.asarray(self.values.shape) - 1
        if np.any(coordinates < 0.0) or np.any(coordinates > maximum):
            raise ValueError("ESDF query lies outside the known grid")
        base = np.minimum(np.floor(coordinates).astype(int), maximum - 1)
        fraction = coordinates - base
        x, y, z = base.T
        u, v, w = fraction.T
        corner = self.values
        d000, d100 = corner[x, y, z], corner[x+1, y, z]
        d010, d110 = corner[x, y+1, z], corner[x+1, y+1, z]
        d001, d101 = corner[x, y, z+1], corner[x+1, y, z+1]
        d011, d111 = corner[x, y+1, z+1], corner[x+1, y+1, z+1]
        dx00 = d000*(1-u) + d100*u
        dx10 = d010*(1-u) + d110*u
        dx01 = d001*(1-u) + d101*u
        dx11 = d011*(1-u) + d111*u
        dxy0 = dx00*(1-v) + dx10*v
        dxy1 = dx01*(1-v) + dx11*v
        distance = dxy0*(1-w) + dxy1*w
        gradient = np.empty_like(points)
        gradient[:, 0] = ((d100-d000)*(1-v)*(1-w) + (d110-d010)*v*(1-w)
                          + (d101-d001)*(1-v)*w + (d111-d011)*v*w) / self.resolution
        gradient[:, 1] = ((d010-d000)*(1-u)*(1-w) + (d110-d100)*u*(1-w)
                          + (d011-d001)*(1-u)*w + (d111-d101)*u*w) / self.resolution
        gradient[:, 2] = ((d001-d000)*(1-u)*(1-v) + (d101-d100)*u*(1-v)
                          + (d011-d010)*(1-u)*v + (d111-d110)*u*v) / self.resolution
        return distance, gradient

    @classmethod
    def from_axis_aligned_boxes(
            cls, boxes: np.ndarray, lower: np.ndarray, upper: np.ndarray,
            resolution: float, *, ground_height: float | None = None) -> "ESDF":
        """Rasterize the union of NED axis-aligned boxes and optional ground plane."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 6)
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(upper <= lower)):
            raise ValueError("grid bounds must be ordered 3-vectors")
        if (not np.isfinite(resolution) or resolution <= 0
                or not np.all(np.isfinite(boxes))
                or (len(boxes) and np.any(boxes[:, 3:] < boxes[:, :3]))):
            raise ValueError("boxes must be finite and ordered; resolution must be positive")
        if ground_height is not None and not np.isfinite(ground_height):
            raise ValueError("ground_height must be finite")
        counts = np.ceil((upper-lower)/resolution).astype(int) + 1
        axes = [lower[i] + np.arange(counts[i])*resolution for i in range(3)]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        points = np.stack((xx, yy, zz), axis=-1)
        distance = np.full(points.shape[:-1], np.inf)
        for box in boxes:
            center = 0.5*(box[:3] + box[3:])
            half = 0.5*(box[3:] - box[:3])
            q = np.abs(points-center)-half
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
            inside = np.minimum(np.max(q, axis=-1), 0.0)
            distance = np.minimum(distance, outside+inside)
        if ground_height is not None:
            distance = np.minimum(distance, float(ground_height)-zz)
        if not np.all(np.isfinite(distance)):
            raise ValueError("ESDF construction requires at least one box or a ground plane")
        return cls(distance, lower, resolution)


__all__ = ["ESDF"]
