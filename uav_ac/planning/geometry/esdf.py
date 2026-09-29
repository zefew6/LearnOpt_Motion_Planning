"""Signed Euclidean distance queries backed by the open-source ``edt`` package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import NdBSpline

import edt


@dataclass(frozen=True)
class InflatedOccupancyGrid:
    """Binary voxel map for fast collision feasibility queries.

    ``occupied`` contains the uninflated static geometry.  Query-time
    inflation uses an integral volume, so a batch of robot sphere points can
    be checked with integer indexing and no trilinear distance evaluation.
    This is intentionally a feasibility-only structure; MINCO still uses the
    signed field and its analytic gradient.
    """

    occupied: np.ndarray
    origin: np.ndarray
    resolution: float

    def __post_init__(self):
        occupied = np.asarray(self.occupied, dtype=bool).copy()
        origin = np.asarray(self.origin, dtype=float).copy()
        if occupied.ndim != 3 or min(occupied.shape) < 2:
            raise ValueError("occupancy must be a 3-D grid with at least two cells per axis")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("occupancy origin must be a finite 3-vector")
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("occupancy resolution must be positive and finite")
        # A padded 3-D prefix sum lets every query use one indexed box sum.
        prefix = np.pad(occupied.astype(np.int32), 1)
        prefix = prefix.cumsum(0).cumsum(1).cumsum(2)
        occupied.setflags(write=False)
        origin.setflags(write=False)
        prefix.setflags(write=False)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "_prefix", prefix)

    @property
    def upper(self) -> np.ndarray:
        return self.origin + self.resolution*(np.asarray(self.occupied.shape)-1)

    @classmethod
    def from_esdf(cls, esdf: "ESDF") -> "InflatedOccupancyGrid":
        """Build a binary static map from the ESDF's zero-level set."""
        return cls(esdf.values <= 0.0, esdf.origin, esdf.resolution)

    @classmethod
    def from_axis_aligned_boxes(
            cls, boxes: np.ndarray, lower: np.ndarray, upper: np.ndarray,
            resolution: float, *, ground_height: float | None = None,
            block_cells: int = 2_000_000) -> "InflatedOccupancyGrid":
        """Rasterize static boxes directly, without evaluating distances.

        This constructor is used by the RRT front-end.  The returned map is
        the uninflated voxel union; sphere and safety-buffer inflation is
        applied by :meth:`collision_mask` with integer prefix-sum queries.
        """
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
        if (isinstance(block_cells, bool)
                or not isinstance(block_cells, (int, np.integer))
                or block_cells < 1):
            raise ValueError("block_cells must be a positive integer")
        counts = np.ceil((upper-lower)/resolution).astype(int)+1
        axes = [lower[i]+np.arange(counts[i])*resolution for i in range(3)]
        occupied = np.zeros(tuple(counts), dtype=bool)
        yz_cells = int(counts[1]*counts[2])
        block_x = max(1, int(block_cells)//yz_cells)
        half_voxel = .5*float(resolution)
        y = axes[1][None, :, None]
        z = axes[2][None, None, :]
        for first in range(0, int(counts[0]), block_x):
            last = min(int(counts[0]), first+block_x)
            x = axes[0][first:last, None, None]
            slab = np.zeros((last-first, int(counts[1]), int(counts[2])), dtype=bool)
            for box in boxes:
                # A voxel is occupied when its volume intersects a box. This
                # supercover rasterization preserves thin obstacles between
                # voxel-center samples.
                slab |= ((x+half_voxel >= box[0]) & (x-half_voxel <= box[3])
                         & (y+half_voxel >= box[1]) & (y-half_voxel <= box[4])
                         & (z+half_voxel >= box[2]) & (z-half_voxel <= box[5]))
            if ground_height is not None:
                slab |= z+half_voxel >= float(ground_height)
            occupied[first:last] = slab
        return cls(occupied, lower, resolution)

    @classmethod
    def from_scene_geometries(cls, geometries, lower, upper, resolution):
        """Conservatively rasterize static MuJoCo primitives in NED coordinates."""
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower)
                or not np.isfinite(resolution) or resolution <= 0):
            raise ValueError("grid bounds and resolution are invalid")
        counts = np.ceil((upper-lower)/resolution).astype(int)+1
        axes = [lower[i]+np.arange(counts[i])*resolution for i in range(3)]
        occupied = np.zeros(tuple(counts), dtype=bool)
        cell_radius = np.sqrt(3.)*.5*float(resolution)
        for geometry in geometries:
            center = np.asarray(geometry.center, float)
            rotation = np.asarray(geometry.rotation, float)
            size = np.asarray(geometry.size, float)
            if geometry.kind == "plane":
                if abs(rotation[2, 2]) < 1.-1e-8:
                    raise ValueError(f"planning ground '{geometry.name}' must be horizontal")
                z0 = int(np.clip(np.floor(
                    (center[2]-cell_radius-lower[2])/resolution), 0, counts[2]-1))
                occupied[:, :, z0:] = True
                continue
            if geometry.kind not in {"box", "sphere", "cylinder"}:
                raise ValueError(f"unsupported planning geometry '{geometry.kind}'")
            half_extent = geometry.half_extents
            lo = np.maximum(0, np.floor(
                (center-half_extent-cell_radius-lower)/resolution).astype(int))
            hi = np.minimum(counts, np.ceil(
                (center+half_extent+cell_radius-lower)/resolution).astype(int)+1)
            if np.any(hi <= lo):
                continue
            x, y, z = np.meshgrid(axes[0][lo[0]:hi[0]], axes[1][lo[1]:hi[1]],
                                  axes[2][lo[2]:hi[2]], indexing="ij")
            local = np.column_stack((x.ravel(), y.ravel(), z.ravel()))-center
            local = local @ rotation
            if geometry.kind == "sphere":
                inside = np.linalg.norm(local, axis=1) <= size[0]+cell_radius
            elif geometry.kind == "cylinder":
                radial = np.linalg.norm(local[:, :2], axis=1)-size[0]
                axial = np.abs(local[:, 2])-size[1]
                outside = np.linalg.norm(np.maximum(
                    np.column_stack((radial, axial)), 0.), axis=1)
                inside = outside <= cell_radius
            else:
                q = np.abs(local)-size[:3]
                inside = np.linalg.norm(np.maximum(q, 0.), axis=1) <= cell_radius
            region = occupied[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            region |= inside.reshape(region.shape)
        return cls(occupied, lower, float(resolution))

    def collision_mask(self, points: np.ndarray, radii: np.ndarray,
                       margin: float) -> tuple[np.ndarray, np.ndarray]:
        """Return per-point occupancy hits and out-of-grid flags.

        The neighborhood is an axis-aligned voxel box enclosing each sphere.
        One extra voxel covers point-to-grid quantization; this is a
        conservative RRT feasibility test, not a distance estimate.
        """
        points = np.asarray(points, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3
                or radii.shape != (len(points),)
                or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(radii))
                or np.any(radii < 0) or not np.isfinite(margin) or margin < 0):
            raise ValueError("points, radii, and margin have invalid shapes or values")
        coordinates = np.rint((points-self.origin)/self.resolution).astype(int)
        shape = np.asarray(self.occupied.shape)
        outside = np.any((coordinates < 0) | (coordinates >= shape), axis=1)
        hits = np.zeros(len(points), dtype=bool)
        valid = ~outside
        if np.any(valid):
            index = coordinates[valid]
            cells = np.ceil((radii[valid]+margin)/self.resolution).astype(int)+1
            # ``cells`` is one radius per point; expand it across xyz so
            # each query gets an axis-aligned voxel box.
            lower = np.maximum(index-cells[:, None], 0)
            upper = np.minimum(index+cells[:, None], shape-1)
            # Prefix is padded by one element on every side.  Its inclusive
            # cumulative indices exclude the lower voxel at ``lower`` and
            # include the upper voxel at ``upper+1``.
            x0, y0, z0 = lower.T
            x1, y1, z1 = (upper+1).T
            prefix = self._prefix
            sums = (prefix[x1, y1, z1]-prefix[x0, y1, z1]
                    -prefix[x1, y0, z1]-prefix[x1, y1, z0]
                    +prefix[x0, y0, z1]+prefix[x0, y1, z0]
                    +prefix[x1, y0, z0]-prefix[x0, y0, z0])
            hits[valid] = sums > 0
        return hits, outside


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
    def from_occupancy(cls, occupancy: InflatedOccupancyGrid) -> "ESDF":
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
        if isinstance(block_cells, bool) or not isinstance(block_cells, (int, np.integer)) \
                or block_cells < 1:
            raise ValueError("block_cells must be a positive integer")
        occupancy = InflatedOccupancyGrid.from_axis_aligned_boxes(
            boxes, lower, upper, resolution, ground_height=ground_height,
            block_cells=block_cells)
        return cls.from_occupancy(occupancy)


__all__ = ["ESDF", "InflatedOccupancyGrid"]
