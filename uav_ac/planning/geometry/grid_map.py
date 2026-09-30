"""Reusable 3-D occupancy maps in NED world coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .esdf import ESDF


@dataclass(frozen=True)
class GridMap:
    """Immutable voxel occupancy map with world/index conversion and queries.

    ``occupied[i, j, k]`` describes the voxel centered at
    ``origin + resolution * [i, j, k]``. Query-time sphere inflation is
    conservative and uses an integral volume for batched collision checks.
    World coordinates use NED axes.
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
    def shape(self) -> tuple[int, int, int]:
        """Number of voxel centers on each NED axis."""
        return tuple(int(value) for value in self.occupied.shape)

    @property
    def upper(self) -> np.ndarray:
        """World coordinate of the last voxel center on each axis."""
        return self.origin + self.resolution*(np.asarray(self.shape)-1)

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        """Map one point or an ``(N, 3)`` array to nearest voxel indices.

        Half-cell ties follow NumPy's ``rint`` convention (ties to even).
        Indices may lie outside the map; use :meth:`collision_mask` to query
        occupancy and out-of-map status together.
        """
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,) or points.ndim not in (1, 2):
            raise ValueError("points must have shape (3,) or (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        return np.rint((points-self.origin)/self.resolution).astype(np.intp)

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Return the world-space center of one or more integer voxel indices."""
        indices = np.asarray(indices)
        if indices.shape[-1:] != (3,) or indices.ndim not in (1, 2):
            raise ValueError("indices must have shape (3,) or (N, 3)")
        if (not np.issubdtype(indices.dtype, np.integer)
                and (not np.all(np.isfinite(indices))
                     or np.any(indices != np.rint(indices)))):
            raise ValueError("indices must contain finite integers")
        return self.origin + self.resolution*indices.astype(float)

    @classmethod
    def from_esdf(cls, esdf: "ESDF") -> "GridMap":
        """Build occupancy from the ESDF zero level set."""
        return cls(esdf.values <= 0.0, esdf.origin, esdf.resolution)

    @classmethod
    def from_axis_aligned_boxes(
            cls, boxes: np.ndarray, lower: np.ndarray, upper: np.ndarray,
            resolution: float, *, ground_height: float | None = None,
            block_cells: int = 2_000_000) -> "GridMap":
        """Conservatively rasterize NED AABBs and an optional ground plane."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 6)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(upper <= lower)):
            raise ValueError("grid bounds must be ordered finite 3-vectors")
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
                slab |= ((x+half_voxel >= box[0]) & (x-half_voxel <= box[3])
                         & (y+half_voxel >= box[1]) & (y-half_voxel <= box[4])
                         & (z+half_voxel >= box[2]) & (z-half_voxel <= box[5]))
            if ground_height is not None:
                slab |= z+half_voxel >= float(ground_height)
            occupied[first:last] = slab
        return cls(occupied, lower, resolution)

    @classmethod
    def from_scene_geometries(
            cls, geometries, lower: np.ndarray, upper: np.ndarray,
            resolution: float) -> "GridMap":
        """Conservatively rasterize parsed MuJoCo primitives in NED coordinates."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(upper <= lower)
                or not np.isfinite(resolution) or resolution <= 0):
            raise ValueError("grid bounds and resolution are invalid")

        counts = np.ceil((upper-lower)/resolution).astype(int)+1
        axes = [lower[i]+np.arange(counts[i])*resolution for i in range(3)]
        occupied = np.zeros(tuple(counts), dtype=bool)
        cell_radius = np.sqrt(3.)*.5*float(resolution)
        for geometry in geometries:
            center = np.asarray(geometry.center, dtype=float)
            rotation = np.asarray(geometry.rotation, dtype=float)
            size = np.asarray(geometry.size, dtype=float)
            if geometry.kind not in {"box", "sphere", "cylinder", "plane"}:
                raise ValueError(f"unsupported planning geometry '{geometry.kind}'")
            allowed_size_lengths = {
                "box": {3}, "sphere": {1, 3}, "cylinder": {2, 3},
                "plane": {0, 3},
            }[geometry.kind]
            if (center.shape != (3,) or rotation.shape != (3, 3)
                    or not np.all(np.isfinite(center))
                    or not np.all(np.isfinite(rotation))
                    or size.ndim != 1 or len(size) not in allowed_size_lengths
                    or not np.all(np.isfinite(size))
                    or not np.allclose(rotation.T@rotation, np.eye(3),
                                       rtol=0., atol=1e-6)):
                raise ValueError(f"planning geometry '{geometry.name}' is malformed")
            size = np.pad(size, (0, 3-len(size)))
            dimensions = {"box": size[:3], "sphere": size[:1],
                          "cylinder": size[:2]}.get(geometry.kind, np.empty(0))
            if np.any(dimensions <= 0.):
                raise ValueError(f"planning geometry '{geometry.name}' has invalid size")
            if geometry.kind == "plane":
                if abs(rotation[2, 2]) < 1.-1e-8:
                    raise ValueError(f"planning ground '{geometry.name}' must be horizontal")
                z0 = int(np.clip(np.floor(
                    (center[2]-cell_radius-lower[2])/resolution), 0, counts[2]-1))
                occupied[:, :, z0:] = True
                continue
            half_extent = np.asarray(geometry.half_extents, dtype=float)
            if half_extent.shape != (3,) or not np.all(np.isfinite(half_extent)):
                raise ValueError(f"planning geometry '{geometry.name}' has invalid extents")
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

    def collision_mask(
            self, points: np.ndarray, radii: np.ndarray,
            margin: float) -> tuple[np.ndarray, np.ndarray]:
        """Return per-point occupancy hits and out-of-map flags.

        Each sphere is enclosed by an axis-aligned voxel box. An extra voxel
        covers point-to-grid quantization, preserving the conservative
        feasibility semantics used by the RRT and trajectory evaluator.
        """
        points = np.asarray(points, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3
                or radii.shape != (len(points),)
                or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(radii))
                or np.any(radii < 0) or not np.isfinite(margin) or margin < 0):
            raise ValueError("points, radii, and margin have invalid shapes or values")

        coordinates = self.world_to_index(points)
        shape = np.asarray(self.shape)
        outside = np.any((coordinates < 0) | (coordinates >= shape), axis=1)
        hits = np.zeros(len(points), dtype=bool)
        valid = ~outside
        if np.any(valid):
            index = coordinates[valid]
            cells = np.ceil((radii[valid]+margin)/self.resolution).astype(int)+1
            lower = np.maximum(index-cells[:, None], 0)
            upper = np.minimum(index+cells[:, None], shape-1)
            x0, y0, z0 = lower.T
            x1, y1, z1 = (upper+1).T
            prefix = self._prefix
            sums = (prefix[x1, y1, z1]-prefix[x0, y1, z1]
                    -prefix[x1, y0, z1]-prefix[x1, y1, z0]
                    +prefix[x0, y0, z1]+prefix[x0, y1, z0]
                    +prefix[x1, y0, z0]-prefix[x0, y0, z0])
            hits[valid] = sums > 0
        return hits, outside

    def to_pathfinding3d_matrix(
            self, lower: np.ndarray, shape: tuple[int, int, int],
            resolution: float, *, radius: float, margin: float) -> np.ndarray:
        """Rasterize this map onto pathfinding3D's 1-free/0-blocked grid.

        ``lower`` is the NED world center of index ``(0, 0, 0)`` in the
        returned matrix. Conversion is chunked along x to cap temporary memory.
        """
        lower = np.asarray(lower, dtype=float)
        shape_array = np.asarray(shape)
        if (lower.shape != (3,) or not np.all(np.isfinite(lower))
                or shape_array.shape != (3,)
                or not np.issubdtype(shape_array.dtype, np.integer)
                or np.any(shape_array < 1)):
            raise ValueError("lower and shape must be a finite 3-vector and positive integer 3-vector")
        if (not np.isfinite(resolution) or resolution <= 0
                or not np.isfinite(radius) or radius < 0
                or not np.isfinite(margin) or margin < 0):
            raise ValueError("resolution must be positive; radius and margin must be non-negative")
        shape_tuple = tuple(int(value) for value in shape_array)
        matrix = np.zeros(shape_tuple, dtype=np.int8)
        yz_cells = shape_tuple[1]*shape_tuple[2]
        block_x = max(1, 2_000_000//yz_cells)
        y = lower[1]+np.arange(shape_tuple[1])*resolution
        z = lower[2]+np.arange(shape_tuple[2])*resolution
        for first in range(0, shape_tuple[0], block_x):
            last = min(shape_tuple[0], first+block_x)
            x = lower[0]+np.arange(first, last)*resolution
            xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
            points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            # Mark a search cell blocked if any part of its volume intersects
            # an obstacle inflated by the robot radius. The circumsphere of
            # the coarse cell preserves thin obstacles during downsampling.
            hits, outside = self.collision_mask(
                points, np.full(len(points), radius+np.sqrt(3.)*.5*resolution),
                margin)
            matrix[first:last] = (~(hits | outside)).reshape(
                last-first, shape_tuple[1], shape_tuple[2])
        return matrix


__all__ = ["GridMap"]
