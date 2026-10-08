"""Initialized generic collision broad phase with native and NumPy backends."""

from dataclasses import dataclass, field
import math
import numpy as np

from .grid_map import GridMap

try:
    from ..native._collision_broadphase import query as _native_query
except ImportError:
    _native_query = None


def _indices(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integer indices')
    return np.array(array, dtype=np.int64, order='C', copy=True)


@dataclass(frozen=True, init=False, eq=False)
class RRTBroadphase:
    """Own validated numeric metadata for one immutable scene/map lifetime."""

    grid: GridMap
    radii: np.ndarray
    self_pairs: np.ndarray
    payload_pairs: np.ndarray
    self_indices: np.ndarray
    payload_indices: np.ndarray
    geom_indices: np.ndarray
    world_lower: np.ndarray
    world_upper: np.ndarray
    world_table: np.ndarray
    query_count: int
    self_clearance: float
    prefix: np.ndarray = field(repr=False)
    native: bool

    def __init__(self, grid, *, radii, self_pairs, payload_pairs,
                 self_query_indices, payload_query_indices, sphere_geom_indices,
                 world_lower, world_upper, world_query_table, query_count,
                 self_clearance, backend='auto'):
        if not isinstance(grid, GridMap):
            raise TypeError('grid must be a GridMap')
        if backend not in {'auto', 'numpy', 'native'}:
            raise ValueError('backend must be auto, numpy or native')
        if backend == 'native' and _native_query is None:
            raise RuntimeError('native broadphase is not built; run python setup.py build_ext --inplace')
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "radii", np.array(radii, dtype=float, order='C', copy=True))
        object.__setattr__(self, "self_pairs", _indices(self_pairs, 'self_pairs'))
        object.__setattr__(self, "payload_pairs", _indices(payload_pairs, 'payload_pairs'))
        object.__setattr__(self, "self_indices", _indices(self_query_indices, 'self_query_indices'))
        object.__setattr__(self, "payload_indices", _indices(payload_query_indices, 'payload_query_indices'))
        object.__setattr__(self, "geom_indices", _indices(sphere_geom_indices, 'sphere_geom_indices'))
        object.__setattr__(self, "world_lower", np.array(world_lower, dtype=float, order='C', copy=True))
        object.__setattr__(self, "world_upper", np.array(world_upper, dtype=float, order='C', copy=True))
        if not self.world_lower.size and self.world_lower.ndim == 1:
            object.__setattr__(self, "world_lower", self.world_lower.reshape(0, 3))
        if not self.world_upper.size and self.world_upper.ndim == 1:
            object.__setattr__(self, "world_upper", self.world_upper.reshape(0, 3))
        object.__setattr__(self, "world_table", _indices(world_query_table, 'world_query_table'))
        object.__setattr__(self, "query_count", int(query_count))
        object.__setattr__(self, "self_clearance", float(self_clearance))
        if (self.radii.ndim != 1 or not np.all(np.isfinite(self.radii))
                or np.any(self.radii < 0) or self.query_count < 0
                or not math.isfinite(self.self_clearance) or self.self_clearance < 0):
            raise ValueError('invalid broadphase radii, query count or clearance')
        for pairs, ids in ((self.self_pairs, self.self_indices),
                           (self.payload_pairs, self.payload_indices)):
            if (pairs.ndim != 2 or pairs.shape[1] != 2 or ids.shape != (len(pairs),)
                    or np.any(pairs < 0) or np.any(pairs >= len(self.radii))
                    or np.any(ids < 0) or np.any(ids >= self.query_count)):
                raise ValueError('invalid sphere pair or query indices')
        if (self.world_lower.ndim != 2 or self.world_lower.shape[1] != 3
                or self.world_upper.shape != self.world_lower.shape
                or np.any(np.isnan(self.world_lower)) or np.any(np.isnan(self.world_upper))
                or np.any(self.world_upper < self.world_lower)
                or self.world_table.ndim != 2
                or self.world_table.shape[1] != len(self.world_lower)
                or self.geom_indices.shape != self.radii.shape
                or np.any(self.geom_indices < -1)
                or np.any(self.geom_indices >= self.world_table.shape[0])
                or np.any(self.world_table < -1) or np.any(self.world_table >= self.query_count)):
            raise ValueError('invalid world bounds or geometry indices')
        object.__setattr__(self, "prefix", np.ascontiguousarray(grid._prefix, dtype=np.int64))
        for array in (self.radii, self.self_pairs, self.payload_pairs,
                      self.self_indices, self.payload_indices, self.geom_indices,
                      self.world_lower, self.world_upper, self.world_table):
            array.setflags(write=False)
        object.__setattr__(self, "native", backend != 'numpy' and _native_query is not None)

    def query(self, points, margin, world_clearance, *, narrow_phase=True):
        points = np.asarray(points, dtype=float, order='C')
        if points.ndim == 2:
            points = points[None, :, :]
        if (points.ndim != 3 or points.shape[1:] != (len(self.radii), 3)
                or not np.all(np.isfinite(points))):
            raise ValueError('points must be finite (S, 3) or (B, S, 3) arrays')
        margin, world_clearance = float(margin), float(world_clearance)
        if (not math.isfinite(margin) or margin < 0
                or not math.isfinite(world_clearance) or world_clearance < 0):
            raise ValueError('margin and world clearance must be nonnegative finite values')
        if self.native:
            return _native_query(
                self.prefix, self.grid.origin, self.grid.resolution, points,
                self.radii, margin, self.self_pairs, self.payload_pairs,
                self.self_indices, self.payload_indices, self.geom_indices,
                self.world_lower, self.world_upper, self.world_table,
                self.query_count, self.self_clearance, world_clearance, narrow_phase)
        return self._numpy_query(points, margin, world_clearance, narrow_phase)

    def _numpy_query(self, points, margin, world_clearance, narrow_phase):
        batch = len(points)
        env = np.full(batch, -np.inf if not len(self.radii) else -1.)
        self_v, payload_v = np.full(batch, -np.inf), np.full(batch, -np.inf)
        selected = np.zeros((batch, self.query_count), dtype=bool)
        worlds = np.zeros(batch, dtype=np.int64)
        outside_any = np.zeros(batch, dtype=bool)
        for row, positions in enumerate(points):
            hits, outside = self.grid.collision_mask(positions, self.radii, margin)
            outside_any[row] = np.any(outside)
            broad = hits | outside
            if np.any(broad):
                env[row] = 1.
            for pairs, ids, values in ((self.self_pairs, self.self_indices, self_v),
                                       (self.payload_pairs, self.payload_indices, payload_v)):
                if not len(pairs):
                    continue
                first, second = pairs.T
                violations = (self.radii[first]+self.radii[second]+self.self_clearance
                              -np.linalg.norm(positions[first]-positions[second], axis=1))
                values[row] = np.max(violations)
                selected[row, ids[violations > 0.]] = True
            if not narrow_phase:
                continue
            sphere_ids = np.flatnonzero(broad & (self.geom_indices >= 0))
            if not len(sphere_ids) or not len(self.world_lower):
                continue
            centers = positions[sphere_ids]
            closest = np.minimum(np.maximum(centers[:, None, :], self.world_lower),
                                 self.world_upper)
            distances = np.sum((centers[:, None, :]-closest)**2, axis=2)
            limits = self.radii[sphere_ids]+world_clearance
            spheres, world = np.nonzero(distances <= limits[:, None]**2+1e-12)
            ids = self.world_table[self.geom_indices[sphere_ids[spheres]], world]
            ids = np.unique(ids[ids >= 0])
            selected[row, ids] = True
            worlds[row] = len(ids)
        return env, self_v, payload_v, selected, worlds, outside_any
