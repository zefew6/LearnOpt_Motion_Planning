"""Object-oriented 3D Fast Iterative Region Inflation (FIRI).

All geometry uses ``A @ x <= b`` half-spaces in the project's NED frame.
The public API is intentionally small: construct :class:`FIRI3D` once for a
scene, then inflate a region, build a path corridor, or cover known free space.
"""

from __future__ import annotations

import numpy as np

from uav_ac.planning.corridor.firi.config import FIRIConfig
from uav_ac.planning.corridor.firi.mvie import (
    deepest_interior,
    ellipsoid_volume,
    maximum_volume_inscribed_ellipsoid,
    mvie_cost_gradient,
)
from uav_ac.planning.corridor.firi.separator import (
    minimum_norm_point,
    separating_halfspaces,
)
from uav_ac.planning.corridor.firi.types import FIRIRegion, FreeSpaceCover
from uav_ac.planning.geometry import sample_path, sample_path_preserving_vertices


class FIRI3D:
    """Generate obstacle-free convex regions for one bounded 3D scene."""

    # Compatibility aliases for callers that previously used these helpers as
    # class-level utilities.  Their implementations now live in focused layers.
    sample_path = staticmethod(sample_path)
    sample_path_preserving_vertices = staticmethod(sample_path_preserving_vertices)
    _minimum_norm_point = staticmethod(minimum_norm_point)
    _deepest_interior = staticmethod(deepest_interior)
    _mvie_cost_gradient = staticmethod(mvie_cost_gradient)

    def __init__(
            self,
            obstacle_points: np.ndarray,
            lower_bound: np.ndarray,
            upper_bound: np.ndarray,
            config: FIRIConfig | None = None,
    ):
        self.obstacle_points = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
        self.lower_bound = self._vector(lower_bound, "lower_bound")
        self.upper_bound = self._vector(upper_bound, "upper_bound")
        if np.any(self.lower_bound >= self.upper_bound):
            raise ValueError("lower_bound must be smaller than upper_bound")
        if not np.all(np.isfinite(self.obstacle_points)):
            raise ValueError("obstacle_points must be finite")
        self.config = FIRIConfig() if config is None else config

    @classmethod
    def from_aabbs(
            cls,
            obstacles: np.ndarray,
            lower_bound: np.ndarray,
            upper_bound: np.ndarray,
            *,
            surface_spacing: float = 0.4,
            obstacle_padding: float = 0.15,
            config: FIRIConfig | None = None,
    ) -> FIRI3D:
        """Construct a planner by sampling axis-aligned obstacle boxes."""
        points = cls.sample_aabb_surfaces(
            obstacles, spacing=surface_spacing, padding=obstacle_padding)
        return cls(points, lower_bound, upper_bound, config)

    def inflate(
            self,
            seed_points: np.ndarray,
            *,
            lower_bound: np.ndarray | None = None,
            upper_bound: np.ndarray | None = None,
            max_iterations: int | None = None,
    ) -> FIRIRegion:
        """Inflate one obstacle-free region around a seed set."""
        seed_points = np.asarray(seed_points, dtype=float).reshape(-1, 3)
        if len(seed_points) == 0:
            raise ValueError("at least one seed point is required")
        lower = self.lower_bound if lower_bound is None else self._vector(
            lower_bound, "lower_bound")
        upper = self.upper_bound if upper_bound is None else self._vector(
            upper_bound, "upper_bound")
        A, b = self._box_planes(lower, upper)
        if not np.all(A @ seed_points.T <= b[:, None] + 1e-9):
            raise ValueError("seed points must lie inside the region bounds")
        relevant_obstacles = self._obstacles_in_box(lower, upper)
        iterations = self.config.max_iterations if max_iterations is None else max_iterations
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")
        return self._inflate(seed_points, relevant_obstacles, A, b, iterations)

    def build_safe_flight_corridor(
            self,
            path: np.ndarray,
            *,
            seed_spacing: float = 2.0,
            local_half_size: np.ndarray | tuple[float, float, float] = (2.0, 2.0, 1.2),
            seed_window_size: int = 3,
            preserve_path_vertices: bool = False,
            max_iterations: int | None = None,
    ) -> list[FIRIRegion]:
        """Build overlapping local regions along an RRT/A* geometric path."""
        local_half_size = self._positive_vector(local_half_size, "local_half_size")
        if seed_window_size < 2:
            raise ValueError("seed_window_size must be at least two")
        seeds = (
            self.sample_path_preserving_vertices(path, seed_spacing)
            if preserve_path_vertices else self.sample_path(path, seed_spacing)
        )
        window_size = min(seed_window_size, len(seeds))
        regions = []
        for index in range(len(seeds) - window_size + 1):
            seed_window = seeds[index:index + window_size]
            lower = np.maximum(
                self.lower_bound, np.min(seed_window, axis=0) - local_half_size)
            upper = np.minimum(
                self.upper_bound, np.max(seed_window, axis=0) + local_half_size)
            regions.append(self.inflate(
                seed_window,
                lower_bound=lower,
                upper_bound=upper,
                max_iterations=max_iterations,
            ))
        return regions

    @staticmethod
    def sample_path_preserving_vertices(path: np.ndarray, spacing: float) -> np.ndarray:
        """Sample every path edge while retaining all RRT vertices and corners."""
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] < 3 or len(path) < 2:
            raise ValueError("path must have shape (n, >=3) with n >= 2")
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")
        path = path[:, :3]
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

    def cover_free_space(
            self,
            free_space_samples: np.ndarray,
            *,
            local_half_size: np.ndarray | tuple[float, float, float] = (3.0, 3.0, 2.0),
            target_coverage: float = 0.95,
            max_regions: int = 64,
            max_iterations: int | None = None,
    ) -> FreeSpaceCover:
        """Greedily cover collision-checked free-space samples."""
        samples = np.asarray(free_space_samples, dtype=float).reshape(-1, 3)
        if len(samples) == 0:
            return FreeSpaceCover([], samples, np.zeros(0, dtype=bool))
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must lie in (0, 1]")
        if max_regions <= 0:
            raise ValueError("max_regions must be positive")
        local_half_size = self._positive_vector(local_half_size, "local_half_size")
        if not np.all((samples > self.lower_bound) & (samples < self.upper_bound)):
            raise ValueError("free-space samples must lie strictly inside the global bounds")

        clearance = self._nearest_obstacle_distances(samples)
        covered = np.zeros(len(samples), dtype=bool)
        regions = []
        while np.mean(covered) < target_coverage and len(regions) < max_regions:
            candidates = np.flatnonzero(~covered)
            seed_index = candidates[np.argmax(clearance[candidates])]
            seed = samples[seed_index]
            lower = np.maximum(self.lower_bound, seed - local_half_size)
            upper = np.minimum(self.upper_bound, seed + local_half_size)
            region = self.inflate(
                seed[None, :],
                lower_bound=lower,
                upper_bound=upper,
                max_iterations=max_iterations,
            )
            newly_covered = np.asarray(region.contains(samples))
            if not np.any(newly_covered & ~covered):
                covered[seed_index] = True
                continue
            regions.append(region)
            covered |= newly_covered
        return FreeSpaceCover(regions, samples, covered)

    @staticmethod
    def sample_path(path: np.ndarray, spacing: float) -> np.ndarray:
        """Resample a 3D path at approximately uniform arc-length spacing."""
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] < 3 or len(path) < 2:
            raise ValueError("path must have shape (n, >=3) with n >= 2")
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")
        path = path[:, :3]
        keep = np.concatenate((
            [True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-10))
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
            sampled.append(
                path[segment] + fraction * (path[segment + 1] - path[segment]))
        return np.asarray(sampled)

    @staticmethod
    def sample_aabb_surfaces(
            obstacles: np.ndarray,
            *,
            spacing: float = 0.4,
            padding: float = 0.0,
    ) -> np.ndarray:
        """Sample surfaces of ``[xmin, xmax, ...]`` axis-aligned boxes."""
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 6)
        if spacing <= 0.0 or padding < 0.0:
            raise ValueError("spacing must be positive and padding non-negative")
        points = []
        for obstacle in obstacles:
            lower = obstacle[[0, 2, 4]] - padding
            upper = obstacle[[1, 3, 5]] + padding
            axes = [
                np.linspace(low, high, max(2, int(np.ceil((high - low) / spacing)) + 1))
                for low, high in zip(lower, upper)
            ]
            for fixed_axis in range(3):
                free_axes = [axis for axis in range(3) if axis != fixed_axis]
                grid = np.meshgrid(
                    axes[free_axes[0]], axes[free_axes[1]], indexing="ij")
                for fixed_value in (lower[fixed_axis], upper[fixed_axis]):
                    face = np.empty((grid[0].size, 3))
                    face[:, fixed_axis] = fixed_value
                    face[:, free_axes[0]] = grid[0].ravel()
                    face[:, free_axes[1]] = grid[1].ravel()
                    points.append(face)
        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.unique(np.round(np.vstack(points), decimals=10), axis=0)

    def _inflate(
            self,
            seed_points: np.ndarray,
            obstacles: np.ndarray,
            bbox_A: np.ndarray,
            bbox_b: np.ndarray,
            max_iterations: int,
    ) -> FIRIRegion:
        center = seed_points.mean(axis=0)
        # GCOPTER's FIRI starts in the Euclidean metric.  The ellipsoid is a
        # coordinate transform during the first separation pass and need not
        # already be inscribed in the bounding polytope.
        shape = np.eye(3)
        previous_volume = ellipsoid_volume(shape)
        region_A, region_b = bbox_A.copy(), bbox_b.copy()
        iterations = 0

        for iteration in range(max_iterations):
            iterations = iteration + 1
            region_A, region_b = separating_halfspaces(
                obstacles, seed_points, shape, center, bbox_A, bbox_b,
                self.config.max_planes)
            # As in GCOPTER firi.hpp, MVIE is only needed to define the metric
            # of the *next* separating pass.  Solving it after the final pass
            # changes no half-space and wastes most of a one-iteration run.
            if iteration == max_iterations - 1:
                break
            next_shape, next_center = maximum_volume_inscribed_ellipsoid(
                region_A, region_b, shape, center, self.config)
            volume = ellipsoid_volume(next_shape)
            improvement = (volume - previous_volume) / (previous_volume + 1e-15)
            shape, center = next_shape, next_center
            if iteration > 0 and improvement < self.config.convergence_tolerance:
                break
            previous_volume = volume
        available = region_b - region_A @ center
        radii = np.linalg.norm(region_A @ shape, axis=1)
        scale = float(np.min(available / np.maximum(radii, 1.0e-15)))
        if scale <= 1.0e-10:
            center, depth = deepest_interior(region_A, region_b)
            shape = 0.999 * depth * np.eye(3)
        else:
            shape *= min(1.0, scale * (1.0 - 1.0e-10))
        return FIRIRegion(region_A, region_b, iterations, center, shape)

    def _obstacles_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        relevant = np.all(
            (self.obstacle_points >= lower - 1e-9)
            & (self.obstacle_points <= upper + 1e-9),
            axis=1,
        )
        return self.obstacle_points[relevant]

    def _nearest_obstacle_distances(self, samples: np.ndarray) -> np.ndarray:
        if len(self.obstacle_points) == 0:
            return np.full(len(samples), np.inf)
        distances = np.empty(len(samples))
        for start in range(0, len(samples), 256):
            batch = samples[start:start + 256]
            offsets = batch[:, None, :] - self.obstacle_points[None, :, :]
            distances[start:start + len(batch)] = np.sqrt(
                np.min(np.sum(offsets * offsets, axis=2), axis=1))
        return distances

    @staticmethod
    def _box_planes(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.any(lower >= upper):
            raise ValueError("box lower bounds must be smaller than upper bounds")
        return np.vstack((np.eye(3), -np.eye(3))), np.concatenate((upper, -lower))

    @staticmethod
    def _vector(values: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite values")
        return vector

    @classmethod
    def _positive_vector(cls, values: np.ndarray, name: str) -> np.ndarray:
        vector = cls._vector(values, name)
        if np.any(vector <= 0.0):
            raise ValueError(f"{name} must be positive")
        return vector
