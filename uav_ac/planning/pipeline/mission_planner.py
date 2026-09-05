"""Thin composition layer for the UAV mission planning workflow."""

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..corridor.firi import FIRI3D, FIRIConfig, FIRIRegion
from ..search import RRTStar
from ..trajectory.gcopter import GCOPTER, GCOPTERConfig, GCOPTERTrajectory
from ..trajectory.gcs import GCSConfig, GCSPlanner, GCSTrajectory
from ..trajectory.gcs.bezier import evaluate_bezier_derivative
from ..trajectory.minimum_snap import MinimumSnap
from .config import CorridorPlanningConfig
from .result import MissionCorridor


def generate_minimum_snap_mission(
    waypoints: np.ndarray, obstacles: np.ndarray, velocity: float, dt: float,
) -> np.ndarray:
    """Generate isolated takeoff and course segments with MinimumSnap."""
    takeoff = MinimumSnap(waypoints[:2], obstacles, velocity, dt).get_trajectory()
    course = MinimumSnap(waypoints[1:], obstacles, velocity, dt).get_trajectory()
    return np.vstack((takeoff, course))


def gcopter_controller_trajectory(trajectory: GCOPTERTrajectory, dt: float) -> np.ndarray:
    """Convert a GCOPTER result to the controller's p/v/a/yaw sample layout."""
    samples = trajectory.sample(dt)
    yaws = _velocity_yaws(samples.velocities)
    return np.hstack((samples.positions, samples.velocities, samples.accelerations,
                      yaws[:, np.newaxis]))


def generate_gcopter_mission(
    waypoints: np.ndarray,
    corridor: MissionCorridor | list[FIRIRegion],
    quad: Any,
    velocity: float,
    dt: float,
    length_per_piece: float = 1.5,
    max_acceleration: float = 6.0,
    verbose: bool = True,
) -> np.ndarray:
    """Optimize and sample one controller-ready GCOPTER trajectory."""
    regions = corridor.regions if isinstance(corridor, MissionCorridor) else corridor
    boundaries = corridor.fixed_boundaries if isinstance(corridor, MissionCorridor) else None
    config = GCOPTERConfig(
        length_per_piece=length_per_piece, max_velocity=velocity,
        max_acceleration=max_acceleration, mass=quad.m, gravity=quad.g,
        min_thrust=4.0 * quad.min_thrust, max_thrust=4.0 * quad.max_thrust,
        max_tilt_angle=quad.max_tilt_angle)
    optimized = GCOPTER(config).plan(
        waypoints[0], waypoints[-1], regions, fixed_corridor_boundaries=boundaries)
    if verbose:
        print(f"GCOPTER: {optimized.piece_count} pieces, {optimized.duration:.2f} s, "
              f"{optimized.message}.")
    return gcopter_controller_trajectory(optimized, dt)


def build_mission_corridor(
    simulation: Any,
    waypoints: np.ndarray,
    config: CorridorPlanningConfig | None = None,
    *,
    visualize: bool = True,
) -> MissionCorridor:
    """Run RRT* per mission leg and inflate an ordered FIRI corridor."""
    config = CorridorPlanningConfig() if config is None else config
    padded_obstacles = simulation.obstacles.copy()
    padded_obstacles[:, ::2] -= config.obstacle_padding
    padded_obstacles[:, 1::2] += config.obstacle_padding
    rrt_legs = RRTStar.plan_mission_legs(
        simulation.space_limits, waypoints, max_distance=config.rrt_step_size,
        max_iterations=config.rrt_max_iterations, obstacles=padded_obstacles,
        seed=config.rrt_seed)
    firi = FIRI3D(
        simulation.get_planning_obstacle_points(spacing=config.obstacle_point_spacing),
        simulation.space_limits[0], simulation.space_limits[1],
        FIRIConfig(max_iterations=config.firi_max_iterations))
    regions: list[FIRIRegion] = []
    fixed_boundaries: list[tuple[int, np.ndarray]] = []
    for leg_index, leg in enumerate(rrt_legs):
        regions.extend(firi.build_safe_flight_corridor(
            leg, seed_spacing=config.seed_spacing,
            local_half_size=config.local_half_size,
            seed_window_size=config.seed_window_size,
            preserve_path_vertices=True))
        if leg_index < len(rrt_legs) - 1:
            fixed_boundaries.append((len(regions) - 1, waypoints[leg_index + 1]))
    if visualize:
        simulation.set_convex_polyhedra_visualization(regions)
    return MissionCorridor(regions, fixed_boundaries, rrt_legs)


def build_gcs_corridor(
    simulation: Any,
    config: CorridorPlanningConfig | None = None,
    visualize: bool = True,
) -> MissionCorridor:
    """Cover the complete collision-free maze volume with convex boxes."""
    config = CorridorPlanningConfig() if config is None else config
    regions = _voxel_free_space_cover(
        simulation, clearance=config.obstacle_padding)
    if visualize:
        simulation.set_convex_polyhedra_visualization(regions)
    return MissionCorridor(regions, [], simulation.gcs_guide_paths)


def _voxel_free_space_cover(
    simulation: Any,
    *,
    cell_size: tuple[float, float, float] = (0.5, 0.5, 0.25),
    clearance: float = 0.15,
    max_regions: int = 40,
) -> list[FIRIRegion]:
    """Greedily merge every collision-free voxel into obstacle-safe boxes."""
    lower, upper = np.asarray(simulation.space_limits, dtype=float)
    requested_size = np.asarray(cell_size, dtype=float)
    if np.any(requested_size <= 0.0) or clearance < 0.0:
        raise ValueError("cell_size must be positive and clearance non-negative")
    shape = np.ceil((upper - lower) / requested_size).astype(int)
    voxel_size = (upper - lower) / shape
    axes = [
        lower[axis] + (np.arange(shape[axis]) + 0.5) * voxel_size[axis]
        for axis in range(3)
    ]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    free = np.ones(tuple(shape), dtype=bool)
    for obstacle in np.asarray(simulation.obstacles, dtype=float).reshape(-1, 6):
        obstacle_lower = obstacle[[0, 2, 4]] - clearance
        obstacle_upper = obstacle[[1, 3, 5]] + clearance
        overlaps = np.all(
            (centers + 0.5 * voxel_size > obstacle_lower)
            & (centers - 0.5 * voxel_size < obstacle_upper),
            axis=-1,
        )
        free &= ~overlaps
    if not np.any(free):
        raise ValueError("the planning volume contains no collision-free voxels")

    covered = np.zeros_like(free)
    clearance_score = distance_transform_edt(free, sampling=voxel_size)
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    while np.any(free & ~covered):
        if len(bounds) >= max_regions:
            remaining = int(np.count_nonzero(free & ~covered))
            raise RuntimeError(
                f"free-space cover needs more than {max_regions} regions "
                f"({remaining} voxels remain)")
        remaining = free & ~covered
        seed = np.array(np.unravel_index(
            np.argmax(np.where(remaining, clearance_score, -1.0)), shape))
        box_lower = seed.copy()
        box_upper = seed + 1
        while True:
            current = tuple(slice(box_lower[i], box_upper[i]) for i in range(3))
            current_uncovered = int(np.count_nonzero(~covered[current]))
            best_expansion = None
            best_gain = 0
            for axis in range(3):
                for direction in (-1, 1):
                    candidate_lower = box_lower.copy()
                    candidate_upper = box_upper.copy()
                    if direction < 0:
                        if candidate_lower[axis] == 0:
                            continue
                        candidate_lower[axis] -= 1
                    else:
                        if candidate_upper[axis] == shape[axis]:
                            continue
                        candidate_upper[axis] += 1
                    candidate = tuple(
                        slice(candidate_lower[i], candidate_upper[i])
                        for i in range(3))
                    if not np.all(free[candidate]):
                        continue
                    gain = int(np.count_nonzero(~covered[candidate])) - current_uncovered
                    if gain > best_gain:
                        best_gain = gain
                        best_expansion = candidate_lower, candidate_upper
            if best_expansion is None:
                break
            box_lower, box_upper = best_expansion
        cells = tuple(slice(box_lower[i], box_upper[i]) for i in range(3))
        covered[cells] |= free[cells]
        bounds.append((
            lower + box_lower * voxel_size,
            lower + box_upper * voxel_size,
        ))

    A = np.vstack((np.eye(3), -np.eye(3)))
    regions = []
    for region_lower, region_upper in bounds:
        center = 0.5 * (region_lower + region_upper)
        half_size = 0.5 * (region_upper - region_lower)
        regions.append(FIRIRegion(
            A, np.concatenate((region_upper, -region_lower)), 0, center,
            0.95 * float(np.min(half_size)) * np.eye(3)))
    return regions


def cover_scene_free_space(simulation: Any) -> list[FIRIRegion]:
    """Build and visualize a complete obstacle-safe voxel cover."""
    regions = _voxel_free_space_cover(simulation)
    simulation.set_convex_polyhedra_visualization(regions)
    return regions


def generate_gcs_trajectory(
    endpoints: np.ndarray,
    corridor: MissionCorridor | list[FIRIRegion],
    config: GCSConfig | None = None,
) -> GCSTrajectory:
    """Plan over corridor regions while constraining only start and goal."""
    endpoints = np.asarray(endpoints, dtype=float)
    if endpoints.shape != (2, 3):
        raise ValueError("endpoints must have shape (2, 3)")
    regions = corridor.regions if isinstance(corridor, MissionCorridor) else corridor
    return GCSPlanner(config).plan(endpoints[0], endpoints[1], regions)


def gcs_controller_trajectory(
    trajectory: GCSTrajectory,
    max_velocity: float,
    dt: float,
    max_acceleration: float = 6.0,
    max_vertical_velocity: float = 1.0,
) -> np.ndarray:
    """Curvature-aware time-scale a GCS curve into controller samples."""
    if (max_velocity <= 0.0 or dt <= 0.0 or max_acceleration <= 0.0
            or max_vertical_velocity <= 0.0):
        raise ValueError("velocity, dt and acceleration limits must be positive")
    parameters, positions, first, second = _sample_gcs_derivatives(trajectory, 201)
    chord_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    keep = np.concatenate(([True], chord_lengths > 1.0e-10))
    parameters, positions, first, second = (
        values[keep] for values in (parameters, positions, first, second))
    arc_length = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(positions, axis=0), axis=1))))
    if arc_length[-1] <= 1.0e-10:
        raise ValueError("GCS trajectory must have non-zero length")

    parameter_speed = np.linalg.norm(first, axis=1)
    tangents = first / np.maximum(parameter_speed[:, None], 1.0e-12)
    curvature_vectors = (
        second / np.maximum(parameter_speed[:, None] ** 2, 1.0e-12)
        - first * (
            np.sum(first * second, axis=1)
            / np.maximum(parameter_speed**4, 1.0e-12))[:, None]
    )
    curvature = np.linalg.norm(curvature_vectors, axis=1)
    speed_limits = np.minimum(
        0.999 * max_velocity,
        np.sqrt(0.8 * max_acceleration / np.maximum(curvature, 1.0e-12)),
    )
    speed_limits = np.minimum(
        speed_limits,
        0.999 * max_vertical_velocity
        / np.maximum(np.abs(tangents[:, 2]), 1.0e-12),
    )
    speeds = speed_limits.copy()
    speeds[0] = speeds[-1] = 0.0
    tangential_limit = 0.6 * max_acceleration
    for index, distance in enumerate(np.diff(arc_length), start=1):
        speeds[index] = min(
            speeds[index],
            np.sqrt(speeds[index - 1] ** 2 + 2.0 * tangential_limit * distance),
        )
    for index in range(len(speeds) - 2, -1, -1):
        speeds[index] = min(
            speeds[index],
            np.sqrt(speeds[index + 1] ** 2
                    + 2.0 * tangential_limit * (arc_length[index + 1]
                                                 - arc_length[index])),
        )
    node_durations = 2.0 * np.diff(arc_length) / np.maximum(
        speeds[:-1] + speeds[1:], 1.0e-12)
    node_times = np.concatenate(([0.0], np.cumsum(node_durations)))
    intervals = max(1, int(np.ceil(node_times[-1] / dt)))
    duration = intervals * dt
    time_scale = duration / node_times[-1]
    nominal_times = np.arange(intervals + 1) * dt / time_scale
    sample_distance = np.interp(nominal_times, node_times, arc_length)
    sample_parameters = np.interp(sample_distance, arc_length, parameters)
    sample_speeds = np.interp(nominal_times, node_times, speeds) / time_scale
    node_accelerations = np.clip(
        np.gradient(speeds, node_times, edge_order=2),
        -tangential_limit, tangential_limit)
    sample_tangential_acceleration = np.interp(
        nominal_times, node_times, node_accelerations) / time_scale**2

    position_samples, first_samples, second_samples = _evaluate_gcs_parameters(
        trajectory, sample_parameters)
    parameter_speed = np.linalg.norm(first_samples, axis=1)
    tangent_samples = first_samples / np.maximum(
        parameter_speed[:, None], 1.0e-12)
    curvature_samples = (
        second_samples / np.maximum(parameter_speed[:, None] ** 2, 1.0e-12)
        - first_samples * (
            np.sum(first_samples * second_samples, axis=1)
            / np.maximum(parameter_speed**4, 1.0e-12))[:, None]
    )
    velocity_samples = tangent_samples * sample_speeds[:, None]
    acceleration_samples = (
        curvature_samples * sample_speeds[:, None] ** 2
        + tangent_samples * sample_tangential_acceleration[:, None])
    position_samples[[0, -1]] = positions[[0, -1]]
    velocity_samples[[0, -1]] = 0.0
    acceleration_samples[[0, -1]] = 0.0
    yaws = _velocity_yaws(velocity_samples)
    return np.hstack((position_samples, velocity_samples, acceleration_samples,
                      yaws[:, None]))


def _sample_gcs_derivatives(
    trajectory: GCSTrajectory,
    samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pieces = []
    for index in range(trajectory.segment_count):
        local = np.linspace(0.0, 1.0, samples_per_segment)
        if index:
            local = local[1:]
        pieces.append(index + local)
    parameters = np.concatenate(pieces)
    positions, first, second = _evaluate_gcs_parameters(trajectory, parameters)
    return parameters, positions, first, second


def _evaluate_gcs_parameters(
    trajectory: GCSTrajectory,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameters = np.asarray(parameters, dtype=float)
    segment_indices = np.minimum(
        np.floor(parameters).astype(int), trajectory.segment_count - 1)
    local = parameters - segment_indices
    local[parameters >= trajectory.segment_count] = 1.0
    outputs = [np.empty((len(parameters), 3)) for _ in range(3)]
    for segment_index in np.unique(segment_indices):
        selected = segment_indices == segment_index
        for derivative, output in enumerate(outputs):
            output[selected] = evaluate_bezier_derivative(
                trajectory.control_points[segment_index], local[selected], derivative)
    return outputs[0], outputs[1], outputs[2]


def _velocity_yaws(velocities: np.ndarray) -> np.ndarray:
    horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
    moving = np.flatnonzero(horizontal_speeds >= 1.0e-3)
    if len(moving) == 0:
        return np.zeros(len(velocities))
    moving_yaws = np.unwrap(np.arctan2(
        velocities[moving, 1], velocities[moving, 0]))
    previous = np.searchsorted(moving, np.arange(len(velocities)), side="right") - 1
    previous = np.clip(previous, 0, len(moving) - 1)
    return moving_yaws[previous]


__all__ = [
    "build_gcs_corridor", "build_mission_corridor", "cover_scene_free_space",
    "gcopter_controller_trajectory",
    "gcs_controller_trajectory",
    "generate_gcopter_mission", "generate_gcs_trajectory", "generate_minimum_snap_mission",
]
