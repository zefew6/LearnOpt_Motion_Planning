"""BMTP: alternate timed curves and planes, tagging only encountered collisions."""

from time import perf_counter

import cvxpy as cp
import numpy as np

from ...geometry.polytope import ConvexPolytope
from .bezier import derivative_matrix
from .collision import collisions, normalized
from .config import BMTPConfig, BMTPLimits
from .convex import PlaneProgram, TrajectoryProgram
from .native import NativeTrajectoryProgram
from .types import BMTPResult, BMTPIteration, BMTPTrajectory


def constraint_residuals(trajectory: BMTPTrajectory, start: np.ndarray, goal: np.ndarray,
                         domain: ConvexPolytope, limits: BMTPLimits,
                         config: BMTPConfig) -> dict[str, float]:
    """Nonnegative residuals; derivative residuals relative to their SI bounds."""
    points, h = trajectory.control_points, trajectory.segment_time
    n = points.shape[1]-1
    boundary = max(np.linalg.norm(points[0, 0]-start), np.linalg.norm(points[-1, -1]-goal))
    workspace = max(0., float(np.max(points @ domain.A.T-domain.b)))
    continuity, terminal, derivative = 0., 0., 0.
    for k in range(max(len(limits.values), config.continuity_order, config.terminal_order)+1):
        controls = np.einsum("ij,mjk->mik", derivative_matrix(n, k), points)/h**k
        scale = limits.values[k-1] if 0 < k <= len(limits.values) else 1.
        if k <= config.continuity_order and len(points) > 1:
            continuity = max(continuity, float(np.max(np.linalg.norm(controls[:-1, -1]-controls[1:, 0], axis=1)))/scale)
        if 0 < k <= config.terminal_order:
            terminal = max(terminal, float(max(np.linalg.norm(controls[0, 0]), np.linalg.norm(controls[-1, -1])))/scale)
        if 0 < k <= len(limits.values):
            derivative = max(derivative, float(np.max(np.linalg.norm(controls, axis=2)))/scale-1.)
    return dict(boundary=float(boundary), workspace=workspace, continuity=continuity,
                terminal=terminal, derivative=derivative)


class BMTPPlanner:
    def __init__(self, config: BMTPConfig | None = None):
        self.config = config or BMTPConfig()

    def plan(self, initial_path: np.ndarray, obstacles: list[ConvexPolytope],
             domain: ConvexPolytope, limits: BMTPLimits | None = None) -> BMTPResult:
        config, limits = self.config, limits or BMTPLimits()
        path = np.asarray(initial_path, float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.all(np.isfinite(path)):
            raise ValueError("initial_path must be finite (N, 3), N >= 2")
        if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) < 1e-8):
            raise ValueError("consecutive initial waypoints must be distinct")
        if max(config.continuity_order, config.terminal_order) > len(limits.values):
            raise ValueError("continuity/terminal order must not exceed bounded derivative order")
        obstacles, domain = [normalized(o) for o in obstacles], normalized(domain)
        if not np.all(domain.contains(path)):
            raise ValueError("initial path lies outside domain")
        chords = np.stack((path[:-1], path[1:]), axis=1)
        if collisions(chords, obstacles, config.collision_tolerance, config.collision_max_depth):
            raise ValueError("initial path collides or cannot be certified collision-free")
        started = perf_counter()
        result = BMTPResult(None, "max_iterations", "trajectory update budget exhausted", path.copy())
        timings = dict(build_seconds=0., trajectory_seconds=0., plane_seconds=0.,
                       collision_seconds=0., solver_seconds=0., initialization_seconds=0.)
        result.timings = timings
        plane_programs = {}
        planes = {}
        cached_tags = None
        trajectory_program = None
        try:
            # A collision-free seed is not necessarily C^I. Never expose it as
            # the anytime solution until the full constraints have been checked.
            seed_start = perf_counter()
            chord_program = TrajectoryProgram(1, domain, limits, config, on_segment=True)
            pieces, durations = [], []
            for begin, end in zip(path[:-1], path[1:]):
                piece, _, solver_time = chord_program.solve(begin, end, {})
                pieces.append(piece.control_points[0])
                durations.append(piece.segment_time)
                timings["solver_seconds"] += solver_time
            seed = BMTPTrajectory(np.asarray(pieces), max(durations))
            result.initialization = seed
            timings["initialization_seconds"] = perf_counter()-seed_start
            reference = seed
            seed_residuals = constraint_residuals(seed, path[0], path[-1], domain, limits, config)
            if max(seed_residuals.values()) <= config.feasibility_tolerance and not collisions(
                    seed.control_points, obstacles, config.collision_tolerance, config.collision_max_depth):
                result.trajectory = seed
            if config.trajectory_backend == "clarabel":
                build_start = perf_counter()
                trajectory_program = NativeTrajectoryProgram(len(path)-1, domain, limits, config)
                timings["build_seconds"] += perf_counter()-build_start
            for iteration in range(1, config.max_iterations+1):
                tags = tuple(sorted(planes))
                if config.trajectory_backend == "cvxpy" and tags != cached_tags:
                    build_start = perf_counter()
                    trajectory_program = TrajectoryProgram(len(path)-1, domain, limits, config, tags)
                    timings["build_seconds"] += perf_counter()-build_start
                    cached_tags = tags
                candidate, elapsed, solver_time = trajectory_program.solve(path[0], path[-1], planes)
                timings["trajectory_seconds"] += elapsed
                timings["solver_seconds"] += solver_time
                residuals = constraint_residuals(candidate, path[0], path[-1], domain, limits, config)
                if max(residuals.values()) > config.feasibility_tolerance:
                    raise RuntimeError(f"trajectory constraint residual exceeds tolerance: {residuals}")
                if not tags:
                    # Numerical obstacle-free optimum in this fixed spline family;
                    # not a rigorous global lower bound or proof of optimality.
                    result.lower_bound_duration = candidate.duration
                check_start = perf_counter()
                hits = collisions(candidate.control_points, obstacles, config.collision_tolerance, config.collision_max_depth)
                timings["collision_seconds"] += perf_counter()-check_start
                new_tags = hits - set(tags)
                accepted = not hits
                previous_time = result.trajectory.duration if result.trajectory else None
                if accepted and previous_time is not None and candidate.duration > previous_time*(1+config.feasibility_tolerance):
                    raise RuntimeError("candidate increased feasible duration beyond numerical tolerance")
                result.history.append(BMTPIteration(iteration, candidate, accepted, tuple(sorted(hits)),
                                                    tuple(sorted(new_tags)), perf_counter()-started, residuals))
                if accepted:
                    result.trajectory = candidate
                    reference = candidate
                    if previous_time is not None and (previous_time-candidate.duration)/previous_time < config.relative_tolerance:
                        result.status, result.message = "converged", "relative feasible-duration improvement below threshold"
                        break
                    if not tags:
                        result.status, result.message = "converged", "obstacle-free optimum is collision-free"
                        break
                    update_tags = tags
                else:
                    if not new_tags:
                        raise RuntimeError("tagged collision remains after plane-constrained solve")
                    update_tags = tuple(sorted(new_tags))
                # New tags use the previous collision-free curve, NOT candidate.
                for segment, obstacle_index in update_tags:
                    if obstacle_index not in plane_programs:
                        build_start = perf_counter()
                        plane_programs[obstacle_index] = PlaneProgram(obstacles[obstacle_index], config)
                        timings["build_seconds"] += perf_counter()-build_start
                    plane, elapsed, solver_time = plane_programs[obstacle_index].solve(reference.control_points[segment])
                    planes[segment, obstacle_index] = plane
                    timings["plane_seconds"] += elapsed
                    timings["solver_seconds"] += solver_time
        except (cp.error.SolverError, RuntimeError, FloatingPointError) as error:
            result.status, result.message = "solver_failure", str(error)
        timings["total_seconds"] = perf_counter()-started
        return result
