"""8-D RRT-Connect initialization and equal-duration MINCO/L-BFGS refinement."""

import time

import numpy as np

from ...search.rrt_connect import RRTConnect
from ..gcopter.mappings import backward_time_gradient, forward_time, inverse_time
from ..gcopter.minco import MINCOQuintic
from ..gcopter.optimizer import scipy_lbfgs
from .config import AerialManipulatorMINCOConfig
from .evaluator import AerialManipulatorTrajectoryEvaluator
from .types import AerialManipulatorTrajectory


class AerialManipulatorMINCO:
    def __init__(self, config=None):
        self.config = AerialManipulatorMINCOConfig() if config is None else config
        self.last_metrics = {}

    def plan(self, start_state, goal_state, *, robot, esdf, quad, workspace_bounds,
             gripper_opening, carry_payload=False, rng=None, deadline=None):
        started = time.perf_counter()
        deadline = (started+self.config.planning_budget_s
                    if deadline is None else float(deadline))

        def check_deadline():
            if time.perf_counter() >= deadline:
                raise TimeoutError("aerial_manipulator_minco planning budget exceeded")

        cfg = self.config
        start, goal = _state(start_state), _state(goal_state)
        bounds = np.asarray(workspace_bounds, dtype=float)
        if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("workspace_bounds must have shape (2, 3)")
        lower = np.r_[bounds[0], -np.pi, robot.limits.joint_lower]
        upper = np.r_[bounds[1], np.pi, robot.limits.joint_upper]
        evaluator = AerialManipulatorTrajectoryEvaluator(
            robot, esdf, quad, cfg, bounds, gripper_opening, carry_payload, deadline)

        def distance(a, b):
            delta = b-a
            delta[3] = _wrap(delta[3])
            scale = np.r_[np.full(3, cfg.position_scale), cfg.yaw_scale,
                          np.asarray(cfg.joint_scales)]
            return float(np.linalg.norm(delta/scale))

        metric_scale = np.r_[np.full(3, cfg.position_scale), cfg.yaw_scale,
                             np.asarray(cfg.joint_scales)]

        def distance_batch(states, target):
            delta = np.asarray(target)[None, :]-np.asarray(states)
            delta[:, 3] = (delta[:, 3]+np.pi)%(2*np.pi)-np.pi
            return np.linalg.norm(delta/metric_scale, axis=1)

        def interpolate(a, b, alpha):
            delta = b-a
            delta[3] = _wrap(delta[3])
            result = a+alpha*delta
            result[3] = _wrap(result[3])
            return result

        def state_valid(state):
            check_deadline()
            if (np.any(state[:3] < bounds[0]) or np.any(state[:3] > bounds[1])
                    or np.any(state[4:8] < robot.limits.joint_lower)
                    or np.any(state[4:8] > robot.limits.joint_upper)):
                return False
            # RRT uses the conservative batched sphere/ESDF proxy only. Calling
            # MuJoCo's exact distance query for every interpolated search state
            # duplicates this geometry work; exact geometry is checked densely
            # on the optimized trajectory and every physics step in execution.
            geometry = evaluator.collision_feasible(state)
            return geometry[0] <= 0.0 and geometry[1] <= 0.0

        def edge_valid(a, b, position_resolution=None):
            check_deadline()
            position_resolution = (cfg.edge_position_resolution if position_resolution is None
                                   else position_resolution)
            delta = b-a; delta[3] = _wrap(delta[3])
            count = max(1, int(np.ceil(np.max(np.abs(delta[:3]))/position_resolution)),
                        int(np.ceil(abs(delta[3])/cfg.edge_yaw_resolution)),
                        int(np.ceil(np.max(np.abs(delta[4:8]))/cfg.edge_joint_resolution)))
            return all(state_valid(interpolate(a, b, alpha))
                       for alpha in np.linspace(0., 1., count+1))

        def sample_state():
            sample = rng.uniform(lower, upper)
            # Keep a grasp-stable arm fixed during carry search when both
            # endpoint configurations match. The MINCO optimizer still owns
            # all eight trajectory dimensions and can articulate between the
            # searched knots if collision costs require it.
            if carry_payload and np.allclose(start[4:8], goal[4:8], atol=1e-9):
                sample[4:8] = start[4:8]
            return sample

        search = RRTConnect(
            lower, upper, distance, interpolate, state_valid,
            lambda a, b: edge_valid(
                a, b, min(cfg.rrt_edge_position_resolution,
                          cfg.edge_position_resolution)),
            cfg.rrt_step_size, max_iterations=cfg.rrt_max_iterations,
            goal_bias=cfg.rrt_goal_bias, rng=rng, distance_batch_fn=distance_batch,
            simplify_edge_valid_fn=edge_valid, sample_fn=sample_state)
        phase = time.perf_counter()
        path = search.simplify(search.plan(start, goal))
        for smoothed_path in _smoothed_path_candidates(path, start, goal):
            if all(edge_valid(a, b) for a, b in zip(
                    smoothed_path[:-1], smoothed_path[1:], strict=True)):
                path = smoothed_path
                break
        rrt_seconds = time.perf_counter()-phase
        path = _unwrap_path_yaw(path)
        knots = _resample_preserving_corners(path, max(cfg.pieces, len(path)-1))
        pieces = len(knots)-1
        initial_duration_scale = cfg.initial_duration_scale
        # A carry route found with a larger RRT tree tends to contain a harder
        # detour. Start its time variable with 20% more slack so L-BFGS can
        # satisfy acceleration and rate limits without extra search restarts.
        if carry_payload and search.last_node_count >= 70:
            initial_duration_scale *= 1.2
        total_time = _initial_duration(knots, cfg)*initial_duration_scale
        minco = MINCOQuintic(_boundary_pva(knots[0]), _boundary_pva(knots[-1]), pieces)
        initial = np.r_[knots[1:-1].reshape(-1),
                        inverse_time(np.array([total_time-cfg.minimum_total_time]))]
        # Optimize dimensionless coordinates so meter, radian, and mapped-time
        # variables present comparable step sizes to L-BFGS. The log-time
        # coordinate is already dimensionless and uses unit scale.
        variable_scale = np.r_[np.tile(metric_scale, pieces-1), 1.0]

        objective_calls = 0

        def objective(variables):
            nonlocal objective_calls
            check_deadline()
            objective_calls += 1
            return self._objective(variables, minco, pieces, evaluator)

        def scaled_objective(scaled_variables):
            cost, gradient = objective(scaled_variables*variable_scale)
            return cost, gradient*variable_scale

        phase = time.perf_counter()
        result = scipy_lbfgs(
            scaled_objective, initial/variable_scale,
            max_iterations=cfg.max_iterations,
            memory=cfg.lbfgs_memory, gradient_tolerance=cfg.gradient_tolerance,
            relative_cost_tolerance=1e-7,
            # Keep reserve between quadrature nodes so dense inter-node checks
            # do not routinely invalidate an otherwise stationary iterate.
            is_feasible=lambda: evaluator.last_violation <= -1.0e-2,
            feasible_iteration_patience=2)
        optimizer_seconds = time.perf_counter()-phase
        optimized_variables = result.x*variable_scale
        points = optimized_variables[:8*(pieces-1)].reshape(pieces-1, 8)
        total = cfg.minimum_total_time+float(forward_time(optimized_variables[-1:])[0])
        durations = np.full(pieces, total/pieces)
        coefficients, _ = minco.solve(points, durations)
        coefficients = coefficients.reshape(pieces, 6, 8)
        provisional = AerialManipulatorTrajectory(
            durations, coefficients, path, result.cost, result.iterations,
            result.converged, result.message)
        check_deadline()
        phase = time.perf_counter()
        valid, clearance, violation, sample_dt = evaluator.dense_validate(provisional)
        validation_seconds = time.perf_counter()-phase
        retiming_attempts = 0
        retiming_scale = 1.0
        # A near-feasible optimizer result can exceed a dynamic limit slightly
        # even when its geometric path is safe. Uniform time stretching keeps
        # the complete 8-D path unchanged and lowers derivatives; every
        # stretched candidate still goes through full dense validation.
        while (not valid and retiming_attempts < 3
               and _is_retimeable_validation(evaluator.last_validation_metrics)):
            scale = _retiming_scale(
                evaluator.last_validation_metrics["maximum_violation_kind"],
                violation, cfg)
            provisional = _time_stretch(provisional, scale)
            retiming_scale *= scale
            retiming_attempts += 1
            phase = time.perf_counter()
            valid, clearance, violation, sample_dt = evaluator.dense_validate(provisional)
            validation_seconds += time.perf_counter()-phase
        self.last_metrics = {
            "rrt_seconds": rrt_seconds,
            "rrt_iterations": search.last_iterations,
            "rrt_nodes": search.last_node_count,
            "initial_duration_scale": initial_duration_scale,
            "optimizer_seconds": optimizer_seconds,
            "objective_calls": objective_calls,
            "objective_samples": evaluator.objective_samples,
            "validation_seconds": validation_seconds,
            "validation_samples": evaluator.validation_samples,
            "retiming_attempts": retiming_attempts,
            "retiming_scale": retiming_scale,
            "total_seconds": time.perf_counter()-started,
        }
        self.last_metrics.update(evaluator.last_validation_metrics)
        return AerialManipulatorTrajectory(
            provisional.durations, provisional.coefficients, path,
            result.cost, result.iterations,
            result.converged, result.message, valid, clearance, violation,
            sample_dt)


    def _objective(self, variables, minco, pieces, evaluator):
        cfg = self.config
        variables = np.asarray(variables, dtype=float)
        waypoint_count = pieces-1
        points = variables[:8*waypoint_count].reshape(waypoint_count, 8)
        tau = variables[-1:]
        total = cfg.minimum_total_time+float(forward_time(tau)[0])
        durations = np.full(pieces, total/pieces)
        try:
            coefficients, system = minco.solve(points, durations)
        except (np.linalg.LinAlgError, ValueError):
            return 1.0e30, np.zeros_like(variables)
        coefficients = coefficients.reshape(pieces, 6, 8)
        cost, grad_coefficients, grad_times = MINCOQuintic.jerk_energy(
            coefficients, durations, np.asarray(cfg.jerk_weights))
        penalty_cost, penalty_grad, penalty_times = evaluator.integrated_penalty(
            durations, coefficients)
        cost += penalty_cost+cfg.time_weight*total
        direct_times = grad_times+penalty_times+cfg.time_weight
        grad_points, grad_times = minco.propagate_gradient(
            system, coefficients.reshape(-1, 8), durations,
            (grad_coefficients.reshape(-1, 8)+penalty_grad.reshape(-1, 8)),
            direct_times)
        grad_total = float(np.mean(grad_times))
        grad_tau = backward_time_gradient(tau, np.array([grad_total]))
        gradient = np.r_[grad_points.reshape(-1), grad_tau]
        if not np.isfinite(cost) or not np.all(np.isfinite(gradient)):
            return 1.0e30, np.zeros_like(variables)
        return float(cost), gradient


def _is_retimeable_validation(metrics):
    kind = metrics.get("maximum_violation_kind")
    dynamic = {
        "linear_speed", "linear_acceleration", "yaw_rate", "yaw_acceleration",
        "joint_velocity", "joint_acceleration", "maximum_thrust", "minimum_thrust",
        "tilt", "body_rate",
    }
    return (kind in dynamic and not metrics.get("collision_detected", True)
            and metrics.get("map_inside", False))


def _retiming_scale(kind, violation, config):
    limits = {
        "linear_speed": config.max_speed,
        "linear_acceleration": config.max_acceleration,
        "yaw_rate": config.max_yaw_rate,
        "yaw_acceleration": config.max_yaw_acceleration,
        "joint_velocity": min(config.joint_velocity_limits),
        "joint_acceleration": min(config.joint_acceleration_limits),
        "body_rate": config.max_body_rate,
    }
    if kind in limits:
        # The constraints are expressed as squared magnitudes. Acceleration
        # scales with time^-2, while velocity/body rate scale with time^-1.
        power = 0.25 if kind.endswith("acceleration") else 0.5
        needed = (1.0+max(0.0, float(violation))/limits[kind]**2)**power
        return max(1.05, 1.05*needed)
    # Thrust and tilt depend nonlinearly on acceleration; use a conservative
    # bounded retry and let full-pose validation decide whether it is safe.
    return 1.1


def _time_stretch(trajectory, scale):
    if not np.isfinite(scale) or scale <= 1.0:
        raise ValueError("time stretch must be finite and greater than one")
    coefficients = trajectory.coefficients/scale**np.arange(6)[None, :, None]
    return AerialManipulatorTrajectory(
        trajectory.durations*scale, coefficients, trajectory.rrt_path,
        trajectory.cost, trajectory.iterations, trajectory.optimizer_converged,
        trajectory.optimizer_message, trajectory.validation_passed,
        trajectory.minimum_clearance, trajectory.maximum_violation,
        trajectory.validation_sample_dt)


def _boundary_pva(state):
    result = np.zeros((3, 8)); result[0] = state
    return result


def _state(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (8,) or not np.all(np.isfinite(result)):
        raise ValueError("planning states must be finite 8-vectors")
    result = result.copy(); result[3] = _wrap(result[3])
    return result


def _wrap(angle):
    return float((angle+np.pi)%(2*np.pi)-np.pi)


def _unwrap_path_yaw(path):
    result = np.asarray(path, dtype=float).copy()
    for i in range(1, len(result)):
        result[i, 3] = result[i-1, 3]+_wrap(result[i, 3]-result[i-1, 3])
    return result


def _smoothed_path_candidates(path, start, goal):
    """Smooth random RRT attitude/joint oscillations without dropping turns."""
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return ()
    lengths = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] <= 1e-12:
        cumulative = np.linspace(0., 1., len(path))
    else:
        cumulative /= cumulative[-1]
    yaw_delta = _wrap(goal[3]-start[3])
    target_yaw = start[3]+cumulative*yaw_delta
    original_yaw = _unwrap_path_yaw(path)[:, 3]
    candidates = []
    for amount in (1., .75, .5, .25):
        candidate = path.copy()
        candidate[:, 3] = original_yaw+amount*(target_yaw-original_yaw)
        candidates.append(candidate)
    smooth_joints = candidates[0].copy()
    smooth_joints[:, 4:8] = (start[4:8]
                              +cumulative[:, None]*(goal[4:8]-start[4:8]))
    candidates.append(smooth_joints)
    return tuple(candidates)


def _resample_preserving_corners(path, pieces):
    if len(path) < 2:
        raise ValueError("RRT path must have distinct start and goal entries")
    edge_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    edge_count = len(edge_lengths)
    pieces = max(int(pieces), edge_count)
    extras = pieces-edge_count
    weights = edge_lengths/max(float(np.sum(edge_lengths)), 1e-12)
    allocations = np.ones(edge_count, dtype=int)
    if extras:
        raw = weights*extras
        whole = np.floor(raw).astype(int)
        allocations += whole
        remainder = extras-int(np.sum(whole))
        if remainder:
            allocations[np.argsort(raw-whole)[-remainder:]] += 1
    knots = [path[0]]
    for edge, count in enumerate(allocations):
        for index in range(1, count+1):
            knots.append(path[edge]+(path[edge+1]-path[edge])*(index/count))
    return np.asarray(knots)


def _initial_duration(path, config):
    lengths = np.sum(np.abs(np.diff(path, axis=0)), axis=0)
    time = max(float(np.sum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1))
                            /config.max_speed),
               float(lengths[3]/config.max_yaw_rate),
               float(np.max(lengths[4:8]/np.asarray(config.joint_velocity_limits))),
               config.minimum_total_time+.25)
    return config.minimum_total_time+1.35*max(time, .25)


__all__ = ["AerialManipulatorMINCO"]
