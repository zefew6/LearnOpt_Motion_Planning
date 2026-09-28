"""8-D RRT-Connect initialization and equal-duration MINCO/L-BFGS refinement."""

import numpy as np

from ....search.rrt_connect import RRTConnect
from ..mappings import backward_time_gradient, forward_time, inverse_time
from ..minco import MINCOQuintic
from ..optimizer import scipy_lbfgs
from .config import AerialManipulatorGCOPTERConfig
from .evaluator import AerialManipulatorTrajectoryEvaluator
from .task_targets import yaw_quaternion
from .types import AerialManipulatorTrajectory


class AerialManipulatorGCOPTER:
    def __init__(self, config=None):
        self.config = AerialManipulatorGCOPTERConfig() if config is None else config

    def plan(self, start_state, goal_state, *, robot, esdf, quad, workspace_bounds,
             gripper_opening, carry_payload=False, rng=None):
        cfg = self.config
        start, goal = _state(start_state), _state(goal_state)
        bounds = np.asarray(workspace_bounds, dtype=float)
        if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("workspace_bounds must have shape (2, 3)")
        lower = np.r_[bounds[0], -np.pi, robot.limits.joint_lower]
        upper = np.r_[bounds[1], np.pi, robot.limits.joint_upper]
        evaluator = AerialManipulatorTrajectoryEvaluator(
            robot, esdf, quad, cfg, bounds, gripper_opening, carry_payload)

        def distance(a, b):
            delta = b-a
            delta[3] = _wrap(delta[3])
            scale = np.r_[np.full(3, cfg.position_scale), cfg.yaw_scale,
                          np.asarray(cfg.joint_scales)]
            return float(np.linalg.norm(delta/scale))

        def interpolate(a, b, alpha):
            delta = b-a
            delta[3] = _wrap(delta[3])
            result = a+alpha*delta
            result[3] = _wrap(result[3])
            return result

        def full_configuration(state):
            return np.r_[state[:3], yaw_quaternion(state[3]), state[4:8], gripper_opening]

        def state_valid(state):
            if (np.any(state[:3] < bounds[0]) or np.any(state[:3] > bounds[1])
                    or np.any(state[4:8] < robot.limits.joint_lower)
                    or np.any(state[4:8] > robot.limits.joint_upper)):
                return False
            try:
                result = robot.check_collision(
                    full_configuration(state), clearance=0.)
                if result["collision"]:
                    return False
                geometry = evaluator.collision_cost_gradient(state)
                return geometry[3] <= 0.0 and geometry[6] <= 0.0
            except ValueError:
                return False

        def edge_valid(a, b):
            delta = b-a; delta[3] = _wrap(delta[3])
            count = max(1, int(np.ceil(np.max(np.abs(delta[:3]))/cfg.edge_position_resolution)),
                        int(np.ceil(abs(delta[3])/cfg.edge_yaw_resolution)),
                        int(np.ceil(np.max(np.abs(delta[4:8]))/cfg.edge_joint_resolution)))
            return all(state_valid(interpolate(a, b, alpha))
                       for alpha in np.linspace(0., 1., count+1))

        search = RRTConnect(
            lower, upper, distance, interpolate, state_valid, edge_valid,
            cfg.rrt_step_size, max_iterations=cfg.rrt_max_iterations, rng=rng)
        path = search.simplify(search.plan(start, goal))
        path = _unwrap_path_yaw(path)
        knots = _resample_preserving_corners(path, max(cfg.pieces, len(path)-1))
        pieces = len(knots)-1
        total_time = _initial_duration(knots, cfg)
        minco = MINCOQuintic(_boundary_pva(start), _boundary_pva(goal), pieces)
        initial = np.r_[knots[1:-1].reshape(-1),
                        inverse_time(np.array([total_time-cfg.minimum_total_time]))]

        def objective(variables):
            return self._objective(variables, minco, pieces, evaluator)

        result = scipy_lbfgs(
            objective, initial, max_iterations=cfg.max_iterations,
            memory=cfg.lbfgs_memory, gradient_tolerance=cfg.gradient_tolerance,
            relative_cost_tolerance=1e-7,
            is_feasible=lambda: evaluator.last_violation <= 1.0e-4,
            feasible_iteration_patience=2)
        points = result.x[:8*(pieces-1)].reshape(pieces-1, 8)
        total = cfg.minimum_total_time+float(forward_time(result.x[-1:])[0])
        durations = np.full(pieces, total/pieces)
        coefficients, _ = minco.solve(points, durations)
        coefficients = coefficients.reshape(pieces, 6, 8)
        provisional = AerialManipulatorTrajectory(
            durations, coefficients, path, result.cost, result.iterations,
            result.converged, result.message)
        valid, clearance, violation = evaluator.dense_validate(provisional)
        return AerialManipulatorTrajectory(
            durations, coefficients, path, result.cost, result.iterations,
            result.converged, result.message, valid, clearance, violation,
            cfg.validation_dt)

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


__all__ = ["AerialManipulatorGCOPTER"]
