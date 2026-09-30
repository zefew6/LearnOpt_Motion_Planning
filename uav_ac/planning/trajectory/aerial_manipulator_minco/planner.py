"""8-D RRT-Connect initialization and equal-duration MINCO/L-BFGS refinement."""

import time

import numpy as np

from ...search.rrt_connect import plan_rrt_connect
from ..gcopter.mappings import backward_time_gradient, forward_time, inverse_time
from ..gcopter.minco import MINCOQuintic
from ..gcopter.optimizer import scipy_lbfgs
from .config import AerialManipulatorMINCOConfig
from .evaluator import AerialManipulatorTrajectoryEvaluator
from .search_adapter import (
    AerialManipulatorStateSpaceAdapter,
    plan_aerial_astar_guide,
)
from .types import AerialManipulatorSearchResult, AerialManipulatorTrajectory


_TANH_JOINT_INVERSE_MARGIN = 1.0e-6


class AerialManipulatorMINCO:
    def __init__(self, config=None):
        self.config = AerialManipulatorMINCOConfig() if config is None else config
        self.last_metrics = {}

    def plan(self, start_state, goal_state, *, robot, esdf, quad, workspace_bounds,
             gripper_opening, carry_payload=False, deadline=None,
             occupancy=None, search_only=False):
        started = time.perf_counter()
        deadline = (started+self.config.planning_budget_s
                    if deadline is None else float(deadline))

        def check_deadline():
            if time.perf_counter() >= deadline:
                raise TimeoutError("aerial_manipulator_minco planning budget exceeded")

        cfg = self.config
        rrt_started = time.perf_counter()
        start, goal = _state(start_state), _state(goal_state)
        bounds = np.asarray(workspace_bounds, dtype=float)
        if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("workspace_bounds must have shape (2, 3)")
        lower = np.r_[bounds[0], -np.pi, robot.limits.joint_lower]
        upper = np.r_[bounds[1], np.pi, robot.limits.joint_upper]
        evaluator = AerialManipulatorTrajectoryEvaluator(
            robot, esdf, quad, cfg, bounds, gripper_opening, carry_payload,
            deadline, occupancy=occupancy)
        metric_scale = np.r_[np.full(3, cfg.position_scale), cfg.yaw_scale,
                             np.asarray(cfg.joint_scales)]
        state_space_adapter = AerialManipulatorStateSpaceAdapter()
        guide = None
        if cfg.astar_guidance_enabled and occupancy is not None:
            try:
                guide, astar_metrics = plan_aerial_astar_guide(
                    occupancy, start[:3], goal[:3], esdf,
                    proxy_radius=robot.base_inscribed_collision_radius(),
                    margin=cfg.rrt_obstacle_margin,
                    grid_resolution=cfg.astar_grid_resolution,
                    fallback_resolution=cfg.astar_fallback_grid_resolution,
                    clearance_weight_m=cfg.astar_clearance_weight_m,
                    clearance_offset_m=cfg.astar_clearance_offset_m,
                    sample_spacing_m=cfg.astar_guide_sample_spacing_m,
                    clearance_error_m=cfg.esdf_discretization_margin,
                    heuristic_weight=cfg.astar_heuristic_weight)
            except (ValueError, RuntimeError, IndexError, FloatingPointError) as error:
                astar_metrics = {
                    "astar_seconds": 0.0,
                    "astar_expansions": 0,
                    "astar_grid_resolution": float(cfg.astar_grid_resolution),
                    "astar_route_length_m": None,
                    "astar_minimum_clearance_m": None,
                    "astar_fallback_used": False,
                    "astar_failure_reason": f"guide_error: {error}",
                }
        else:
            astar_metrics = {
                "astar_seconds": 0.0,
                "astar_expansions": 0,
                "astar_grid_resolution": float(cfg.astar_grid_resolution),
                "astar_route_length_m": None,
                "astar_minimum_clearance_m": None,
                "astar_fallback_used": False,
                "astar_failure_reason": "disabled" if not cfg.astar_guidance_enabled
                else "shared_occupancy_unavailable",
            }
        validity_cache = {}
        edge_cache = {}
        rrt_state_queries = 0
        rrt_state_cache_hits = 0
        rrt_collision_batches = 0
        rrt_edge_cache_hits = 0

        def validate_states(states):
            nonlocal rrt_state_queries, rrt_state_cache_hits, rrt_collision_batches
            states = np.asarray(states, dtype=float).reshape(-1, 8)
            result = np.empty(len(states), dtype=bool)
            pending, pending_keys, pending_indices = [], [], []
            rrt_state_queries += len(states)
            for index, state in enumerate(states):
                key = np.ascontiguousarray(state).tobytes()
                if key in validity_cache:
                    rrt_state_cache_hits += 1
                    result[index] = validity_cache[key]
                elif (np.any(state[:3] < bounds[0]) or np.any(state[:3] > bounds[1])
                      or np.any(state[4:8] < robot.limits.joint_lower)
                      or np.any(state[4:8] > robot.limits.joint_upper)):
                    validity_cache[key] = False
                    result[index] = False
                else:
                    pending.append(state.copy())
                    pending_keys.append(key)
                    pending_indices.append(index)
            for first in range(0, len(pending), 32):
                check_deadline()
                last = min(len(pending), first+32)
                valid = evaluator.collision_feasible_batch(
                    np.asarray(pending[first:last]), exact_candidates=True)
                rrt_collision_batches += 1
                for offset, is_valid in enumerate(valid, start=first):
                    key = pending_keys[offset]
                    result[pending_indices[offset]] = bool(is_valid)
                    validity_cache[key] = bool(is_valid)
            return result

        def state_valid(state):
            check_deadline()
            return bool(validate_states(np.asarray(state)[None, :])[0])

        def edge_valid(first, second):
            nonlocal rrt_edge_cache_hits
            check_deadline()
            first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
            first_key = first.copy()
            second_key = second.copy()
            first_key[3], second_key[3] = _wrap(first_key[3]), _wrap(second_key[3])
            first_key[first_key == 0.] = 0.
            second_key[second_key == 0.] = 0.
            first_bytes, second_bytes = first_key.tobytes(), second_key.tobytes()
            key = tuple(sorted((first_bytes, second_bytes)))
            if key in edge_cache:
                rrt_edge_cache_hits += 1
                return edge_cache[key]
            delta = second-first
            delta[3] = _wrap(delta[3])
            count = max(1,
                        int(np.ceil(np.linalg.norm(delta[:3])/cfg.edge_position_resolution)),
                        int(np.ceil(abs(delta[3])/cfg.edge_yaw_resolution)),
                        int(np.ceil(np.max(np.abs(delta[4:8]))/cfg.edge_joint_resolution)))
            fractions = np.arange(1, count+1, dtype=float)/count
            for first_index in range(0, len(fractions), 32):
                states = np.asarray([_interpolate_state(first, second, alpha)
                                     for alpha in fractions[first_index:first_index+32]])
                if not np.all(validate_states(states)):
                    edge_cache[key] = False
                    return False
            edge_cache[key] = True
            return True

        def sampling_regions():
            if not cfg.astar_guidance_enabled:
                return [{"name": "global_workspace",
                         "lower": lower.copy(), "upper": upper.copy(),
                         "fraction": 1.0}]
            endpoint_lower = np.maximum(
                bounds[0], np.minimum(start[:3], goal[:3])-.40)
            endpoint_upper = np.minimum(
                bounds[1], np.maximum(start[:3], goal[:3])+.40)
            def joint_bounds(padding):
                return (
                    np.maximum(lower[4:8],
                               np.minimum(start[4:8], goal[4:8])-padding),
                    np.minimum(upper[4:8],
                               np.maximum(start[4:8], goal[4:8])+padding),
                )

            # OMPL 2.0.1's Python bindings omit StateSpace.setStateSamplerAllocator,
            # so approximate the intended mixture with ordered sampling-bound
            # stages on one persistent RRTConnect tree. Try the A* corridor
            # first; endpoint and global bounds remain recovery stages.
            endpoint_joint_lower, endpoint_joint_upper = joint_bounds(
                cfg.rrt_joint_sampling_padding_rad)
            endpoint_region = {"name": "valid_endpoint_region",
                               "lower": np.r_[endpoint_lower, lower[3], endpoint_joint_lower],
                               "upper": np.r_[endpoint_upper, upper[3], endpoint_joint_upper],
                               "fraction": .40}
            if guide is not None:
                route_lower = np.maximum(
                    bounds[0], np.min(guide.path, axis=0)-3.*cfg.astar_tube_std_m)
                route_upper = np.minimum(
                    bounds[1], np.max(guide.path, axis=0)+3.*cfg.astar_tube_std_m)
                route_joint_lower, route_joint_upper = joint_bounds(
                    2.*cfg.rrt_joint_sampling_padding_rad)
                route_region = {"name": "astar_route_region",
                                "lower": np.r_[route_lower, lower[3], route_joint_lower],
                                "upper": np.r_[route_upper, upper[3], route_joint_upper],
                                "fraction": .40}
                global_fraction = .20
                regions = [route_region, endpoint_region]
            else:
                global_fraction = .60
                regions = [endpoint_region]
            regions.append({"name": "global_workspace",
                            "lower": lower.copy(), "upper": upper.copy(),
                            "fraction": global_fraction})
            return regions

        regions = []
        phase = rrt_started
        search_error = None
        try:
            if not state_valid(start) or not state_valid(goal):
                raise ValueError("OMPL RRT start and goal must be valid states")
            regions = sampling_regions()
            path, search_metrics = plan_rrt_connect(
                state_space_adapter, start, goal, lower, upper, metric_scale,
                state_valid=state_valid, edge_valid=edge_valid,
                range_size=cfg.rrt_step_size,
                sampling_regions=regions,
                simplify_attempts=cfg.rrt_simplify_attempts,
                simplify_budget_s=cfg.rrt_simplify_budget_s,
                timeout_s=max(.001, deadline-time.perf_counter()))
        except (RuntimeError, ValueError, TimeoutError) as error:
            search_error = error
            search_metrics = getattr(error, "metrics", {})
        # Preserve search diagnostics on later optimization failures.
        rrt_seconds = time.perf_counter()-phase
        self.last_metrics = {
            **search_metrics,
            **astar_metrics,
            "rrt_seconds": rrt_seconds,
            "rrt_collision_model": ("shared_edt_occupancy_full_envelope"
                                    if search_error else
                                    "ompl_shared_edt_occupancy_full_envelope"),
            "rrt_occupancy_margin": float(cfg.rrt_obstacle_margin),
            "rrt_state_queries": rrt_state_queries,
            "rrt_state_cache_hits": rrt_state_cache_hits,
            "rrt_collision_batches": rrt_collision_batches,
            "rrt_edge_cache_hits": rrt_edge_cache_hits,
            "rrt_sampling_guidance_mode": "progressive_position_joint_bounds",
            "rrt_sampling_regions": [region["name"] for region in regions],
            "rrt_exact_candidate_states": evaluator.rrt_exact_candidate_states,
            "rrt_exact_candidate_seconds": evaluator.rrt_exact_candidate_seconds,
            "rrt_batch_fk_seconds": evaluator.rrt_batch_fk_seconds,
            "rrt_occupancy_check_seconds": evaluator.rrt_occupancy_check_seconds,
            "rrt_sphere_pair_check_seconds": evaluator.rrt_sphere_pair_check_seconds,
            "rrt_world_pair_filter_seconds": evaluator.rrt_world_pair_filter_seconds,
            "rrt_exact_geometry_seconds": evaluator.rrt_exact_geometry_seconds,
            "rrt_exact_pair_queries": evaluator.rrt_exact_pair_queries,
            "rrt_environment_candidates": evaluator.rrt_environment_candidates,
            "rrt_self_candidates": evaluator.rrt_self_candidates,
            "rrt_fixed_clearance_self_pairs_skipped":
                evaluator.rrt_fixed_clearance_self_pairs_skipped,
            "rrt_payload_candidates": evaluator.rrt_payload_candidates,
            "rrt_exact_rejected_states": evaluator.rrt_exact_rejected_states,
        }
        if search_error is not None:
            self.last_metrics.update(
                failure_reason=str(search_error), total_seconds=time.perf_counter()-started)
            raise search_error
        if search_only:
            self.last_metrics["rrt_path_states"] = int(len(path))
            return AerialManipulatorSearchResult(path, self.last_metrics)
        path = _unwrap_path_yaw(path)
        path, shortcut_attempts = _greedy_shortcut(path, edge_valid)
        knots = _resample_preserving_corners(
            path, cfg.minco_sample_spacing_m, cfg.position_scale,
            cfg.yaw_scale, cfg.joint_scales)
        search_diagnostics = dict(self.last_metrics)
        attempts = []
        trajectory = None

        for attempt_index in range(2):
            if attempt_index:
                evaluator = AerialManipulatorTrajectoryEvaluator(
                    robot, esdf, quad, cfg, bounds, gripper_opening, carry_payload,
                    deadline, occupancy=occupancy)
                validity_cache.clear()
                edge_cache.clear()
                rrt_state_queries = rrt_state_cache_hits = 0
                rrt_collision_batches = rrt_edge_cache_hits = 0
            self.last_metrics = dict(search_diagnostics)
            try:
                trajectory = self._optimize_path(
                    path, knots, evaluator, robot, edge_valid, metric_scale,
                    deadline, shortcut_attempts)
            except (RuntimeError, TimeoutError, ValueError, np.linalg.LinAlgError) as error:
                attempts.append(dict(self.last_metrics))
                self.last_metrics = _combined_metrics(
                    search_diagnostics, attempts, time.perf_counter()-started)
                self.last_metrics["failure_reason"] = str(error)
                raise
            attempts.append(dict(self.last_metrics))
            if trajectory.validation_passed or attempt_index:
                break

            failing_piece = int(evaluator.last_validation_metrics.get(
                "maximum_violation_piece", -1))
            if not 0 <= failing_piece < len(knots)-1:
                break
            midpoint = _interpolate_state(
                knots[failing_piece], knots[failing_piece+1], .5)
            if not (edge_valid(knots[failing_piece], midpoint)
                    and edge_valid(midpoint, knots[failing_piece+1])):
                break
            path = np.insert(knots, failing_piece+1, midpoint, axis=0)
            knots = path.copy()

        self.last_metrics = _combined_metrics(
            search_diagnostics, attempts, time.perf_counter()-started)
        return trajectory

    def _optimize_path(self, path, knots, evaluator, robot, edge_valid,
                       metric_scale, deadline, shortcut_attempts):
        cfg = self.config

        def check_deadline():
            if time.perf_counter() >= deadline:
                raise TimeoutError("aerial_manipulator_minco planning budget exceeded")

        pieces = len(knots)-1
        time_proportions = _segment_time_proportions(knots, cfg)
        initial_duration_scale = cfg.initial_duration_scale
        total_time = _initial_duration(knots, cfg)*initial_duration_scale
        minco = MINCOQuintic(_boundary_pva(knots[0]), _boundary_pva(knots[-1]), pieces)
        joint_lower = np.asarray(robot.limits.joint_lower, dtype=float)
        joint_upper = np.asarray(robot.limits.joint_upper, dtype=float)
        waypoint_parameterization = cfg.joint_waypoint_parameterization
        initial_points = _encode_internal_joint_waypoints(
            knots[1:-1], waypoint_parameterization, joint_lower, joint_upper)
        # An RRT knot exactly on a joint limit is moved inward only to form a
        # finite atanh initialization.  Verify the resulting physical edges
        # rather than silently assuming that the tiny move retained clearance.
        mapped_knots = np.vstack((
            knots[:1],
            _decode_internal_joint_waypoints(
                initial_points, waypoint_parameterization,
                joint_lower, joint_upper),
            knots[-1:],
        ))
        if not all(edge_valid(first, second) for first, second in zip(
                mapped_knots[:-1], mapped_knots[1:], strict=True)):
            if waypoint_parameterization == "tanh":
                # A path knot on a physical limit can lose a few microns of
                # clearance when it is moved inward for atanh.  Preserve the
                # validated RRT path and use the direct physical coordinate
                # for this plan; the default remains tanh whenever mapping is
                # well-conditioned.
                waypoint_parameterization = "direct"
                initial_points = knots[1:-1].copy()
                mapped_knots = knots.copy()
            else:
                raise RuntimeError("joint-limit waypoint inverse mapping invalidated RRT path")
        initial = np.r_[initial_points.reshape(-1),
                        inverse_time(np.array([total_time-cfg.minimum_total_time]))]
        # Optimize dimensionless coordinates so meter, radian, and mapped-time
        # variables present comparable step sizes to L-BFGS. The log-time
        # coordinate is already dimensionless and uses unit scale.
        waypoint_variable_scale = metric_scale.copy()
        if waypoint_parameterization == "tanh":
            # The tanh coordinates are already dimensionless.  Applying the
            # physical joint-angle scale here would distort the chain-rule
            # gradient sent to L-BFGS.
            waypoint_variable_scale[4:8] = 1.0
        variable_scale = np.r_[
            np.tile(waypoint_variable_scale*cfg.optimizer_waypoint_step_scale,
                    pieces-1),
            cfg.optimizer_waypoint_step_scale,
        ]
        objective_calls = 0
        # Publish the optimizer setup before its first objective evaluation so
        # a shared planning deadline still leaves useful diagnostics behind.
        self.last_metrics.update({
            "minco_sample_spacing_m": float(cfg.minco_sample_spacing_m),
            "minco_initial_waypoints": int(len(knots)),
            "minco_pieces": int(pieces),
            "initial_duration_scale": initial_duration_scale,
            "objective_calls": 0,
            "objective_samples": 0,
            "optimizer_seconds": 0.0,
            "optimizer_status": "not_started",
        })

        def objective(variables):
            nonlocal objective_calls
            check_deadline()
            objective_calls += 1
            return self._objective(
                variables, minco, pieces, evaluator,
                joint_lower=joint_lower, joint_upper=joint_upper,
                joint_parameterization=waypoint_parameterization,
                time_proportions=time_proportions)

        def scaled_objective(scaled_variables):
            cost, gradient = objective(scaled_variables*variable_scale)
            return cost, gradient*variable_scale

        phase = time.perf_counter()
        try:
            result = scipy_lbfgs(
                scaled_objective, initial/variable_scale,
                max_iterations=cfg.max_iterations,
                memory=cfg.lbfgs_memory, gradient_tolerance=cfg.gradient_tolerance,
                relative_cost_tolerance=cfg.relative_cost_tolerance,
                # Require all optimization quadrature samples to clear their
                # constraints; dense full-geometry validation below remains
                # authoritative between quadrature samples.
                is_feasible=lambda: evaluator.last_violation <= 0.0,
                feasible_iteration_patience=10_000,
                require_convergence=True)
        except (RuntimeError, TimeoutError, ValueError, np.linalg.LinAlgError):
            self.last_metrics.update({
                "objective_calls": int(objective_calls),
                "objective_samples": int(evaluator.objective_samples),
                "optimizer_seconds": time.perf_counter()-phase,
                "optimizer_status": "failed_or_timed_out",
                "optimizer_last_violation": float(evaluator.last_violation),
            })
            raise
        optimizer_seconds = time.perf_counter()-phase
        relative_cost_change = (float(result.relative_cost_change)
                                if np.isfinite(result.relative_cost_change) else None)
        self.last_metrics.update({
            "optimizer_seconds": optimizer_seconds,
            "objective_calls": int(objective_calls),
            "objective_samples": int(evaluator.objective_samples),
            "optimizer_status": result.message,
            "optimizer_converged": bool(result.converged),
            "optimizer_iterations": int(result.iterations),
            "optimizer_gradient_inf_norm": float(np.linalg.norm(result.gradient, ord=np.inf)),
            "optimizer_relative_cost_change": relative_cost_change,
            "optimizer_last_violation": float(evaluator.last_violation),
        })
        optimized_variables = result.x*variable_scale
        points = _decode_internal_joint_waypoints(
            optimized_variables[:8*(pieces-1)].reshape(pieces-1, 8),
            waypoint_parameterization, joint_lower, joint_upper)
        total = cfg.minimum_total_time+float(forward_time(optimized_variables[-1:])[0])
        durations = time_proportions*total
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
        self.last_metrics.update({
            "esdf_discretization_margin_used": evaluator.esdf_discretization_margin,
            "rrt_greedy_shortcut_attempts": int(shortcut_attempts),
            "validation_seconds": validation_seconds,
            "validation_samples": evaluator.validation_samples,
            "optimized_total_time_s": float(provisional.total_time),
            "retiming_attempts": retiming_attempts,
            "retiming_scale": retiming_scale,
            "joint_waypoint_parameterization_used": waypoint_parameterization,
        })
        self.last_metrics.update(evaluator.last_validation_metrics)
        trajectory = AerialManipulatorTrajectory(
            provisional.durations, provisional.coefficients, path,
            result.cost, result.iterations,
            result.converged, result.message, valid, clearance, violation,
            sample_dt)
        return trajectory


    def search_initial_path(self, *args, **kwargs):
        """Return only the exact RRT path and search diagnostics."""
        kwargs["search_only"] = True
        return self.plan(*args, **kwargs)


    def _objective(self, variables, minco, pieces, evaluator, *,
                   joint_lower=None, joint_upper=None,
                   joint_parameterization=None, time_proportions=None):
        cfg = self.config
        joint_parameterization = (cfg.joint_waypoint_parameterization
                                  if joint_parameterization is None
                                  else joint_parameterization)
        variables = np.asarray(variables, dtype=float)
        waypoint_count = pieces-1
        raw_points = variables[:8*waypoint_count].reshape(waypoint_count, 8)
        if joint_parameterization == "tanh":
            if joint_lower is None or joint_upper is None:
                joint_lower = evaluator.robot.limits.joint_lower
                joint_upper = evaluator.robot.limits.joint_upper
            points = _decode_internal_joint_waypoints(
                raw_points, joint_parameterization,
                joint_lower, joint_upper)
        else:
            points = raw_points
        tau = variables[-1:]
        total = cfg.minimum_total_time+float(forward_time(tau)[0])
        time_proportions = (np.full(pieces, 1./pieces) if time_proportions is None
                            else np.asarray(time_proportions, dtype=float))
        if (time_proportions.shape != (pieces,) or np.any(time_proportions <= 0.)
                or not np.isclose(np.sum(time_proportions), 1.)):
            raise ValueError("time_proportions must be positive and sum to one")
        durations = time_proportions*total
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
        grad_total = float(np.dot(grad_times, time_proportions))
        grad_tau = backward_time_gradient(tau, np.array([grad_total]))
        if joint_parameterization == "tanh":
            # ``grad_points`` is the physical waypoint gradient after the
            # MINCO adjoint.  Map it once into unconstrained tanh coordinates.
            grad_points = grad_points.copy()
            grad_points[:, 4:8] *= _tanh_joint_waypoint_derivative(
                raw_points[:, 4:8], joint_lower, joint_upper)
        gradient = np.r_[grad_points.reshape(-1), grad_tau]
        if not np.isfinite(cost) or not np.all(np.isfinite(gradient)):
            return 1.0e30, np.zeros_like(variables)
        return float(cost), gradient


def _joint_center_radius(joint_lower, joint_upper):
    lower = np.asarray(joint_lower, dtype=float)
    upper = np.asarray(joint_upper, dtype=float)
    if (lower.shape != (4,) or upper.shape != (4,)
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(upper <= lower)):
        raise ValueError("joint limits must be four finite, strictly ordered values")
    return .5*(lower+upper), .5*(upper-lower)


def _tanh_joint_waypoint_derivative(unconstrained, joint_lower, joint_upper):
    """Derivative of physical joint angles with respect to tanh coordinates."""
    unconstrained = np.asarray(unconstrained, dtype=float)
    if unconstrained.shape[-1:] != (4,):
        raise ValueError("unconstrained joint waypoints must end in four coordinates")
    _, radius = _joint_center_radius(joint_lower, joint_upper)
    tangent = np.tanh(unconstrained)
    return radius*(1.0-tangent*tangent)


def _decode_tanh_joint_waypoints(unconstrained, joint_lower, joint_upper):
    """Map four unconstrained joint coordinates into the open joint-limit box."""
    unconstrained = np.asarray(unconstrained, dtype=float)
    if unconstrained.shape[-1:] != (4,) or not np.all(np.isfinite(unconstrained)):
        raise ValueError("unconstrained joint waypoints must be finite four-vectors")
    center, radius = _joint_center_radius(joint_lower, joint_upper)
    return center+radius*np.tanh(unconstrained)


def _encode_tanh_joint_waypoints(physical, joint_lower, joint_upper):
    """Safely inverse-map physical joints for an RRT-derived initialization.

    MINCO endpoints remain exact physical states.  Only interior RRT knots are
    represented here, so a knot exactly on a limit is moved inward by the
    documented small amount before applying ``arctanh``.
    """
    physical = np.asarray(physical, dtype=float)
    if physical.shape[-1:] != (4,) or not np.all(np.isfinite(physical)):
        raise ValueError("physical joint waypoints must be finite four-vectors")
    center, radius = _joint_center_radius(joint_lower, joint_upper)
    lower, upper = center-radius, center+radius
    if (np.any(physical < lower-_TANH_JOINT_INVERSE_MARGIN)
            or np.any(physical > upper+_TANH_JOINT_INVERSE_MARGIN)):
        raise ValueError("physical joint waypoint exceeds its limit")
    margin = np.minimum(_TANH_JOINT_INVERSE_MARGIN, .25*(upper-lower))
    interior = np.clip(physical, lower+margin, upper-margin)
    return np.arctanh((interior-center)/radius)


def _decode_internal_joint_waypoints(raw_points, parameterization,
                                     joint_lower, joint_upper):
    """Return physical 8-D MINCO waypoints from optimizer coordinates."""
    raw_points = np.asarray(raw_points, dtype=float)
    if raw_points.ndim != 2 or raw_points.shape[1] != 8:
        raise ValueError("internal waypoints must have shape (count, 8)")
    if parameterization == "direct":
        return raw_points
    if parameterization != "tanh":
        raise ValueError("unknown joint waypoint parameterization")
    points = raw_points.copy()
    points[:, 4:8] = _decode_tanh_joint_waypoints(
        raw_points[:, 4:8], joint_lower, joint_upper)
    return points


def _encode_internal_joint_waypoints(physical_points, parameterization,
                                     joint_lower, joint_upper):
    """Encode physical RRT knots without changing position, yaw, or endpoints."""
    physical_points = np.asarray(physical_points, dtype=float)
    if physical_points.ndim != 2 or physical_points.shape[1] != 8:
        raise ValueError("internal waypoints must have shape (count, 8)")
    if parameterization == "direct":
        return physical_points.copy()
    if parameterization != "tanh":
        raise ValueError("unknown joint waypoint parameterization")
    raw_points = physical_points.copy()
    raw_points[:, 4:8] = _encode_tanh_joint_waypoints(
        physical_points[:, 4:8], joint_lower, joint_upper)
    return raw_points


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


def _interpolate_state(first, second, alpha):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    delta = second-first
    delta[3] = _wrap(delta[3])
    result = first+float(alpha)*delta
    result[3] = _wrap(result[3])
    return result


def _unwrap_path_yaw(path):
    result = np.asarray(path, dtype=float).copy()
    for i in range(1, len(result)):
        original = result[i, 3]
        result[i, 3] = original+2.*np.pi*round(
            (result[i-1, 3]-original)/(2.*np.pi))
    return result


def _greedy_shortcut(path, edge_valid):
    """Remove redundant route knots while validating every replacement edge."""
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path.copy(), 0
    result, current, attempts = [path[0].copy()], 0, 0
    while current < len(path)-1:
        following = len(path)-1
        while following > current+1:
            attempts += 1
            if edge_valid(path[current], path[following]):
                break
            following -= 1
        result.append(path[following].copy())
        current = following
    return _unwrap_path_yaw(np.asarray(result)), attempts


def _combined_metrics(search_metrics, attempts, total_seconds):
    metrics = dict(search_metrics)
    if attempts:
        metrics.update(attempts[-1])
        for key in ("objective_calls", "objective_samples", "optimizer_iterations",
                    "validation_samples"):
            metrics[key] = sum(int(attempt.get(key, 0)) for attempt in attempts)
        for key in ("optimizer_seconds", "validation_seconds"):
            metrics[key] = sum(float(attempt.get(key, 0.)) for attempt in attempts)
        if len(attempts) > 1:
            metrics["optimizer_refinement_attempts"] = len(attempts)-1
    metrics["total_seconds"] = float(total_seconds)
    return metrics


def _resample_preserving_corners(path, spacing, position_scale=.5,
                                 yaw_scale=.7, joint_scales=(.8, .8, .8, .8)):
    """Sample an 8-D RRT polyline by equivalent arc length, retaining corners."""
    if len(path) < 2:
        raise ValueError("RRT path must have distinct start and goal entries")
    path = _unwrap_path_yaw(path)
    path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0., axis=1)]]
    if not np.isfinite(spacing) or spacing <= 0.:
        raise ValueError("MINCO sample spacing must be finite and positive")
    if len(path) < 2:
        raise ValueError("RRT path must contain distinct 8-D start and goal states")
    position_scale, yaw_scale = float(position_scale), float(yaw_scale)
    joint_scales = np.asarray(joint_scales, dtype=float)
    if (position_scale <= 0. or yaw_scale <= 0. or joint_scales.shape != (4,)
            or np.any(joint_scales <= 0.)):
        raise ValueError("8-D path metric scales must be positive")
    scale = np.r_[np.full(3, position_scale), yaw_scale, joint_scales]
    delta = np.diff(path, axis=0)
    edge_lengths = position_scale*np.sqrt(np.sum((delta/scale)**2, axis=1))
    if np.sum(edge_lengths) <= 1e-12:
        raise ValueError("RRT start and goal have zero 8-D path length")
    knots = [path[0].copy()]
    for edge, length in enumerate(edge_lengths):
        count = max(1, int(np.ceil(length/float(spacing))))
        knots.extend((path[edge+1].copy() if step == count else
                      path[edge]+(path[edge+1]-path[edge])*(step/count))
                     for step in range(1, count+1))
    return np.asarray(knots)


def _segment_time_proportions(path, config):
    delta = np.abs(np.diff(np.asarray(path, dtype=float), axis=0))
    estimates = np.maximum.reduce((
        np.linalg.norm(delta[:, :3], axis=1)/config.max_speed,
        delta[:, 3]/config.max_yaw_rate,
        np.max(delta[:, 4:8]/np.asarray(config.joint_velocity_limits), axis=1),
        np.full(len(delta), .05),
    ))
    return estimates/np.sum(estimates)


def _initial_duration(path, config):
    delta = np.abs(np.diff(np.asarray(path, dtype=float), axis=0))
    position_length = float(np.sum(np.linalg.norm(delta[:, :3], axis=1)))
    yaw_length = float(np.sum(delta[:, 3]))
    joint_lengths = np.sum(delta[:, 4:8], axis=0)
    velocity_time = max(
        position_length/config.max_speed,
        yaw_length/config.max_yaw_rate,
        float(np.max(joint_lengths/np.asarray(config.joint_velocity_limits))),
    )
    acceleration_time = max(
        np.sqrt(6.0*position_length/config.max_acceleration),
        np.sqrt(6.0*yaw_length/config.max_yaw_acceleration),
        float(np.max(np.sqrt(
            6.0*joint_lengths/np.asarray(config.joint_acceleration_limits)))),
    )
    return max(velocity_time, acceleration_time, config.minimum_total_time+.25)


__all__ = ["AerialManipulatorMINCO"]
