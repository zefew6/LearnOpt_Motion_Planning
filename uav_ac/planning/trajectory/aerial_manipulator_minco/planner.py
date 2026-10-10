"""Whole-body planning entrypoints using shared MINCO/L-BFGS mechanics."""

import time
import numpy as np
from collections.abc import Callable
from ...search.rrt_connect import plan_rrt_connect
from .config import AerialManipulatorMINCOConfig
from .constraints import AerialManipulatorTrajectoryEvaluator
from .search import (
    AerialManipulatorStateSpaceAdapter,
    plan_aerial_astar_guide,
    _greedy_shortcut,
    _interpolate_state,
    _resample_preserving_corners,
    _unwrap_path_yaw,
    _wrap,
)
from .trajectory import AerialManipulatorSearchResult, AerialManipulatorTrajectory
from .optimization import (
    AerialManipulatorOptimization,
    _boundary_pva,
    _decode_internal_joint_waypoints,
    _decode_tanh_joint_waypoints,
    _encode_internal_joint_waypoints,
    _encode_tanh_joint_waypoints,
    _initial_duration,
    _joint_center_radius,
    _segment_time_proportions,
    _tanh_joint_waypoint_derivative,
)
from .validation import (
    _is_retimeable_validation,
    _retiming_scale,
    _time_stretch,
)

class AerialManipulatorMINCO(AerialManipulatorOptimization):
    def __init__(self, config=None):
        self.config = AerialManipulatorMINCOConfig() if config is None else config
        self.last_metrics = {}

    def plan_task(self, start, targets, seeds, *, robot, esdf, quad, workspace_bounds,
                  gaps, occupancy=None, astar_maps=None, on_rrt_path=None, dwell_time=.3):
        """Joint fixed-target pick/place solve; searches only provide initialization."""
        from .optimization import plan_task
        plans, legs = plan_task(self, start, targets, seeds, robot=robot, esdf=esdf,
            quad=quad, workspace_bounds=workspace_bounds, gaps=gaps,
            occupancy=occupancy, astar_maps=astar_maps, on_rrt_path=on_rrt_path,
            dwell_time=dwell_time)
        return plans, legs

    def plan(self, start_state, goal_state, *, robot, esdf, quad, workspace_bounds,
             gripper_opening, carry_payload=False,
             occupancy=None, astar_maps=None, search_only=False,
             on_rrt_path: Callable[[np.ndarray], None] | None = None):
        started = time.perf_counter()
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
            occupancy=occupancy)
        metric_scale = np.r_[np.full(3, cfg.position_scale), cfg.yaw_scale,
                             np.asarray(cfg.joint_scales)]
        state_space_adapter = AerialManipulatorStateSpaceAdapter()
        guide = None
        if cfg.astar_guidance_enabled and astar_maps is not None:
            try:
                guide, astar_metrics = plan_aerial_astar_guide(
                    astar_maps, start[:3], goal[:3],
                    sample_spacing_m=cfg.astar_guide_sample_spacing_m,
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
                else "shared_astar_maps_unavailable",
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
            return bool(validate_states(np.asarray(state)[None, :])[0])

        def edge_valid(first, second):
            nonlocal rrt_edge_cache_hits
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
                timeout_s=None)
        except (RuntimeError, ValueError) as error:
            search_error = error
            search_metrics = getattr(error, "metrics", {})
        # Preserve search diagnostics on later optimization failures.
        rrt_seconds = time.perf_counter()-phase
        self.last_metrics = {
            **search_metrics,
            **astar_metrics,
            "rrt_seconds": rrt_seconds,
            "rrt_collision_model": ("manual_spheres_esdf" if evaluator.manual_spheres else
                                    "shared_edt_occupancy_full_envelope" if search_error else
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
            "rrt_broadphase_backend": ("esdf_spheres" if evaluator.manual_spheres else
                                       "native" if evaluator._broadphase.native else "numpy"),
            "rrt_broadphase_seconds": evaluator.rrt_broadphase_seconds,
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
        if on_rrt_path is not None:
            on_rrt_path(path.copy())
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
                    occupancy=occupancy)
                validity_cache.clear()
                edge_cache.clear()
                rrt_state_queries = rrt_state_cache_hits = 0
                rrt_collision_batches = rrt_edge_cache_hits = 0
            self.last_metrics = dict(search_diagnostics)
            try:
                trajectory = self._optimize_path(
                    path, knots, evaluator, robot, edge_valid, metric_scale,
                    shortcut_attempts)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
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


    def search_initial_path(self, *args, **kwargs):
        """Return only the exact RRT path and search diagnostics."""
        kwargs["search_only"] = True
        return self.plan(*args, **kwargs)


def _state(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (8,) or not np.all(np.isfinite(result)):
        raise ValueError("planning states must be finite 8-vectors")
    result = result.copy(); result[3] = _wrap(result[3])
    return result


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


__all__ = ["AerialManipulatorMINCO"]
