"""Joint task-variable mapping; numerical spline mechanics live in GCOPTER."""

import numpy as np
import time
from dataclasses import replace
from ..gcopter.mappings import (
    forward_time,
    inverse_time,
    backward_time_gradient,
)
from ..gcopter.minco import MINCOQuintic
from ..gcopter.optimization import (
    evaluate_minco_objective,
    evaluate_minco_constraints,
    boundary_derivative_jacobian,
    FixedWaypointMap,
    FixedBoundaryDerivativeMap,
)
from .flatness import stationary_waypoint
from .trajectory import AerialManipulatorTrajectory
from .validation import (
    _is_retimeable_validation,
    _retiming_scale,
    _time_stretch,
)
from .constraints import task_residual, TaskEventConstraint
from ..gcopter.optimizer import scipy_lbfgs, solve_equalities

def plan_task(planner, start, targets, seeds, *, robot, esdf, quad,
              workspace_bounds, gaps, occupancy=None, astar_maps=None, on_rrt_path=None, dwell_time=.3):
    from .search import task_initialization
    from .constraints import AerialManipulatorTrajectoryEvaluator
    paths, searches = task_initialization(planner, start, seeds, robot=robot,
        esdf=esdf, quad=quad, workspace_bounds=workspace_bounds, gaps=gaps,
        occupancy=occupancy, astar_maps=astar_maps, on_rrt_path=on_rrt_path)
    evaluators = [AerialManipulatorTrajectoryEvaluator(robot, esdf, quad, planner.config,
        workspace_bounds, gap, bool(i), occupancy=occupancy) for i, gap in enumerate(gaps)]
    problem = JointTaskObjective(robot, planner.config, start, targets, paths, evaluators, gaps,
                                 dwell_time=dwell_time)
    calls = 0
    def objective(x):
        nonlocal calls
        calls += 1
        return problem.objective(x)
    from types import SimpleNamespace
    started = time.perf_counter()
    if any(target.orientation is not None for target in targets):
        constrained = solve_equalities(objective, problem.initial, problem.extra_equalities,
            tolerance=1e-3/np.sqrt(3), max_iterations=planner.config.max_iterations)
        numerical = SimpleNamespace(x=constrained.x, iterations=constrained.iterations,
            converged=constrained.feasible, message=constrained.message)
    else:
        numerical = scipy_lbfgs(objective, problem.initial,
            max_iterations=planner.config.max_iterations, memory=planner.config.lbfgs_memory,
            gradient_tolerance=planner.config.gradient_tolerance,
            relative_cost_tolerance=planner.config.relative_cost_tolerance,
            is_feasible=lambda: all(e.last_violation <= 3e-3 for e in evaluators),
            require_convergence=True)
    seconds = time.perf_counter()-started
    residual, _ = problem.equalities(numerical.x)
    extra, _ = problem.extra_equalities(numerical.x)
    residual = np.r_[residual, extra]
    result = SimpleNamespace(x=numerical.x, residual=residual,
        feasible=bool(np.max(np.abs(residual), initial=0.) <= 1e-3),
        converged=numerical.converged, iterations=numerical.iterations, message=numerical.message)
    plans = problem.trajectories(result.x, result)
    calls = getattr(problem, 'objective_calls', calls)
    metrics = {}
    for i, name in enumerate(('pick', 'place')):
        evaluator = evaluators[i]
        metrics[name] = dict(searches[i].metrics)
        metrics[name].update(validation_performed=False, validation_passed=None,
            sampled_maximum_violation=float(evaluator.last_violation),
            sampled_minimum_clearance=float(evaluator.minimum_clearance),
            objective_samples=evaluator.objective_samples,
            optimizer_seconds=seconds, joint_optimizer_calls=calls,
            joint_problem=True, task_equality_feasible=result.feasible)
    planner.last_metrics = dict(legs=metrics, joint_problem=True,
        joint_optimizer_calls=calls, optimizer_seconds=seconds,
        joint_optimizer_restarts=0, joint_solver_runs=1, validation_performed=False,
        task_equality_residual=result.residual.tolist(), task_equality_feasible=result.feasible)
    return dict(zip(('pick', 'place'), plans)), metrics


class JointTaskObjective:
    """Two movement blocks sharing an optimized stationary grasp configuration.

    Fixed task positions eliminate base translations. Event yaw/joints remain
    free. Zero p/v/a boundaries plus zero lateral jerk give stationary end
    effectors under the flatness map; the jerk equalities use analytic adjoints.
    """

    def __init__(self, robot, config, start, targets, knots, evaluators, gaps, *, dwell_time=0.):
        self.robot, self.config, self.start = robot, config, np.array(start, dtype=float)
        self.effort_scale = 1.
        self.objective_calls = 0
        self.dwell_time = dwell_time
        self.targets, self.knots, self.evaluators, self.gaps = targets, knots, evaluators, gaps
        for target in targets:
            if not np.all(target.position_mask):
                raise ValueError('stationary pick/place requires all position components')
            if any(getattr(target, name) is not None
                   and np.any(np.abs(getattr(target, name)[getattr(target, name+'_mask')]) > 1e-12)
                   for name in ('linear_velocity', 'angular_velocity')):
                raise ValueError('grasp/release dwell requires zero task velocity; use TaskEventConstraint for moving tasks')
        self.lower = np.asarray(robot.limits.joint_lower)
        self.upper = np.asarray(robot.limits.joint_upper)
        self.parameterization = config.joint_waypoint_parameterization
        raw_events = _encode_internal_joint_waypoints(
            np.stack([path[-1] for path in knots]), self.parameterization, self.lower, self.upper)
        values, scales = [raw_events[:, 3:].ravel()], [np.tile(np.r_[config.yaw_scale, np.ones(4)], 2)]
        self.layouts, self.point_maps, self.derivative_maps = [], [], []
        offset = 10
        for path in knots:
            count = len(path)-1
            raw = _encode_internal_joint_waypoints(path[1:-1], self.parameterization, self.lower, self.upper)
            dependent = np.zeros_like(raw, dtype=bool)
            dependent[[0, -1], :2] = True
            point_map = FixedWaypointMap(raw, dependent)
            self.point_maps.append(point_map)
            self.derivative_maps.append(FixedBoundaryDerivativeMap(dependent))
            point_slice = slice(offset, offset+point_map.dimension); offset += point_map.dimension
            time_slice = slice(offset, offset+count); offset += count
            times = _segment_time_proportions(path, config)*_initial_duration(path, config)*config.initial_duration_scale
            minimum = config.minimum_total_time/count
            values.extend([point_map.encode(raw), inverse_time(np.maximum(times-minimum, .01))])
            point_scales = np.r_[np.full(3, config.position_scale), config.yaw_scale,
                                 np.ones(4) if self.parameterization == 'tanh' else config.joint_scales]
            scales.extend([point_map.encode(np.tile(point_scales, (count-1, 1))), np.ones(count)])
            self.layouts.append((point_slice, time_slice, count, minimum))
        self.scale = np.concatenate(scales)*config.optimizer_waypoint_step_scale
        # Positive-time coordinates are already dimensionless. The spatial
        # waypoint preconditioner must not shrink independent time steps.
        for _, time_slice, _, _ in self.layouts:
            self.scale[time_slice] = 1.
        self.initial = np.concatenate(values)/self.scale

    def decode(self, x):
        raw = np.asarray(x)*self.scale
        events_raw = np.zeros((2, 8)); events_raw[:, 3:] = raw[:10].reshape(2, 5)
        events = _decode_internal_joint_waypoints(events_raw, self.parameterization, self.lower, self.upper)
        states, derivatives = [], []
        for i in range(2):
            state, jacobian = stationary_waypoint(self.robot, self.targets[i], events[i, 3:], self.gaps[i])
            if self.parameterization == 'tanh':
                jacobian[:, 1:] *= _tanh_joint_waypoint_derivative(events_raw[i:i+1, 4:], self.lower, self.upper)[0]
            states.append(state); derivatives.append(jacobian)
        blocks = []
        self._projections = []
        for i, (ps, ts, count, minimum) in enumerate(self.layouts):
            points = _decode_internal_joint_waypoints(self.point_maps[i].decode(raw[ps]),
                        self.parameterization, self.lower, self.upper)
            times = minimum + forward_time(raw[ts])
            block = (self.start if i == 0 else states[0], states[i], points, times)
            points, projection = self.derivative_maps[i].project(self._minco(block), points, times)
            blocks.append((block[0], block[1], points, times))
            self._projections.append(projection)
        return blocks, derivatives, raw

    def _pullback(self, i, gp, gt, gh, ge, derivatives, raw):
        gp, gt, gh, ge = self.derivative_maps[i].pullback(
            gp, gt, gh, ge, self._projections[i])
        gradient = np.zeros_like(raw)
        gradient[i*5:i*5+5] += ge[0] @ derivatives[i]
        if i:
            gradient[:5] += gh[0] @ derivatives[0]
        ps, ts, count, _ = self.layouts[i]
        gp = gp.copy()
        if self.parameterization == 'tanh':
            gp[:, 4:] *= _tanh_joint_waypoint_derivative(
                self.point_maps[i].decode(raw[ps])[:, 4:], self.lower, self.upper)
        gradient[ps] = self.point_maps[i].pullback(gp)
        gradient[ts] = backward_time_gradient(raw[ts], gt)
        return gradient*self.scale

    @staticmethod
    def _minco(block):
        start, end, points, times = block
        head, tail = np.zeros((3, 8)), np.zeros((3, 8))
        head[0], tail[0] = start, end
        return MINCOQuintic(head, tail, len(times))

    def objective(self, x):
        self.objective_calls += 1
        blocks, derivatives, raw = self.decode(x)
        cost, gradient = 0., np.zeros_like(raw)
        block_costs = []
        for i, block in enumerate(blocks):
            minco = self._minco(block)
            value, gp, gt, _, gh, ge = evaluate_minco_objective(minco, block[2], block[3],
                terms=(self.evaluators[i].integrated_penalty,), time_weight=self.config.time_weight*self.effort_scale,
                energy_weights=np.asarray(self.config.jerk_weights)*self.effort_scale, boundary_gradients=True)
            cost += value
            block_costs.append(float(value))
            gradient += self._pullback(i, gp, gt, gh, ge, derivatives, raw)
        if self.dwell_time > 0.:
            from .trajectory import gripper_gap_motion
            fractions = np.linspace(0., 1., 5)
            weights = np.array([.5, 1., 1., 1., .5])*self.dwell_time/4
            for i, evaluator in enumerate(self.evaluators):
                old_gap = evaluator.gripper_opening
                start, end = (self.gaps[0], self.gaps[1]) if i == 0 else (self.gaps[1], self.gaps[0])
                try:
                    for alpha, weight in zip(fractions, weights, strict=True):
                        evaluator.gripper_opening = gripper_gap_motion(start, end, alpha*self.dwell_time, self.dwell_time)[0]
                        value, grads, violation, clearance = evaluator.sample_cost_gradient(
                            blocks[i][1], np.zeros(8), np.zeros(8), np.zeros(8))
                        cost += weight*value
                        block_costs[i] += weight*value
                        gradient[i*5:i*5+5] += weight*(grads[0] @ derivatives[i])*self.scale[i*5:i*5+5]
                        evaluator.last_violation = max(evaluator.last_violation, violation)
                        evaluator.minimum_clearance = min(evaluator.minimum_clearance, clearance)
                finally:
                    evaluator.gripper_opening = old_gap
            cost += 2*self.dwell_time*self.config.time_weight*self.effort_scale
            block_costs = [value+self.dwell_time*self.config.time_weight*self.effort_scale
                           for value in block_costs]
        self.last_block_costs = block_costs
        return cost, gradient

    def equalities(self, x):
        blocks, derivatives, raw = self.decode(x)
        residuals, rows = [], []
        for i, block in enumerate(blocks):
            values, gps, gts, ghs, ges = boundary_derivative_jacobian(
                self._minco(block), block[2], block[3], 3, [0, 1])
            residuals.extend(values)
            for gp, gt, gh, ge in zip(gps, gts, ghs, ges, strict=True):
                rows.append(self._pullback(i, gp, gt, gh, ge, derivatives, raw))
        return np.array(residuals), np.array(rows)

    def extra_equalities(self, x):
        """Optional fixed orientation constraints; positions/rates are eliminated."""
        blocks, derivatives, raw = self.decode(x)
        residuals, rows = [], []
        for i, (target, block) in enumerate(zip(self.targets, blocks, strict=True)):
            if target.orientation is None:
                continue
            only_orientation = replace(target, position_mask=np.zeros(3, dtype=bool),
                                       linear_velocity=None, angular_velocity=None)
            minco = self._minco(block)
            event = TaskEventConstraint(self.robot, only_orientation, knot=len(block[3]),
                                       gripper_opening=self.gaps[i])
            residual, gps, gts, ghs, ges = evaluate_minco_constraints(
                minco, block[2], block[3], event)
            residuals.extend(residual)
            for gp, gt, gh, ge in zip(gps, gts, ghs, ges, strict=True):
                rows.append(self._pullback(i, gp, gt, gh, ge, derivatives, raw))
        return np.array(residuals), np.array(rows).reshape(len(rows), len(raw))

    def trajectories(self, x, result):
        from .trajectory import AerialManipulatorTrajectory
        blocks, _, _ = self.decode(x)
        self.objective(x)
        plans = []
        for i, (block, path) in enumerate(zip(blocks, self.knots, strict=True)):
            coefficients, _ = self._minco(block).solve(block[2], block[3])
            plans.append(AerialManipulatorTrajectory(block[3], coefficients.reshape(-1, 6, 8),
                path, self.last_block_costs[i], result.iterations, result.converged, result.message))
        return plans


_TANH_JOINT_INVERSE_MARGIN = 1.0e-6

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


def _boundary_pva(state):
    result = np.zeros((3, 8)); result[0] = state
    return result


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


class AerialManipulatorOptimization:
    """Point-to-point numerical refinement using the same shared spline mechanics."""

    def _optimize_path(self, path, knots, evaluator, robot, edge_valid,
                       metric_scale, shortcut_attempts):
        cfg = self.config

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
        # diagnostics are available if optimization fails.
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
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            self.last_metrics.update({
                "objective_calls": int(objective_calls),
                "objective_samples": int(evaluator.objective_samples),
                "optimizer_seconds": time.perf_counter()-phase,
                "optimizer_status": "failed",
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
            sample_dt, validation_performed=True)
        return trajectory

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
            cost, grad_points, grad_times, _ = evaluate_minco_objective(
                minco, points, durations, terms=(evaluator.integrated_penalty,),
                time_weight=cfg.time_weight, energy_weights=np.asarray(cfg.jerk_weights))
        except (np.linalg.LinAlgError, ValueError):
            return 1.0e30, np.zeros_like(variables)
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
