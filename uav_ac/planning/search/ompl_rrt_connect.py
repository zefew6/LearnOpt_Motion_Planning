"""OMPL-backed RRT-Connect adapter for the aerial manipulator's 8-D space."""

from __future__ import annotations

import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou


_OMPL_SEEDED = False


def seed_ompl_once(seed):
    """Set OMPL's process-wide seed before its first random draw."""
    global _OMPL_SEEDED
    if _OMPL_SEEDED:
        return False
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("OMPL seed must be a non-negative integer")
    ou.RNG.setSeed(int(seed))
    _OMPL_SEEDED = True
    return True


class RRTConnectPlanningError(RuntimeError):
    """Search failure with the OMPL work completed before termination."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = dict(metrics)


def _read_state(state, joint_scales):
    return np.r_[np.asarray([state[0][i] for i in range(3)], dtype=float),
                 float(state[1].value),
                 np.asarray([state[2][i] for i in range(4)], dtype=float)*joint_scales]


def _write_state(state, values, joint_scales):
    values = np.asarray(values, dtype=float)
    for index in range(3):
        state[0][index] = float(values[index])
    state[1].value = float(values[3])
    for index in range(4):
        state[2][index] = float(values[4+index]/joint_scales[index])


class _AerialStateSpace(ob.CompoundStateSpace):
    """R3 x SO2 x R4 compound space with the configured weighted metric."""

    def __init__(self, lower, upper, metric_scale):
        super().__init__()
        self._joint_scales = np.asarray(metric_scale[4:8], dtype=float)
        position = ob.RealVectorStateSpace(3)
        bounds = ob.RealVectorBounds(3)
        for axis in range(3):
            bounds.setLow(axis, float(lower[axis]))
            bounds.setHigh(axis, float(upper[axis]))
        position.setBounds(bounds)
        self._position_space = position
        yaw = ob.SO2StateSpace()
        joints = ob.RealVectorStateSpace(4)
        joint_bounds = ob.RealVectorBounds(4)
        for joint in range(4):
            joint_bounds.setLow(joint, float(lower[4+joint]/self._joint_scales[joint]))
            joint_bounds.setHigh(joint, float(upper[4+joint]/self._joint_scales[joint]))
        joints.setBounds(joint_bounds)
        self._joint_space = joints
        self.addSubspace(position, float(1./metric_scale[0]))
        self.addSubspace(yaw, float(1./metric_scale[3]))
        self.addSubspace(joints, 1.0)

    def set_position_bounds(self, lower, upper):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (lower.shape != (3,) or upper.shape != (3,)
                or np.any(lower >= upper) or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))):
            raise ValueError("position sampling bounds must be ordered finite 3-vectors")
        bounds = ob.RealVectorBounds(3)
        for axis in range(3):
            bounds.setLow(axis, float(lower[axis]))
            bounds.setHigh(axis, float(upper[axis]))
        self._position_space.setBounds(bounds)

    def set_joint_bounds(self, lower, upper):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (lower.shape != (4,) or upper.shape != (4,)
                or np.any(lower >= upper) or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))):
            raise ValueError("joint sampling bounds must be ordered finite 4-vectors")
        bounds = ob.RealVectorBounds(4)
        for joint in range(4):
            bounds.setLow(joint, float(lower[joint]/self._joint_scales[joint]))
            bounds.setHigh(joint, float(upper[joint]/self._joint_scales[joint]))
        self._joint_space.setBounds(bounds)


class _BatchMotionValidator(ob.MotionValidator):
    def __init__(self, space_information, validate_edge, joint_scales):
        super().__init__(space_information)
        self._validate_edge = validate_edge
        self._joint_scales = joint_scales
        self.checked = 0
        self.invalid = 0

    def checkMotion(self, first, second):
        self.checked += 1
        valid = bool(self._validate_edge(
            _read_state(first, self._joint_scales),
            _read_state(second, self._joint_scales)))
        self.invalid += int(not valid)
        return valid


def plan_rrt_connect(start, goal, lower, upper, metric_scale, *,
                     state_valid, edge_valid, seed, range_size,
                     sampling_regions=None,
                     simplify_attempts=32, simplify_budget_s=.02,
                     timeout_s=60.0):
    """Find an exact 8-D route and apply a bounded number of valid shortcuts."""
    if (isinstance(seed, bool) or int(seed) < 0
            or not np.isfinite(timeout_s) or timeout_s <= 0.0
            or isinstance(simplify_attempts, bool) or int(simplify_attempts) < 0
            or not np.isfinite(simplify_budget_s) or simplify_budget_s < 0.0):
        raise ValueError("seed and timeout_s must be valid")
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    metric_scale = np.asarray(metric_scale, float)
    if (start.shape != (8,) or goal.shape != (8,) or lower.shape != (8,)
            or upper.shape != (8,) or metric_scale.shape != (8,)
            or np.any(upper <= lower) or np.any(metric_scale <= 0.0)):
        raise ValueError("aerial RRT start, goal, bounds, and scales must be 8-D")

    started = time.perf_counter()
    ompl_seeded_here = seed_ompl_once(seed)
    setup_started = time.perf_counter()
    space = _AerialStateSpace(lower, upper, metric_scale)
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(lambda state: bool(state_valid(
        _read_state(state, space._joint_scales))))
    motion_validator = _BatchMotionValidator(si, edge_valid, space._joint_scales)
    si.setMotionValidator(motion_validator)
    si.setup()
    start_state, goal_state = space.allocState(), space.allocState()
    _write_state(start_state, start, space._joint_scales)
    _write_state(goal_state, goal, space._joint_scales)
    pdef = ob.ProblemDefinition(si)
    pdef.addStartState(start_state)
    exact_goal = ob.GoalState(si)
    exact_goal.setState(goal_state)
    exact_goal.setThreshold(0.0)
    pdef.setGoal(exact_goal)
    planner = og.RRTConnect(si)
    planner.setRange(float(range_size))
    planner.setProblemDefinition(pdef)
    planner.setup()
    setup_seconds = time.perf_counter()-setup_started

    regions = ([{"name": "global", "lower": lower[:3], "upper": upper[:3],
                 "fraction": 1.0}]
               if sampling_regions is None else list(sampling_regions))
    if not regions or any(float(region["fraction"]) <= 0.
                          for region in regions):
        raise ValueError("sampling_regions must contain positive-duration stages")
    total_fraction = sum(float(region["fraction"]) for region in regions)
    search_started = time.perf_counter()
    stage_seconds = {}
    status = ob.PlannerStatus.TIMEOUT
    for region in regions:
        elapsed = time.perf_counter()-search_started
        remaining = float(timeout_s)-elapsed
        if remaining <= 0.:
            break
        space.set_position_bounds(region["lower"], region["upper"])
        space.set_joint_bounds(region.get("joint_lower", lower[4:8]),
                               region.get("joint_upper", upper[4:8]))
        stage_started = time.perf_counter()
        allotted = min(remaining, float(timeout_s)*float(region["fraction"])/total_fraction)
        name = str(region["name"])
        try:
            status = planner.solve(max(.001, allotted))
        except TimeoutError as error:
            stage_seconds[name] = (stage_seconds.get(name, 0.)
                                   +time.perf_counter()-stage_started)
            search_seconds = time.perf_counter()-search_started
            data = ob.PlannerData(si)
            planner.getPlannerData(data)
            metrics = {
                "rrt_setup_seconds": setup_seconds,
                "rrt_search_seconds": search_seconds,
                "rrt_simplification_seconds": 0.0,
                "rrt_seconds": time.perf_counter()-started,
                "rrt_nodes": int(data.numVertices()),
                "rrt_motion_checks": int(motion_validator.checked),
                "rrt_invalid_motions": int(motion_validator.invalid),
                "rrt_valid_motions": int(motion_validator.checked-motion_validator.invalid),
                "rrt_path_states": 0,
                "rrt_sampling_stage_seconds": stage_seconds,
                "rrt_ompl_seed_initialized": bool(ompl_seeded_here),
                "rrt_first_exact_solution_s": None,
            }
            raise RRTConnectPlanningError(str(error), metrics) from error
        stage_seconds[name] = stage_seconds.get(name, 0.)+time.perf_counter()-stage_started
        if pdef.hasExactSolution():
            break
    search_seconds = time.perf_counter()-search_started
    if not status or not pdef.hasExactSolution():
        data = ob.PlannerData(si)
        planner.getPlannerData(data)
        metrics = {
            "rrt_setup_seconds": setup_seconds,
            "rrt_search_seconds": search_seconds,
            "rrt_simplification_seconds": 0.0,
            "rrt_seconds": time.perf_counter()-started,
            "rrt_nodes": int(data.numVertices()),
            "rrt_motion_checks": int(motion_validator.checked),
            "rrt_invalid_motions": int(motion_validator.invalid),
            "rrt_valid_motions": int(motion_validator.checked-motion_validator.invalid),
            "rrt_path_states": 0,
            "rrt_sampling_stage_seconds": stage_seconds,
            "rrt_first_exact_solution_s": None,
        }
        raise RRTConnectPlanningError(
            "OMPL RRT-Connect did not find an exact goal solution", metrics)
    path = pdef.getSolutionPath()
    states = np.asarray([_read_state(state, space._joint_scales)
                         for state in path.getStates()])
    states[0], states[-1] = start, goal
    simplify_started = time.perf_counter()
    shortcut_rng = np.random.default_rng(int(seed) ^ 0x5EED5EED)
    attempts = 0
    while (len(states) > 2 and attempts < int(simplify_attempts)
           and time.perf_counter()-simplify_started < simplify_budget_s):
        attempts += 1
        first, last = sorted(shortcut_rng.choice(len(states), size=2, replace=False))
        if last <= first+1:
            continue
        if edge_valid(states[first], states[last]):
            states = np.vstack((states[:first+1], states[last:]))
    simplification_seconds = time.perf_counter()-simplify_started
    if (len(states) < 2 or not all(edge_valid(a, b) for a, b in zip(
            states[:-1], states[1:], strict=True))):
        raise RuntimeError("OMPL path failed the shared fine-resolution edge validator")
    data = ob.PlannerData(si)
    planner.getPlannerData(data)
    return states, {
        "rrt_setup_seconds": setup_seconds,
        "rrt_search_seconds": search_seconds,
        "rrt_simplification_seconds": simplification_seconds,
        "rrt_seconds": time.perf_counter()-started,
        "rrt_nodes": int(data.numVertices()),
        "rrt_motion_checks": int(motion_validator.checked),
        "rrt_invalid_motions": int(motion_validator.invalid),
        "rrt_valid_motions": int(motion_validator.checked-motion_validator.invalid),
        "rrt_path_states": int(len(states)),
        "rrt_first_exact_solution_s": float(search_seconds),
        "rrt_ompl_seed_initialized": bool(ompl_seeded_here),
        "rrt_shortcut_attempts": int(attempts),
        "rrt_sampling_stage_seconds": stage_seconds,
    }


__all__ = ["RRTConnectPlanningError", "plan_rrt_connect", "seed_ompl_once"]
