"""Protocol-driven OMPL RRT-Connect search."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou


@runtime_checkable
class StateSpaceAdapter(Protocol):
    """Operations needed to connect a numeric state vector to an OMPL space."""

    def create_space(self, lower, upper, metric_scale): ...

    def read_state(self, ompl_state) -> np.ndarray: ...

    def write_state(self, ompl_state, values) -> None: ...

    def set_sampling_bounds(self, lower, upper) -> None: ...


class RRTConnectPlanningError(RuntimeError):
    """Search failure with the OMPL work completed before termination."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = dict(metrics)


class _BatchMotionValidator(ob.MotionValidator):
    def __init__(self, space_information, adapter, edge_valid):
        super().__init__(space_information)
        self._adapter = adapter
        self._edge_valid = edge_valid
        self.checked = 0
        self.invalid = 0

    def checkMotion(self, first, second):
        self.checked += 1
        valid = bool(self._edge_valid(
            self._adapter.read_state(first), self._adapter.read_state(second)))
        self.invalid += int(not valid)
        return valid


def plan_rrt_connect(
        adapter: StateSpaceAdapter,
        start,
        goal,
        lower,
        upper,
        metric_scale,
        *,
        state_valid,
        edge_valid,
        range_size,
        sampling_regions=None,
        simplify_attempts=32,
        simplify_budget_s=.02,
        timeout_s=60.0,
):
    """Find a route through an adapter-owned OMPL state space."""
    if ((timeout_s is not None and
         (not np.isfinite(timeout_s) or timeout_s <= 0.0))
            or isinstance(simplify_attempts, bool) or int(simplify_attempts) < 0
            or not np.isfinite(simplify_budget_s) or simplify_budget_s < 0.0
            or not np.isfinite(range_size) or range_size <= 0.0):
        raise ValueError("RRT-Connect timeout, range, and simplification settings must be valid")
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    metric_scale = np.asarray(metric_scale, dtype=float)
    if (start.ndim != 1 or goal.shape != start.shape or lower.shape != start.shape
            or upper.shape != start.shape or metric_scale.shape != start.shape
            or not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal))
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or not np.all(np.isfinite(metric_scale)) or np.any(upper <= lower)
            or np.any(metric_scale <= 0.0)):
        raise ValueError("start, goal, bounds, and scales must be finite vectors of one dimension")

    started = time.perf_counter()
    setup_started = time.perf_counter()
    space = adapter.create_space(lower, upper, metric_scale)
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(
        lambda state: bool(state_valid(adapter.read_state(state))))
    motion_validator = _BatchMotionValidator(si, adapter, edge_valid)
    si.setMotionValidator(motion_validator)
    si.setup()

    start_state, goal_state = space.allocState(), space.allocState()
    adapter.write_state(start_state, start)
    adapter.write_state(goal_state, goal)
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

    regions = ([{"name": "global", "lower": lower, "upper": upper, "fraction": 1.0}]
               if sampling_regions is None else list(sampling_regions))
    if not regions or any(float(region["fraction"]) <= 0.0 for region in regions):
        raise ValueError("sampling_regions must contain positive-duration stages")
    total_fraction = sum(float(region["fraction"]) for region in regions)
    stage_seconds = {}
    search_started = time.perf_counter()
    if timeout_s is None:
        while not pdef.hasExactSolution():
            for region in regions:
                if pdef.hasExactSolution():
                    break
                region_lower, region_upper = _region_bounds(region, start.shape)
                adapter.set_sampling_bounds(region_lower, region_upper)
                name = str(region["name"])
                stage_started = time.perf_counter()
                try:
                    log_level = ou.getLogLevel()
                    ou.setLogLevel(ou.LOG_WARN)
                    try:
                        planner.solve(1.0)
                    finally:
                        ou.setLogLevel(log_level)
                except TimeoutError:
                    # The one-second solve is a resumable work slice, not a search deadline.
                    pass
                stage_seconds[name] = (
                    stage_seconds.get(name, 0.0)+time.perf_counter()-stage_started)
    else:
        for region in regions:
            elapsed = time.perf_counter()-search_started
            remaining = float(timeout_s)-elapsed
            if remaining <= 0.0:
                break
            region_lower, region_upper = _region_bounds(region, start.shape)
            adapter.set_sampling_bounds(region_lower, region_upper)
            stage_started = time.perf_counter()
            allotted = min(
                remaining, float(timeout_s)*float(region["fraction"])/total_fraction)
            name = str(region["name"])
            try:
                log_level = ou.getLogLevel()
                ou.setLogLevel(ou.LOG_WARN)
                try:
                    planner.solve(max(.001, allotted))
                finally:
                    ou.setLogLevel(log_level)
            except TimeoutError as error:
                stage_seconds[name] = (
                    stage_seconds.get(name, 0.0)+time.perf_counter()-stage_started)
                metrics = _metrics(
                    started, setup_seconds, search_started, stage_seconds,
                    motion_validator, planner, si, path_states=0,
                    simplification_seconds=0.0, first_solution=None)
                raise RRTConnectPlanningError(str(error), metrics) from error
            stage_seconds[name] = (
                stage_seconds.get(name, 0.0)+time.perf_counter()-stage_started)
            if pdef.hasExactSolution():
                break

    search_seconds = time.perf_counter()-search_started
    if not pdef.hasExactSolution():
        metrics = _metrics(
            started, setup_seconds, search_started, stage_seconds,
            motion_validator, planner, si, path_states=0,
            simplification_seconds=0.0, first_solution=None)
        raise RRTConnectPlanningError(
            "OMPL RRT-Connect did not find an exact goal solution", metrics)

    path = pdef.getSolutionPath()
    states = np.asarray([
        adapter.read_state(state) for state in path.getStates()], dtype=float)
    states[0], states[-1] = start, goal
    simplify_started = time.perf_counter()
    shortcut_rng = np.random.default_rng()
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
    if (len(states) < 2 or not all(edge_valid(first, second)
                                   for first, second in zip(
                                       states[:-1], states[1:], strict=True))):
        raise RuntimeError("OMPL path failed the shared fine-resolution edge validator")

    metrics = _metrics(
        started, setup_seconds, search_started, stage_seconds,
        motion_validator, planner, si, path_states=len(states),
        simplification_seconds=simplification_seconds,
        first_solution=search_seconds)
    metrics["rrt_shortcut_attempts"] = int(attempts)
    return states, metrics


def _region_bounds(region, shape):
    lower = np.asarray(region["lower"], dtype=float)
    upper = np.asarray(region["upper"], dtype=float)
    if (lower.shape != shape or upper.shape != shape
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(upper <= lower)):
        raise ValueError("sampling region bounds must match the state dimension")
    return lower, upper


def _metrics(started, setup_seconds, search_started, stage_seconds,
             motion_validator, planner, space_information, *, path_states,
             simplification_seconds, first_solution):
    data = ob.PlannerData(space_information)
    planner.getPlannerData(data)
    search_seconds = time.perf_counter()-search_started
    return {
        "rrt_setup_seconds": setup_seconds,
        "rrt_search_seconds": search_seconds,
        "rrt_simplification_seconds": simplification_seconds,
        "rrt_seconds": time.perf_counter()-started,
        "rrt_nodes": int(data.numVertices()),
        "rrt_motion_checks": int(motion_validator.checked),
        "rrt_invalid_motions": int(motion_validator.invalid),
        "rrt_valid_motions": int(motion_validator.checked-motion_validator.invalid),
        "rrt_path_states": int(path_states),
        "rrt_sampling_stage_seconds": dict(stage_seconds),
        "rrt_first_exact_solution_s": first_solution,
    }


__all__ = ["RRTConnectPlanningError", "StateSpaceAdapter", "plan_rrt_connect"]
