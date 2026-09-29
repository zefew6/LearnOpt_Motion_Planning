# Aerial Pick/Place Refactor Design

**Status: Awaiting user review**

## Goal

Reorganize `uav_ac/tasks/aerial_pick_place.py` so planning, execution, orchestration,
and result reporting have clear ownership while preserving the task's observable behavior.
Keep the implementation in this single task module and reduce its nonblank,
non-comment Python lines by at least 30%.

The baseline measured before this refactor is 529 counted lines, so the target is
370 lines or fewer. Blank lines and full-line comments are excluded; docstrings
and executable statements are included. Readability takes priority over line
compression: meet the reduction through shared logic and removal of dead or
duplicated code, not semicolon chaining or dense expressions.

## Existing Responsibilities

The current module combines:

- Mission setup and lifecycle in `run_aerial_pick_place`.
- Occupancy/ESDF construction and both MINCO planning legs in `plan_pick_place`.
- Controller-rate execution, state transitions, settling, and trace sampling in
  `_Execution`.
- Public result construction, diagnostic pass-through, and terminal summary output.

The generic transition engine already lives in `uav_ac.utils.StateMachine`; the
pick/place enum and allowed-transition graph remain task-specific.

## Proposed Internal Structure

Keep the same file and public entry points, organized into these sections:

1. **Task state and planning data** — `PickPlaceState`, its transition graph, and
   a private `_PlanBundle` dataclass for the trajectory pair, endpoints, gripper
   gaps, settings, and initial robot configuration.
2. **Mission orchestration** — `run_aerial_pick_place` owns simulation setup,
   planning/execution sequencing, error-to-result handling, and the final summary.
3. **Planning** — `plan_pick_place` keeps its signature and dictionary result for
   direct benchmark callers. Focused helpers handle map construction, one-leg
   planning, and per-leg metric collection. The mission path adapts the public
   mapping into `_PlanBundle` before constructing the executor.
4. **Execution** — `_Execution` owns reset and per-physics-step behavior. Shared
   methods handle the pick/place travel stages and the grasp/release gripper
   stages. Separate small helpers own tracking-error sampling, settling checks,
   and joint-trace output.
5. **Results and reporting** — result assembly keeps every current public key and
   value semantics. Repeated pick/place result fields are produced through a
   shared per-leg helper. The concise summary remains: state, success, collision,
   both plan-valid flags, planning time, and a failure reason when present.

Data flow:

`run_aerial_pick_place` → `plan_pick_place` → `_PlanBundle` → `_Execution` →
result mapping → concise summary.

## Line-Count Reduction

The reduction will come from:

- Using the same travel-stage logic for pick and place, parameterized by plan,
  target, gripper gap, and completion state.
- Using shared gripper-stage logic for grasp and release while keeping their
  task-specific completion checks explicit.
- Grouping repeated execution trace initialization/reset and per-leg result fields.
- Removing fields and resets with no readers, after confirming each with a
  repository search.
- Passing the execution inputs as `_PlanBundle` rather than a long positional
  argument list.

No result keys, diagnostics keys, state names, or failure reason strings will be
removed to reach the line target. If the target cannot be met without making the
code harder to understand or changing behavior, stop and report the gap instead
of compressing the code mechanically.

## Compatibility and Behavior Boundaries

- Keep `uav_ac.tasks.aerial_pick_place.run_aerial_pick_place` and
  `plan_pick_place` import paths and call signatures.
- Preserve `plan_pick_place` dictionary keys and `search_only` behavior used by
  `tests/benchmark_aerial_manipulator_minco.py`.
- Preserve diagnostics accumulation, including partial planner metrics on errors.
- Preserve task result keys, event-sequence names/order, state transitions,
  payload ownership reporting, collision handling, controller timing, and
  trajectory validation behavior.
- Keep the short terminal summary and its failure message; detailed diagnostics
  remain in the returned mapping.
- Do not alter unrelated modified files in the working tree.

## Review and Validation

Before implementation, review this document for scope and behavior boundaries.
After implementation, inspect the public API and result-key mapping, search for
stale private references, check whitespace, and recount source lines against the
370-line target. Do not add or run tests unless the user asks for verification.
