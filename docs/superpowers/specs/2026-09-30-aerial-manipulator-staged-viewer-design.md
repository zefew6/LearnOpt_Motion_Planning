# Aerial Manipulator Staged Trajectory Viewer

## Goal

For the aerial pick/place task, show one MuJoCo viewer before planning begins.
As each leg is planned, display its discrete RRT path as soon as search returns,
then add the continuous MINCO path after optimization completes. Keep the same
viewer open for flight execution.

The existing `visualize` option remains the switch: when false, planning and
execution stay headless. No second viewer is opened and no new rendering system
is introduced.

## Design

- Reuse `MujocoSimulation.set_planning_paths` for both overlays: dashed blue for
  the discrete RRT path and solid green for the MINCO path. Reserve two path
  slots when creating the pick/place simulation.
- Extend the existing interactive-viewer flow with an optional preparation
  callback that runs after the viewer is visible and before flight control
  starts. The callback can request a viewer sync after a planning overlay is
  updated. Existing callers that omit the callback keep their current behavior.
- Add an optional planner callback at the point where RRT returns its path,
  before MINCO optimization. The pick/place task uses it to update the dashed
  path immediately. After each leg's MINCO result returns, sample its position
  curve with `evaluate()` and update the solid path.
- Accumulate pick and place paths separately for RRT and MINCO, joining the
  adjacent leg endpoints once, so the final overlays show each complete route.
- After both legs are planned, construct the existing controller/execution
  objects and continue the normal flight loop in the already-open viewer.
- Preserve the task's current failure result behavior. If planning fails, stop
  preparation, close the viewer cleanly, and report the existing failed result.

## Scope and compatibility

Changes are limited to the simulation viewer lifecycle, the aerial manipulator
MINCO planner's optional RRT callback, and the aerial pick/place task. The new
callbacks are optional, so other planner and viewer callers remain compatible.
Headless runs do not create or synchronize a viewer.

## Verification

No automated tests are added or run for this change. Review the call sequence to
confirm the viewer is opened once before planning, each planning-stage update is
synchronized, and flight execution uses that same viewer.
