# Live whole-body reaching benchmark

Launch from the repository root:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/whole_body_online.yaml
```

This task opens the MuJoCo viewer before submitting the first plan. Green shows
its current selected local trajectory; blue shows actual robot motion. The
viewer remains open at success/failure, and the terminal immediately prints the
outcome. Closing the viewer ends the task. Normal viewer runs write no artifacts.
A graphical desktop/OpenGL and the YAML-selected local checkpoint are required.

## Ownership and configuration

`main.py` dispatches `task: whole_body_planning` to
`runners/whole_body_planning.py`; task execution and actual-state sampling belong
in `tasks/whole_body_reaching.py`. Existing neural modeling/decoding stays in
`planning/end2end`; dataset generation and training stay in `YOPO`.

The task selects `planner: end2end`, `controller: cascaded`, and `wind: none`.
The whole-body adapter adds inverse-dynamics arm tracking to cascaded base
control. Its feedback and integral period equal the XML physics timestep;
reference sampling follows YAML `control_dt` (default 10 ms). No robot resets
occur during motion. Viewer reset reinitializes the episode.

Scene geometry and start/goal live in `simulation/models/whole_body_dense_outdoor.xml`.
XML `whole_body_start` and `whole_body_goal` are
`[x,y,z,yaw,q1,q2,q3,q4]` in NED/radians. `whole_body_gripper` fixes the commanded
opening; no manipulation action or object target is present. `planning_bounds`
sets the task workspace. The outdoor scene spans about 12 m and combines finite
walls with stepped openings, varied-height trunks/crowns and low obstacles.

`end2end.checkpoint` resolves relative to the YAML; `device` selects CPU/CUDA.
Checkpoint camera/robot contracts are checked. The physical gripper servo may
slightly deviate from its command: this YAML explicitly allows 2 mm through
`gripper_tolerance`; observations retain the actual measured gap. The planner's
default tolerance is 0.5 mm, and all tolerances are bounded by 5 mm.

`whole_body` options (defaults in `runners/whole_body_config.py`):

| Setting | Meaning | Default |
| --- | --- | --- |
| replan_dt | Simulated interval between single-frame planning requests | 0.2 s |
| planning_timeout | Maximum wall time per inference | 1 s |
| max_plan_age | Maximum simulated age of an accepted snapshot | 0.5 s |
| position_tolerance / yaw_tolerance / joint_tolerance | Final configuration tolerances | 0.1 m / 0.1 rad / 0.1 rad |
| velocity_tolerance / body_rate_tolerance / joint_rate_tolerance | Actual settling speed tolerances | 0.1 m/s / 0.2 rad/s / 0.1 rad/s |
| settle_time | Continuous time within configuration and speed tolerances | 0.5 s |
| tracking_error_limit | Maximum base reference error | 2 m |
| stall_time / progress_distance | Detect lack of actual position movement | 12 s / 0.05 m |
| log_dt / threads | Console interval / PyTorch CPU threads | 1 s / 2 |

Control and replanning periods must be integer multiples of their inner periods.
`duration` limits simulated episode time. `visualize: false` runs the same physics
and controller loop without the viewer for diagnostics; it is not playback.

## Online execution

On each request the simulation supplies one current depth image, mask, actual
configuration/velocity and the fixed final goal. Acceleration and jerk use
filtered finite differences of actual measured velocity. Rendering forwards a
scratch copy to align camera pose with current qpos without altering live data.
The model ranks and decodes its local candidates without ESDF/map filtering.
No temporal training input is added by online replanning.

A single daemon worker performs inference while the controller keeps tracking
its previous reference. New trajectories are sampled at their snapshot age,
including the time spent computing them. Stale/late plans, exhausted horizons,
physical collisions, workspace violations and tracking divergence report explicit
failure. Success requires the actual full configuration and speeds to settle.
This is soft real time: Python, rendering and the desktop viewer do not guarantee
hard deadlines. A timed-out native inference cannot be forcibly killed inside a
thread; closing/resetting does not wait for it, and its result is discarded.

## Current validation and limitations

An authored eight-piece, stop-at-knot trajectory through the three openings was
validated offline with the existing whole-body evaluator: dense sampling at
25 ms passed with no collision or limit violation, minimum world distance
0.092 m and minimum self distance 0.060 m. This witness is a geometry/kinematics
feasibility check, not a controller-tracked success or an optimizer comparison,
and is never supplied to the neural planner.

The current `p4_portal_forest_seed23_v1/best_100` checkpoint has not solved this
long task. A real-physics diagnostic accepted 10 online plans over 1.87 s and
failed on an actual workspace violation, without collision. Maximum base
tracking error was 3.46 mm and joint tracking error 0.00461 rad. The final goal
starts 11.4 m away, beyond the short-range training distribution; this result
exposes a planning/generalization limitation. Further training or an explicitly
designed navigation interface is needed before claiming end-to-end arrival.
The benchmark deliberately reports that failure rather than declaring success.

The graphical entry point was also launched successfully on the desktop outside
the filesystem sandbox: live planning/tracking and the immediate failure message
were observed. The sandboxed launch cannot connect to DISPLAY `:1`.

Verification: focused online/config/depth/end2end tests passed (38). The broader
unit/integration run had 728 passed, 2 skipped and the same three pre-existing
failures: gate-racing control-period expectation and two fake-viewer cycle/reset
expectations. No unrelated controller or viewer behavior was changed to mask them.
