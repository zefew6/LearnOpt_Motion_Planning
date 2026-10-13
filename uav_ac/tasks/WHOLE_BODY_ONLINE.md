# Live whole-body reaching benchmark

Launch from the repository root:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/yopo_like_wbp.yaml
```

This task opens the MuJoCo viewer before submitting the first plan. Green shows
the selected base trajectory; pale gray shows five diverse alternative candidates,
orange the selected arm grasp-frame path, purple the preceding selected path,
and blue actual base motion. Alternatives are a diagnostic subset of all 27
candidates, not collision-certified paths. The console prints the selected
candidate index, horizon, candidate positional spread and actual speed. The
viewer remains open at success/failure, and the terminal immediately prints the
outcome. Closing the viewer ends the task. Normal viewer runs write no artifacts.
A graphical desktop/OpenGL and the YAML-selected local checkpoint are required.

## Ownership and configuration

`main.py` dispatches `task: whole_body_planning` to
`runners/whole_body_planning.py`; task execution and actual-state sampling belong
in `tasks/whole_body_reaching.py`. Existing neural modeling/decoding stays in
`planning/end2end`; dataset generation and training stay in `YOPO`.

The task selects `planner: yopo`, `controller: cascaded`, and `wind: none`.
The whole-body adapter adds inverse-dynamics arm tracking to cascaded base
control. Its feedback and integral period equal the XML physics timestep;
reference sampling follows YAML `control_dt` (default 10 ms). No robot resets
occur during motion. Viewer reset reinitializes the episode.

Scene geometry and start/goal live in `simulation/models/whole_body_dense_outdoor.xml`.
XML `whole_body_start` and `whole_body_goal` are
`[x,y,z,yaw,q1,q2,q3,q4]` in NED/radians. `whole_body_gripper` fixes the commanded
opening; no manipulation action or object target is present. `planning_bounds`
sets the task workspace. The scene covers about 24 × 12 m, with 154 distributed trunks (0.9–3.9 m
high), crowns, five industrial machine/frame remains and five varied-height
low stumps. Four ruined walls each contain five openings, including stepped
openings. The base height workspace spans 0.8–3.4 m. A winding route retains
whole-body clearance across the scene; obstacle generation does not concentrate
all dense geometry into a few clusters. XML is the source of scene geometry.

`yopo.checkpoint` resolves relative to the YAML; `device` selects CPU/CUDA.
Checkpoint camera/robot contracts are checked. The physical gripper servo may
slightly deviate from its command: this YAML explicitly allows 2 mm through
`gripper_tolerance`; observations retain the actual measured gap. The planner's
default tolerance is 0.5 mm, and all tolerances are bounded by 5 mm.

`yopo.speed_scale` is a deployment adjustment, default 1.0 and bounded to
0.5–2.0. This preset uses 1.3. It divides predicted piece durations by the
factor (clipped to the decoder's 0.1–3.0 s bounds), scales predicted terminal
velocities/accelerations, and re-solves MINCO from the unchanged actual start
PVA. It increases commanded physical motion rather than viewer playback speed.
The original network ranking is retained; this is not speed-conditioned training
or a new collision/dynamics certificate. Nonzero actual start derivatives mean
the resulting path need not be a uniform time-rescaling of the original curve.

A 5 s real-physics smoke test with factor 1.3 progressed about 2.13 m toward the
goal, versus about 1.59 m at factor 1.0. Peak actual base speed was 0.58 m/s;
25 replans were accepted, without collision during this short check. Maximum
base tracking error increased from 3.50 to 4.73 mm and joint error to 0.00931 rad.

At factor 1.0 the first 3 s always selected candidate 4. The candidates themselves
were distinct (initial intermediate lateral/vertical standard deviations about
0.28/0.21 m, with differing arm waypoints). Replacing the depth frame with all
no-return pixels changed parameters by up to 0.188 in their mixed physical
units but did not change the selected index. This is a sensitivity probe, not
proof of adequate visual obstacle avoidance. The overlay exposes the current
ranking preference without introducing a map-based fallback.

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

The enlarged scene has a ten-piece, stop-at-knot whole-body witness tested in
`tests/integration/test_whole_body_online.py`. Its NED route alternates between
wall openings and folds the arm above low obstacles. It is an offline
geometry/kinematics feasibility check, not a controller-tracked success or an
optimizer comparison, and is never supplied to the neural planner.

The existing `p4_portal_forest_seed23_v1/best_100` checkpoint was smoke-tested
at speed factor 1.0 for 5 simulated seconds on this enlarged scene: 25 plans accepted, no collision,
maximum base tracking error 3.50 mm and joint error 0.00454 rad. The diagnostic
ended at its configured 5 s limit; this does not establish full-task arrival.
The goal is initially 22.9 m away, beyond the short-range training distribution.
Normal interactive runs allow 180 s and display terminal success/failure.
A desktop run of the enlarged scene continued to 13.52 s and reported a
physical collision. With factor 1.3 and the candidate overlay, the desktop run
reported collision at 10.19 s; the current checkpoint still does not solve the
benchmark. Faster timing improves motion speed, not the learned avoidance quality.

The previous smaller scene's full unit/integration test run had 728 passed,
2 skipped and three pre-existing failures (gate-racing timing and two fake-viewer
cycle/reset expectations). Those unrelated behaviors were not changed here.

For the enlarged scene and `yopo` selector, the focused runner/end2end/depth/online
checks passed (46 tests), and the dense whole-body witness test passed separately.
The updated graphical entry point was launched on the desktop.

Speed adjustment and actual-head continuity, configuration, runner, depth and
online tracking checks passed (48 focused tests); real graphical execution also
exercised the multi-candidate and grasp-frame overlay.
