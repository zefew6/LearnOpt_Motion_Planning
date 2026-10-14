# Live whole-body reaching benchmark

Launch from the repository root:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/yopo.yaml
```

The preset uses CUDA inference and the fixed-map training run’s `best_17000`
checkpoint (`primitive_goal_v5`). It is an online test configuration; the offline
validation result does not establish successful navigation.

This task opens the MuJoCo viewer before submitting the first plan. Green shows
the selected base trajectory; pale gray shows five diverse alternative candidates,
orange the selected arm grasp-frame path (not another base candidate), purple the preceding selected path,
and blue actual base motion. Alternatives are a diagnostic subset of all 27
candidates, not collision-certified paths. The console prints the selected
candidate index, horizon, candidate positional spread and actual speed. The
viewer remains open at success/failure, and the terminal immediately prints the
outcome. Closing the viewer ends the task. Normal viewer runs write no artifacts.
A graphical desktop/OpenGL and the YAML-selected local checkpoint are required.

## Ownership and configuration

`main.py` dispatches `task: whole_body_planning` to
`runners/whole_body_planning.py`; task execution and actual-state sampling belong
in `tasks/whole_body_reaching.py`. Neural modeling, decoding and deployment live in `yopo/model`,
`yopo/trajectory` and `yopo/deployment.py`; data and training live in
`yopo/data` and `yopo/training`.

The task selects `planner: yopo`, `controller: cascaded`, and `wind: none`.
The whole-body adapter adds inverse-dynamics arm tracking to cascaded base
control. Its feedback and integral period equal the XML physics timestep;
reference sampling follows YAML `control_dt` (default 10 ms). No robot resets
occur during motion. Viewer reset reinitializes the episode.

Scene geometry and start/goal live in `simulation/models/yopo_bamboo.xml`.
XML `whole_body_start` and `whole_body_goal` are
`[x,y,z,yaw,q1,q2,q3,q4]` in NED/radians. `whole_body_gripper` fixes the commanded
opening; no manipulation action or object target is present. `planning_bounds`
sets the task workspace. The scene covers about 24 × 12 m and contains 282
bamboo cylinders, 1.09–3.30 m high, with radii 0.045–0.10 m. Poles are spread
across the map. Shorter poles can intersect the arm below the base; taller poles
require base avoidance. A winding passage retains whole-body clearance.
The base height workspace spans 0.8–3.4 m. XML owns all geometry.

`yopo.checkpoint` resolves relative to the YAML; `device` selects CPU/CUDA.
Checkpoint camera/robot contracts are checked. The physical gripper servo may
slightly deviate from its command: this YAML explicitly allows 2 mm through
`gripper_tolerance`; observations retain the actual measured gap. The planner's
default tolerance is 0.5 mm, and all tolerances are bounded by 5 mm.

`yopo.speed_scale` is a deployment adjustment, default 1.0 and bounded to
0.5–2.0. This preset uses 1.0 after the shape/jerk investigation. It divides predicted piece durations by the
factor (clipped to the decoder's 0.1–3.0 s bounds), scales predicted terminal
velocities/accelerations, and re-solves MINCO from the unchanged actual start
PVA. It increases commanded physical motion rather than viewer playback speed.
The original network ranking is retained; this is not speed-conditioned training
or a new collision/dynamics certificate. Nonzero actual start derivatives mean
the resulting path need not be a uniform time-rescaling of the original curve.

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

## Validation limits

The bamboo scene has a ten-piece whole-body collision-free witness in
`tests/integration/test_whole_body_online.py`. This checks scene feasibility;
the witness is never supplied to the neural planner. Passing it or successfully
loading a checkpoint does not establish successful online navigation.
The selected checkpoint was trained on the earlier wall/tree dataset; changing
the interactive scene does not retrain it. Observe actual base/arm tracking,
collisions and final success/failure in the live viewer.
