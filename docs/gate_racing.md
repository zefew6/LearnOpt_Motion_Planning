# Continuous gate racing

The gate-racing task learns ordered flight through six gates directly from
reward. It does not call GCOPTER, acados, or a trajectory-bank generator.
Both PPO-MLP and PPO-ACMPC use the same gate observations and normalized
collective-thrust/body-moment actions. This is a custom three-dimensional
track, not a reproduction of the paper's Split-S geometry or thrust/body-rate
interface. MPVE, wind and dynamics randomization are outside this version.

Task version 3 uses geometric gate MPC and diameter-relative openings. Both supplied training
YAMLs generate a fresh static six-gate course at each reset. Version 1/2
checkpoints are incompatible: keep old runs and start a new run directory.

## Train, resume and evaluate

Run from the repository root with the existing Python 3.13 environment:

```bash
.venv/bin/python -m uav_ac.rl.training --config configs/ppo_gate_racing.yaml --run-dir runs/gate_racing/mlp_random_v3
.venv/bin/python -m uav_ac.rl.training --config configs/acmpc_gate_racing.yaml --run-dir runs/gate_racing/acmpc_random_v3
.venv/bin/python -m uav_ac.rl.training --config configs/acmpc_gate_racing.yaml --run-dir runs/gate_racing/acmpc_random_v3 --resume runs/gate_racing/acmpc_random_v3/final_model.zip --total-timesteps 1000000
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc_random_v3 --mode metrics --episodes 20
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc_random_v3 --mode metrics --episodes 1 --seed 17
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc_random_v3 --mode interactive --seed 17
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc_random_v3 --mode record --seed 17 --output runs/gate_racing/acmpc_random_v3/replay.mp4
```

For headless recording, use `MUJOCO_GL=egl` or an available offscreen MuJoCo
backend. Recording requires FFmpeg. Viewer and video use exactly the evaluation
environment and deterministic policy actions. The viewer closes at episode end;
closing its window stops replay. Existing video files are not overwritten.
Both replay modes default to a camera 6 m behind/above the drone, looking down
35 degrees at a point 0.3 m above its center. Flight YAML `follow_camera: false`
selects the fixed overview; `true` follows each frame. The flight entry point
loads the exact `rl.checkpoint` file, including final or periodic checkpoints.

Defaults are 8,192 training transitions, four environments, seed 42, CPU, one
Torch thread, 128 rollout steps per environment, minibatches of 64 and three
PPO epochs. These are smoke-training settings, not a convergence budget.
`--total-timesteps` on resume means additional transitions, rounded up by PPO to
a rollout boundary. Gate racing rejects `--prepare-only`; select its XML with
`scene` in training YAML rather than `--model-path` or `flight.yaml`.

`best_model.zip`, `final_model.zip`, periodic checkpoints, Monitor CSVs and
TensorBoard logs live under the run directory. `rl_config.json` records the
task, observation scales, action definition, vehicle parameters, scene SHA-256,
control period and MPC settings. Keep it with the checkpoints. Loading or
resuming after changing the scene or these contracts fails explicitly.
The contract also records the task version, course generator version and all
course parameters. Evaluation episodes include actual gate centers and rotation
matrices in NED and aperture half-sizes; `missed_gate_rate` is reported in metrics
and TensorBoard. Initial-state perturbation uses a separate random stream, so
disabling it does not change the course for a given reset seed.
Old trajectory-tracking configurations and checkpoint classes are preserved.

## Task and scene conventions

The XML owns vehicle physics, bounds and ordered `gate_00` through `gate_05`
box sites. Each site's local +X is the crossing direction and its Y/Z sizes
are aperture half-widths. The four visible frame boxes match that aperture.
Changing a gate means moving its containing body, keeping frame and site
together. Simulation geometry uses the repository's MuJoCo convention;
task/dynamics state uses NED/FRD after conversion at the scene boundary.

The fixed track has 2 m square openings, approximately 8–12 m gate spacing,
turns and heights from 2–5 m. Set `course.mode: fixed` to use it; omitted `course`
also selects fixed geometry for programmatic callers. The supplied YAMLs select
`random`. Optional course settings (distances in meters, angles in degrees):

```yaml
course:
  mode: random
  spacing: [5.0, 8.0]           # 3D distance between successive centers
  height: [2.0, 6.0]           # height above z=0, positive upward
  height_step: 1.0
  turn_degrees: 60.0           # change in horizontal path heading
  width_ratio: [1.3, 4.0]      # full aperture width / vehicle diameter
  height_ratio: [1.3, 4.0]     # full aperture height / vehicle diameter
  yaw_degrees: 15.0            # offset from incoming path heading
  tilt_degrees: 15.0           # independently sampled pitch and roll
```

The first gate is 5–8 m from the XML start, directed toward the field interior;
successive headings, heights, dimensions and orientations are sampled anew.
The diameter is twice the maximum collision-geometry bounding radius measured
from the vehicle origin, including rotor disks (about 0.45 m in this XML).
Widths and heights are sampled independently: approximately 0.59–1.80 m.
The generator checks full frame bounds with vehicle-radius + 0.02 m margin,
nonoverlap using conservative enclosing spheres, approach sides, and ordered
entry/center/exit connections against every expanded frame box. A short segment beyond the finish is
also checked. These are conservative geometry checks, not a dynamics feasibility
certificate. Up to 100 complete candidate courses are tried; exhaustion reports
an error without falling back to a fixed track. MuJoCo recompiles the pristine
template on every random reset to update collision bounds and visible geometry
together; this adds reset cost but does not change physics stepping. Renderers
and viewers must attach after reset, as the existing replay entry point does.

Episodes start in hover at the XML start, with
position perturbations ±0.2 m, velocity perturbations ±0.1 m/s and small
quaternion perturbations. All gates must be passed once in order within 30 s.
There is no tracking-speed or tilt cutoff. The physical step is 1 ms and
actions are updated every 10 ms.

Each physical substep tests the motion segment against the current gate plane.
Only a forward intersection strictly inside the opening counts. Later gates
cannot be collected out of order. Collision or leaving scene bounds terminates
immediately, with collision taking priority over a same-substep crossing. The
first forward crossing outside the current aperture (including its edge)
terminates as `missed_gate`; returning to retry cannot recover the episode.
Starting a substep or activating a new target on its exit side also fails.
Moving backward while still on the approach side is allowed. A reverse crossing
never earns gate credit. The
new vehicle scene includes arm and rotor collision geometry. As with MuJoCo's
discrete collision detection, this does not guarantee collision detection at
arbitrarily large speeds.

Reward sums distance progress toward the current gate, +10 per passed gate,
an additional +10 on completion, and −10 on collision, leaving bounds or missing a gate.
Progress is split at each crossing before changing the target, avoiding a
spurious jump between gate centers. The body-rate penalty is
`−0.01 * ||omega|| * (substep_dt / 0.01)`; its accumulation is independent of
the number of substeps per control action. Timeouts are Gymnasium truncations,
not terminal collisions. Completion uses the vehicle center crossing the final
gate plane; no post-finish settling segment is required.

## Observation and learned cost

Both actors see 40 features: canonical scalar-first quaternion (4), body-frame
velocity (3), body rates (3), previous action (4), next two gates' body-frame
corner offsets (24), and two validity flags (2). Velocity, rates and corner
offsets are divided by 10; other scales are 1. Features are clipped to ±10.
Missing gates are zero-padded and masked. ACMPC additionally receives the
unclipped float64 13-state vector and previous action. These physical fields
retain float64 in the PPO rollout buffer.

The racing network emits 17 positive weight multipliers per stage, bounded to
0.1–10 times fixed base weights. It does not emit arbitrary linear coefficients.
The geometric task reconstructs the next two gates from corners and builds an
ordered entry/center/exit route, with an exit one vehicle diameter beyond each
gate. Previewing the next gate never changes the environment's active gate or
awards credit. Position penalties use the current gate axes and available
clearance; velocity targets follow the route. Linear terms come from these
targets. Attitude, angular-rate and control penalties stabilize the vehicle.
With no active gates the target is local hover. The terminal
dummy input is fixed at zero, with positive numerical curvature to keep the
box-QP factorization nonsingular.

Default prediction is 50 steps at 0.02 s (1 s horizon), while actions are still
updated every 0.01 s. Prediction dt must be an integer multiple of control dt;
trajectory tracking retains its original timing contract. Each solve gets a
stateless stabilizing nominal rollout instead of holding the previous torque
over the entire horizon. This rollout initializes the optimizer; the returned
action and training gradient come from the differentiable MPC layer.

```yaml
racing:
  cruise_speed: 7.0           # configurable up to 8 m/s
  minimum_speed: 2.0         # target floor, not minimum actual speed
  lateral_acceleration: 5.0  # turn-speed scheduling budget, m/s²
  braking_acceleration: 3.0  # target acceleration/deceleration budget, m/s²
  clearance_margin: 0.02
```

The aperture speed target rises from 2 m/s at 1.3 diameters to cruise speed at
4 diameters. Turn geometry and alignment can lower it, and a braking envelope
reduces speed before the gate. These are soft targets, not hard dynamic or
collision guarantees. Fixed base weights must first pass the real MuJoCo
wide/narrow-gate integration tests before training the weight network.

Racing and tracking share the existing dynamics and quadratic solver. The
default uses one iLQR update and its approximate differentiable backward,
not an exact nonlinear sensitivity. Input boxes do not represent joint rotor
feasibility; the MuJoCo plant still applies allocation and motor lag. Gate
alignment is represented in the task cost; collision avoidance is not enforced
as a hard MPC state constraint. For convergence diagnostics, copy the training
YAML and set `mpc.iterations: 5`, `mpc.retry_iterations: 10`; these settings form
part of the checkpoint contract and require a separate run. A one-iteration
finite result is not a convergence certificate.

## Metrics and interpretation

Training evaluates fixed initial-state seeds 10000–10019. Standalone evaluation
defaults to held-out seeds 0–19. Best-model selection prioritizes completion
rate, then successful completion time; before any completions it uses mean
gates passed. `evaluation_metrics.json` reports completion/collision/out-of-bounds/timeout
rates, gate counts, speed, completion time, and solver fallback counts.
Each episode also reports interpolated gate-crossing speeds, vehicle diameter
and mean speed on aligned approach segments (at least 2 m before the gate,
lateral distance less than one diameter). Segments with no samples report
`null`. `max_solver_residual` is reported separately from failures.

Use `--episodes 1` for single-environment action latency P50/P95/P99 and the
fraction above 10 ms. This includes observation batching, policy inference,
MPC and returned action, but excludes simulation stepping and rendering; the
cold first call is included. Multi-episode metrics retain amortized batched
timing and leave single-environment latency `null`. A simulated 7 m/s flight
does not demonstrate 100 Hz wall-clock execution. The 8,192-step ACMPC preset
is a smoke budget, not evidence of convergence or random-course success.

Evaluation batches active episodes. Inference time is amortized policy-call
wall time per action (CUDA synchronized), excludes physics, and is not a
single-vehicle real-time latency claim. The recorded maximum batch size makes
this interpretation explicit. Training fails on invalid solver outputs and
saves an interrupted checkpoint and diagnostic JSON; evaluation can hold the
previous action, counting every fallback.

Passing the smoke checks establishes runnable training, finite gradients and
save/load/replay compatibility. It does not establish high-speed racing skill,
sample-efficiency superiority or paper-level performance.

## Version 3 base-controller validation (2026-09-17)

With zero network outputs, 50 x 0.02 s prediction, 0.01 s control updates and
the real MuJoCo vehicle starting at rest 10 m before an aligned gate:

| Opening / diameter | Crossing speed | Peak speed | Time to cross |
| --- | ---: | ---: | ---: |
| 4.0 | 7.018 m/s | 7.018 m/s | 2.656 s |
| 1.3 | 2.069 m/s | 5.065 m/s | 3.323 s |

Both cases passed without collision. On this shared i5-12400F CPU, one Torch
thread, wide-gate action latency P50/P95/P99 was approximately 130/143/156 ms;
narrow-gate latency was 143/177/198 ms. All measured calls exceeded 10 ms.
These tests prove simulated closed-loop behavior, not 100 Hz real-time execution
or performance on arbitrary random courses. Run the same acceptance check with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q -s tests/integration/test_gate_racing_control.py
```

The follow camera was also rendered with the EGL backend and its actual camera
position checked behind/above the drone for multiple yaw angles.

## Historical version 1 smoke validation (2026-09-12)

Both default configurations completed 8,192 training transitions on CPU.
Their saved best policies were evaluated on seeds 0–19 and recorded through
the headless EGL renderer. The results establish execution, not learned racing:

| Policy | Completion | Mean gates | Collision | Out of bounds | Solver failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO-MLP | 0/20 | 0 | 20/20 | 0/20 | N/A |
| PPO-ACMPC | 0/20 | 0 | 0/20 | 20/20 | 0 |

Local artifacts are in `runs/gate_racing/mlp_validation_01` and
`runs/gate_racing/acmpc_validation_01`; each contains checkpoints,
`evaluation_metrics.json` and `replay.mp4`. Artifacts are ignored by Git.
Focused CPU tests also exercised PPO parameter updates, checkpoint reload and
resume for both actors, finite MPC gradients, batched evaluation equivalence,
gate geometry, collision priorities, substep termination and legacy tracking.
The desktop viewer was not opened; rendering was verified offscreen.
