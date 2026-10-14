# Reinforcement-learning workflows

This directory contains task-owned RL workflows. The unified entry points are
`uav_ac.rl.training` for training and `uav_ac.rl.evaluate` for evaluation.
The task is selected from the training YAML or from `rl_config.json` in a run
directory.

## Train gate-racing MLP

Gate racing is reference-free: it does not generate a trajectory and does not
call GCOPTER. The policy observes the vehicle state and the next gates, then
outputs normalized collective thrust and body moments.

Start with the checked-in smoke configuration:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/rl/ppo_gate_racing.yaml \
  --run-dir runs/gate_racing/mlp_smoke
```

For a real training run, copy `configs/rl/ppo_gate_racing.yaml` to a separate
large-run YAML and increase `total_timesteps`, for example:

```yaml
policy_type: mlp
device: cuda
n_envs: 16
total_timesteps: 10000000
evaluation_interval: 100000
checkpoint_interval: 500000
ppo:
  n_steps: 1024
  batch_size: 1024
  n_epochs: 10
```

Then launch it with a new run directory:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/rl/ppo_gate_racing_large.yaml \
  --run-dir runs/gate_racing/mlp_large_seed42
```

Use separate run directories for different random seeds:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/rl/ppo_gate_racing_large.yaml \
  --run-dir runs/gate_racing/mlp_large_seed43 \
  --seed 43
```

The run directory contains `best_model.zip`, `final_model.zip`, periodic
checkpoints, Monitor CSV files, TensorBoard logs, and `rl_config.json`.
Keep `rl_config.json` beside the checkpoint; it is part of the compatibility
contract.

## Resume training

When a run directory already contains artifacts, pass `--resume`. The value of
`--total-timesteps` is the number of additional transitions:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/rl/ppo_gate_racing_large.yaml \
  --run-dir runs/gate_racing/mlp_large_seed42 \
  --resume runs/gate_racing/mlp_large_seed42/final_model.zip \
  --total-timesteps 9000000
```

The configuration, scene, PPO settings, and checkpoint contract must match.
Use a new run directory if you intend to change those settings.

## Monitor and evaluate

```bash
.venv/bin/tensorboard \
  --logdir runs/gate_racing/mlp_large_seed42/tensorboard

.venv/bin/python -m uav_ac.rl.evaluate \
  runs/gate_racing/mlp_large_seed42 \
  --mode metrics --episodes 100 --nominal
```

For an interactive replay:

```bash
.venv/bin/python -m uav_ac.rl.evaluate \
  runs/gate_racing/mlp_large_seed42 \
  --mode interactive
```

For headless video recording, set an offscreen MuJoCo backend:

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.rl.evaluate \
  runs/gate_racing/mlp_large_seed42 \
  --mode record \
  --output runs/gate_racing/mlp_large_seed42/replay.mp4
```

The unified evaluator reads the task from `rl_config.json`; no task-specific
evaluation module needs to be selected manually.

## Deploy through `main.py`

Edit `configs/flight/flight_gate_racing.yaml` so that `rl.checkpoint` points to the
desired `best_model.zip`, then run:

```bash
.venv/bin/python -m uav_ac.main \
  --config configs/flight/flight_gate_racing.yaml
```

The gate-racing deployment uses `planner: none`. It opens the native MuJoCo
viewer and drives the trained policy directly; no trajectory bank or planner
is involved.

The normal trajectory-tracking deployment remains separate:

```yaml
scene: open_field
planner: gcopter
controller: rl
rl:
  checkpoint: ../../runs/ppo_trajectory/multitraj01/best_model.zip
  device: cuda
```

## ACMPC policy

After the MLP baseline is stable, train the gate-racing ACMPC policy with:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/rl/acmpc_gate_racing.yaml \
  --run-dir runs/gate_racing/acmpc_large_seed42
```

MLP and ACMPC use the same gate task and action interface, but ACMPC has an
additional differentiable solver and is substantially slower. Establish the
MLP reward and gate-completion behavior first.

## Common checks

Run commands from the repository root and use the project Python environment.
Gate racing rejects `--prepare-only`, because it has no trajectory bank to
prepare. A missing or incompatible `rl_config.json`, scene contract, or model
checkpoint should be treated as a run-directory error rather than repaired by
copying files between experiments.

The current gate-racing trainer uses a batched in-process environment. Increasing
`n_envs` improves rollout and policy batch sizes, but does not yet create one
OS process per MuJoCo environment. If environment stepping dominates runtime,
profile that path before increasing the GPU model size.

The implementation layout is:

```text
uav_ac/rl/
├── training.py             unified training entry point
├── evaluate.py             unified evaluation and replay entry point
├── common/                 registry and shared RL primitives
├── tasks/
│   ├── trajectory_tracking/
│   └── gate_racing/
└── acmpc/                  stable ACMPC policy and solver paths
```

<a id="gate-racing-details"></a>

## Gate racing task and metrics

The gate-racing task learns ordered flight through six gates directly from
reward. It does not call GCOPTER, acados, or a trajectory-bank generator.
Both PPO-MLP and PPO-ACMPC use the same gate observations and normalized
collective-thrust/body-moment actions. This is a custom three-dimensional
track, not a reproduction of the paper's Split-S geometry or thrust/body-rate
interface. MPVE, wind and dynamics randomization are outside this version.

Task version 3 uses geometric gate MPC and diameter-relative openings. Both supplied training
YAMLs generate a fresh static six-gate course at each reset. Version 1/2
checkpoints are incompatible: keep old runs and start a new run directory.

### Train, resume and evaluate

Run from the repository root with the existing Python 3.13 environment:

```bash
.venv/bin/python -m uav_ac.rl.training --config configs/rl/ppo_gate_racing.yaml --run-dir runs/gate_racing/mlp_random_v3
.venv/bin/python -m uav_ac.rl.training --config configs/rl/acmpc_gate_racing.yaml --run-dir runs/gate_racing/acmpc_random_v3
.venv/bin/python -m uav_ac.rl.training --config configs/rl/acmpc_gate_racing.yaml --run-dir runs/gate_racing/acmpc_random_v3 --resume runs/gate_racing/acmpc_random_v3/final_model.zip --total-timesteps 1000000
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

### Task and scene conventions

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

### Observation and learned cost

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

### Metrics and interpretation

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

<a id="acmpc-details"></a>

## ACMPC trajectory tracking and solver contracts

AC-MPC uses a neural cost map followed by a differentiable box-constrained iLQR
solver as the mean of a Gaussian PPO policy. The critic is a separate MLP.
Stock SB3 2.8.0, MuJoCo rewards and the trajectory bank are reused. MuJoCo is
not differentiated. MPVE and learned dynamics are not implemented.

### Run

The default configuration reuses the bank at
`runs/ppo_trajectory/multitraj01/trajectory_bank`. Set `trajectory_bank_path` to
another compatible bank, or null to generate one (generation requires acados).
External assets are read-only and fingerprint checked. Their absolute location
is recorded in run metadata; update it when moving the run.

```bash
uv run python -m uav_ac.rl.acmpc.benchmark \
  --run-dir runs/acmpc_trajectory/exp01 --mode preflight
uv run python -m uav_ac.rl.training \
  --config configs/rl/acmpc_trajectory.yaml --run-dir runs/acmpc_trajectory/exp01
uv run python -m uav_ac.rl.mlp_baseline.evaluate runs/acmpc_trajectory/exp01 --mode metrics
uv run python -m uav_ac.rl.acmpc.benchmark \
  --run-dir runs/acmpc_trajectory/exp01 --mode compare
```

The existing `evaluate` interactive/record modes also work. To deploy either
policy through `uav_ac.main`, set `controller: rl` and an explicit `checkpoint`
in `configs/flight/flight.yaml`. Resume training using `--resume CHECKPOINT.zip`. Old
MLP runs retain schema 1; AC-MPC uses schema 2 and validates observation layout,
physical parameters and MPC settings.

The short run uses 8,192 steps, 4 environments, seed 42, 128 rollout steps per
environment, minibatches of 64, three PPO epochs and MPC chunks of 32. A full
run can use `--total-timesteps 10000000`; it is not started automatically.
CPU is the default because sequential small iLQR matrix operations can make
CUDA slower. Measure on the target machine before changing the device.

### Interfaces and assumptions

- State: `[p_NED, q_WB(wxyz), v_NED, omega_FRD]`, with actual `Quad` parameters.
  RK4 uses the 0.01 s controller period and renormalizes quaternions.
- Observation: `features (59,)`, `mpc_state (13,)`, `reference (21,10)` for the
  default 20-control horizon. Physical fields remain float64 in SB3's Dict
  buffer; the MLP sees the original clipped features. End references repeat.
- External actions retain the existing normalized thrust/moment definition,
  with hover at zero. Internally thrust is affine in physical units to remove
  the hover derivative discontinuity, then converted back differentiably.
- Prediction omits wind, motor lag and allocator moment reduction. Input box
  bounds are not joint rotor constraints. The plant retains its allocator and
  motor lag. No state-constraint or safety guarantee is claimed.
- The quadratic tracking cost learns diagonal multipliers in `[0.1,10]` and
  linear corrections in `[-0.5 D0,0.5 D0]`. Zero-initialized outputs recover fixed
  tracking MPC. State errors use fixed scales and a local position origin;
  quaternion references align signs. Terminal state weights are multiplied by
  four. The solver's terminal dummy control is zero and never applied.
- Deterministic initial controls come from the observation's previous action.
  There is no hidden warm-start cache. The default actor performs one iLQR
  update, matching the paper; a fixed-point residual is logged but is not a
  failure criterion in this mode. Multi-iteration diagnostic configurations
  retry from the same initialization and use the `1e-4` convergence tolerance.
  Converged samples freeze independently of their batch.
- The backward is an iLQR fixed-point approximation, not an exact nonlinear
  sensitivity including dynamics Hessians. Linear-quadratic finite differences
  and nonlinear direction checks document the distinction.
- Training failures stop with `solver_failure.json` and an interrupted
  checkpoint. Deployment may hold the previous action and counts failures.
  SB3 executes clipped Gaussian actions but retains original samples and log
  probabilities for PPO updates.

### Verification and interpretation

`preflight.json` checks fixed-cost hover and a smooth 20 cm translation.
`comparison.json` pairs five test trajectories, each with no wind and fixed
wind seeds, for learned AC-MPC, frozen-cost MPC, the existing PPO checkpoint
and native acados MPC. Native acados applies its physical wrench; reward action
penalties use bounded normalization, so tracking and success are the primary
comparison metrics. Allocator scaling and solver failures are reported.

The existing PPO has a different training budget and information interface;
these comparisons do not establish sample-efficiency advantages or isolate
architecture alone. Frozen-cost MPC isolates cost learning within this system.

Panel timings are batched policy-call latency, not single-sample latency;
preflight uses batch size one. CUDA timing requires synchronization. A 100 Hz
simulation control period does not establish 100 Hz wall-clock execution.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/unit/rl/test_acmpc.py tests/unit/control/test_rl_controller.py \
  tests/integration/test_rl_trajectory_tracking.py -o addopts=
```

References: [paper](https://arxiv.org/html/2306.09852v8),
[official integration](https://github.com/uzh-rpg/acmpc_public).
The isolated solver has its MIT license and pinned provenance in
[vendor source and license](acmpc/vendor/SOURCE.md).
