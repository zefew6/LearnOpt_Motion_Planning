# Continuous gate racing

The gate-racing task learns ordered flight through six gates directly from
reward. It does not call GCOPTER, acados, or a trajectory-bank generator.
Both PPO-MLP and PPO-ACMPC use the same gate observations and normalized
collective-thrust/body-moment actions. This is a custom three-dimensional
track, not a reproduction of the paper's Split-S geometry or thrust/body-rate
interface. MPVE, wind and dynamics randomization are outside this version.

## Train, resume and evaluate

Run from the repository root with the existing Python 3.13 environment:

```bash
.venv/bin/python -m uav_ac.rl.training --config configs/ppo_gate_racing.yaml --run-dir runs/gate_racing/mlp01
.venv/bin/python -m uav_ac.rl.training --config configs/acmpc_gate_racing.yaml --run-dir runs/gate_racing/acmpc01
.venv/bin/python -m uav_ac.rl.training --config configs/acmpc_gate_racing.yaml --run-dir runs/gate_racing/acmpc01 --resume runs/gate_racing/acmpc01/final_model.zip --total-timesteps 8192
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc01 --mode metrics --episodes 20
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc01 --mode interactive
.venv/bin/python -m uav_ac.rl.evaluate runs/gate_racing/acmpc01 --mode record --output runs/gate_racing/acmpc01/replay.mp4
```

For headless recording, use `MUJOCO_GL=egl` or an available offscreen MuJoCo
backend. Recording requires FFmpeg. Viewer and video use exactly the evaluation
environment and deterministic policy actions. The viewer closes at episode end;
closing its window stops replay. Existing video files are not overwritten.

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
Old trajectory-tracking configurations and checkpoint classes are preserved.

## Task and scene conventions

The XML owns vehicle physics, bounds and ordered `gate_00` through `gate_05`
box sites. Each site's local +X is the crossing direction and its Y/Z sizes
are aperture half-widths. The four visible frame boxes match that aperture.
Changing a gate means moving its containing body, keeping frame and site
together. Simulation geometry uses the repository's MuJoCo convention;
task/dynamics state uses NED/FRD after conversion at the scene boundary.

The fixed track has 2 m square openings, approximately 8–12 m gate spacing,
turns and heights from 2–5 m. Episodes start in hover at the XML start, with
position perturbations ±0.2 m, velocity perturbations ±0.1 m/s and small
quaternion perturbations. All gates must be passed once in order within 30 s.
There is no tracking-speed or tilt cutoff. The physical step is 1 ms and
actions are updated every 10 ms.

Each physical substep tests the motion segment against the current gate plane.
Only a forward intersection strictly inside the opening counts. Later gates
cannot be collected out of order. Collision or leaving scene bounds terminates
immediately, with collision taking priority over a same-substep crossing. The
new vehicle scene includes arm and rotor collision geometry. As with MuJoCo's
discrete collision detection, this does not guarantee collision detection at
arbitrarily large speeds.

Reward sums distance progress toward the current gate, +10 per passed gate,
an additional +10 on completion, and −10 on collision or leaving bounds.
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

The racing cost network emits diagonal quadratic and linear coefficients for
20 control stages and a terminal state. The positive diagonal is bounded to
0.1–10 times fixed base weights; linear corrections are bounded to ±2 times
base weights in scaled coordinates. The local position origin is the current
vehicle position. Zero network output initializes level hover at the current
yaw; no gate reference or desired time trajectory is supplied. The terminal
dummy input is fixed at zero, with positive numerical curvature to keep the
box-QP factorization nonsingular.

Racing and tracking share the existing dynamics and quadratic solver. The
default uses one iLQR update and its approximate differentiable backward,
not an exact nonlinear sensitivity. Input boxes do not represent joint rotor
feasibility; the MuJoCo plant still applies allocation and motor lag. Gate
avoidance is learned from reward rather than enforced as MPC state constraints.

## Metrics and interpretation

Training evaluates fixed initial-state seeds 10000–10019. Standalone evaluation
defaults to held-out seeds 0–19. Best-model selection prioritizes completion
rate, then successful completion time; before any completions it uses mean
gates passed. `evaluation_metrics.json` reports completion/collision/out-of-bounds/timeout
rates, gate counts, speed, completion time, and solver fallback counts.

Evaluation batches active episodes. Inference time is amortized policy-call
wall time per action (CUDA synchronized), excludes physics, and is not a
single-vehicle real-time latency claim. The recorded maximum batch size makes
this interpretation explicit. Training fails on invalid solver outputs and
saves an interrupted checkpoint and diagnostic JSON; evaluation can hold the
previous action, counting every fallback.

Passing the smoke checks establishes runnable training, finite gradients and
save/load/replay compatibility. It does not establish high-speed racing skill,
sample-efficiency superiority or paper-level performance.

## Initial smoke validation (2026-09-12)

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
