# Differentiable MPC with SB3 PPO

AC-MPC uses a neural cost map followed by a differentiable box-constrained iLQR
solver as the mean of a Gaussian PPO policy. The critic is a separate MLP.
Stock SB3 2.8.0, MuJoCo rewards and the trajectory bank are reused. MuJoCo is
not differentiated. MPVE and learned dynamics are not implemented.

## Run

The default configuration reuses the bank at
`runs/ppo_trajectory/multitraj01/trajectory_bank`. Set `trajectory_bank_path` to
another compatible bank, or null to generate one (generation requires acados).
External assets are read-only and fingerprint checked. Their absolute location
is recorded in run metadata; update it when moving the run.

```bash
uv run python -m uav_ac.rl.acmpc.benchmark \
  --run-dir runs/acmpc_trajectory/exp01 --mode preflight
uv run python -m uav_ac.rl.mlp_baseline.training \
  --config configs/acmpc_trajectory.yaml --run-dir runs/acmpc_trajectory/exp01
uv run python -m uav_ac.rl.mlp_baseline.evaluate runs/acmpc_trajectory/exp01 --mode metrics
uv run python -m uav_ac.rl.acmpc.benchmark \
  --run-dir runs/acmpc_trajectory/exp01 --mode compare
```

The existing `evaluate` interactive/record modes also work. Use `CONTROLLER =
"rl"` and `RL_RUN_DIR` in `uav_ac.main` to deploy either policy. Resume using
`--resume CHECKPOINT.zip`. Old MLP runs retain schema 1; AC-MPC uses schema 2
and validates observation layout, physical parameters and MPC settings.

The short run uses 8,192 steps, 4 environments, seed 42, 128 rollout steps per
environment, minibatches of 64, three PPO epochs and MPC chunks of 32. A full
run can use `--total-timesteps 10000000`; it is not started automatically.
CPU is the default because sequential small iLQR matrix operations can make
CUDA slower. Measure on the target machine before changing the device.

## Interfaces and assumptions

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

## Verification and interpretation

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
`uav_ac/rl/acmpc/vendor/SOURCE.md`.
