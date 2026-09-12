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
  --config configs/ppo_gate_racing.yaml \
  --run-dir runs/gate_racing/mlp_smoke
```

For a real training run, copy `configs/ppo_gate_racing.yaml` to a separate
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
  --config configs/ppo_gate_racing_large.yaml \
  --run-dir runs/gate_racing/mlp_large_seed42
```

Use separate run directories for different random seeds:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/ppo_gate_racing_large.yaml \
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
  --config configs/ppo_gate_racing_large.yaml \
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

Edit `configs/flight_gate_racing.yaml` so that `rl.checkpoint` points to the
desired `best_model.zip`, then run:

```bash
.venv/bin/python -m uav_ac.main \
  --config configs/flight_gate_racing.yaml
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
  checkpoint: ../runs/ppo_trajectory/multitraj01/best_model.zip
  device: cuda
```

## ACMPC policy

After the MLP baseline is stable, train the gate-racing ACMPC policy with:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/acmpc_gate_racing.yaml \
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
