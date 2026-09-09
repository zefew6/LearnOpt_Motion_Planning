# Contributor and Agent Guide

This guide is for contributors and coding agents working on LearnOpt Motion Planning. Start with [README.md](README.md) for installation and runnable examples. Run commands from the repository root.

## Scope and module ownership

The project supports MuJoCo planning experiments and trajectory-tracking RL training and deployment. Custom non-trajectory tasks can implement the Python task interface; they are not yet registered in the YAML CLI.

| Location | Responsibility |
| --- | --- |
| `uav_ac/main.py` | CLI selecting one experiment YAML |
| `uav_ac/experiments/config.py` | YAML parsing, mutually exclusive options, path resolution, and runtime compatibility |
| `uav_ac/experiments/runner.py` | Planning, environment creation, training dispatch, deployment, artifacts, and replay |
| `uav_ac/scenes/` | Explicit XML loading and scene metadata |
| `uav_ac/simulation/` | MuJoCo physics, coordinate conversion, wind primitives, and recording |
| `uav_ac/simulation/models/` | XML scenes and vehicle definitions |
| `uav_ac/tasks/` | Task protocol and existing trajectory-tracking environment |
| `uav_ac/envs/` | Generic `MujocoEnv` for custom tasks |
| `uav_ac/planning/` | Geometry, search, corridors, trajectory algorithms, and mission adapters |
| `uav_ac/control/` | Cascaded/MPC controllers, command interfaces, trajectory scheduler, and tracking policy encoding |
| `uav_ac/deployment/` | Controller/policy adapters for episode execution |
| `uav_ac/rl/training.py` | Shared MLP and ACMPC PPO trainer and bank preparation |
| `uav_ac/rl/common/` | Trajectory banks and initialization assets |
| `uav_ac/rl/acmpc/` | Differentiable MPC policy, solver, dynamics, and diagnostics |
| `uav_ac/visualization/` | Planning overlays and plots |
| `configs/scenes/` | YAML descriptors pointing to XML scenes |
| `configs/experiments/`, `configs/training/` | Experiment and training composition |
| `tests/unit/`, `tests/integration/` | Component and end-to-end verification |

The older `runtime.py`, `record_experiments.py`, and MLP-specific entry points still exist. Some compatibility exports remain in `main.py` and `rl/common/environment.py`. New callers should import the owning module directly. The legacy `config.ini` and flat training YAML files are not additional configuration layers for the unified main entry.

## Entry points and extension contracts

- `uav_ac.experiments.config.load_config(path)` loads and validates experiment YAML.
- `uav_ac.experiments.runner.run(config)` executes a configuration; `make_env(config)` constructs its tracking environment, and `train(config)` dispatches training.
- `uav_ac.planning.plan(planner_config, simulation, dt)` is the mission adapter. Simulator-independent algorithms live under `planning/trajectory/`.
- `uav_ac.deployment.deploy(config)` dispatches configured execution.
- `uav_ac.envs.MujocoEnv(task, model_path=..., steps_per_action=...)` executes a custom `uav_ac.tasks.Task`. Tasks define observation space, reset, observation, reward, and termination.
- Existing tracking controllers implement `reset()` and `step(quad, reference) -> ControlCommand`.

The tracking environment currently has its own implementation in `tasks/trajectory_tracking.py`; it is not a subclass of the generic task environment. Both training and configured deployment use that tracking implementation.

When adding a planner, implement its algorithm and mission adapter, then update configuration validation and scene requirement checks. When adding a CLI task, also register its configuration and runner construction; implementing the Python protocol alone does not make a new task available through YAML.

## Configuration rules

Each option should have one owner and one effective value:

- `scene` selects an explicit XML file; planners must not silently replace it.
- `task.reference.source` selects exactly one of `planner`, `saved_trajectory`, or `trajectory_bank`. Saved references must not have an active `planner` block.
- `agent.type` selects `controller` or `rl`. Training policy settings and deployed checkpoint settings are separate branches.
- Vehicle physics and physics timestep come from XML. `execution.action_dt` must be an integer multiple of that timestep; the stride is derived.
- Deployed RL can explicitly use `action_dt: from_model`. A numeric period must agree with the model metadata.
- Wind force parameters belong to `disturbance.wind`; training wind progression belongs to `training.curriculum`. Custom force wind must not be combined silently with nonzero native XML wind.
- Relative paths are resolved against the YAML containing them. Only `scene.config` includes another YAML; there is no general inheritance stack.
- Reject duplicate YAML keys, unknown options, and incompatible combinations with field-specific errors. Do not add silent environment-variable or INI overrides.
- Display and recording should observe execution without changing physics, random sampling, or optimized trajectory timing.

The unified trainer currently requires an existing trajectory bank and supports no wind or randomized gusts. Bank generation remains available through `python -m uav_ac.rl.training --prepare-only`. Check the full path from configuration to consumer when changing defaults; shared training and environment defaults may still need explicit propagation.

## Numerical and model contracts

Planning and control use NED world coordinates and FRD body coordinates: positive `z` points down. MuJoCo uses ENU/FLU; conversions belong at the simulation boundary.

Controller-ready trajectories contain `[x, y, z, vx, vy, vz, ax, ay, az, yaw, ...]`. Vehicle state has 13 elements: position, scalar-first quaternion, velocity, and body angular velocity. `ControlCommand` contains thrust in newtons and body moments in newton-metres. Normalized RL actions contain four values in `[-1, 1]`.

Preserve controller timing: cascaded feedback runs on every physics substep, with the outer reference updated on control ticks. RL actions are held over the action interval.

Saved MLP and ACMPC policies depend on observation/action layouts, physical parameters, control period, and metadata. Preserve `uav_ac.rl.acmpc.policy.ACMPCPolicy` and other serialized class paths unless a migration explicitly handles existing checkpoints. Keep `rl_config.json` alongside model archives; checkpoint files in a `checkpoints/` subdirectory use the parent run metadata.

BMTP executable results require successful feasibility checks. Keep its optimized timing and report failed initializations as failures. Source attribution for bundled solver code must remain intact.

## Setup and verification

Python 3.13 is required. Dependencies and test tools are installed together; no `dev` extra is defined:

```bash
uv sync --python 3.13
```

For the acados nonlinear controller or MPC-validated bank generation, build acados separately and install its Python interface into this environment:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
uv pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
.venv/bin/python -c "from acados_template import AcadosOcpSolver"
```

Use a real local acados path; do not commit machine-specific paths. Recording requires FFmpeg and a working graphics backend. Headless numerical runs use `output.viewer: false` and `output.record: false`.

Run focused tests first. These commands avoid unrelated installed pytest plugins, including ROS plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q tests/unit
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q tests/integration
git diff --check
```

To use the repository's configured coverage gate, explicitly load the coverage plugin:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -p pytest_cov.plugin -q
```

For configuration changes, verify invalid combinations and effective values at the consumer. For changes to training or model loading, test a short train/save/load/deploy cycle where feasible. For task extensions, verify reset/step/termination without imposing trajectory or waypoint requirements on unrelated tasks.

## Working with artifacts

Inspect `git status --short` before editing and preserve unrelated work. Keep changes within the requested scope and report verification accurately.

Experiment outputs belong under `runs/`. The unified runner writes `resolved_config.yaml` and `experiment.json`; deployment writes episode states and metrics, with optional video. Select a new output directory for a new experiment. Replay consumes recorded states and checks the scene hash.

Do not commit training banks, checkpoints, generated solver code, caches, or local environments. Existing tracked generated files are not removed by `.gitignore`; changing their tracking requires an explicit cleanup task. Preserve existing runs and checkpoints during refactoring.
