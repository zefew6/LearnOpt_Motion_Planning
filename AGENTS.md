# Contributor and Agent Guide

This guide records the repository boundaries that matter when changing LearnOpt Motion Planning. Start with [README.md](README.md), work from the repository root, and preserve unrelated local changes.

## Project flow and ownership

The interactive application deliberately has one short path:

```text
config YAML → main.py → task runner → XML scene / planner / controller → MuJoCo viewer
```

| Location | Responsibility |
| --- | --- |
| `uav_ac/main.py` | CLI and dispatch to interactive task runners |
| `uav_ac/runners/` | Flight configuration validation and interactive task orchestration |
| `uav_ac/planning/pipeline/` | Mission planning, scene map preparation, and trajectory conversion |
| `uav_ac/planning/native/` | Cython kernel sources and planning build maintenance |
| `uav_ac/control/factory.py` | Controller construction and control period selection |
| `uav_ac/simulation/` | MuJoCo physics, scene metadata, coordinate conversion, wind, and recording |
| `uav_ac/robot/` | Quadrotor and articulated aerial-manipulator dynamics, state, and actuation interfaces |
| `uav_ac/simulation/model/` | Quadrotor, aerial-manipulator, and GCS MJCF models |
| `uav_ac/simulation/models/` | Complete XML scenes, initial vehicle poses, waypoints, bounds, and planner guide sites |
| `uav_ac/planning/` | Geometry, search, corridors, trajectory algorithms, and mission conversion |
| `uav_ac/control/` | Cascaded/MPC/RL controllers and trajectory scheduling |
| `uav_ac/rl/tasks/` | Task-owned configuration, environments, training, and evaluation |
| `uav_ac/rl/common/`, `uav_ac/rl/acmpc/` | Shared RL primitives and stable ACMPC implementation/checkpoint paths |
| `uav_ac/tasks/`, `uav_ac/envs/` | Task state/execution logic and environment support |
| `uav_ac/visualization/` | Planning overlays and BMTP reports |

Keep CLI dispatch short. Put interactive orchestration in `runners`, reusable planning in `planning`, and task state/execution in `tasks`. Scene metadata stays in XML; do not add a second flight configuration hierarchy. Batch recording and RL evaluation have their own entry points.

## Entry points

Interactive planning and flight:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight_quickstart.yaml
```

`uav_ac.main` exposes `load_config`, `run`, and `main`. Controller construction lives in `uav_ac.control.factory.build_controller`; trajectory planning lives in `uav_ac.planning.pipeline.flight.plan_trajectory`. Simulator-independent planner APIs remain under `uav_ac.planning.trajectory`.

Trajectory-bank preparation and RL training:

```bash
uv run python -m uav_ac.rl.training \
  --config configs/ppo_trajectory.yaml \
  --run-dir runs/ppo_trajectory/<run-id> --prepare-only

uv run python -m uav_ac.rl.training \
  --config configs/acmpc_trajectory.yaml \
  --run-dir runs/acmpc_trajectory/<run-id>
```

Ordinary viewer runs write no files. `uav_ac.record_experiments` owns conventional comparison videos; `uav_ac.rl.evaluate` owns trained-policy metrics, viewing, and recording. The old `uav_ac.rl.mlp_baseline` and `uav_ac.rl.gate_racing` imports remain compatibility facades.

RL training and evaluation dispatch through the explicit registry in `uav_ac/rl/common/registry.py`. New tasks belong under `uav_ac/rl/tasks/<task>/`, expose one `WORKFLOW`, and add one registry entry; do not add task-name conditionals to the unified entrypoints.

## Flight configuration contract

`configs/flight.yaml` has three required selectors: `scene`, `planner`, and `controller`. Scene names resolve only within `uav_ac/simulation/models/`; adding a scene requires adding an XML file, not another YAML descriptor.

- Planner values: `none` for gate racing, `mini_snap`, `gcopter`, `gcs`, `bmtp`, and the task-owned
  `aerial_manipulator_minco` planner for `aerial_pick_place`.
- Controller values: `cascaded`, `mpc`, `rl`.
- Common optional values: `speed`, `control_dt`, `wind`, `visualize`, and `seed`.
- `bmtp`, `gcopter`, `gcs`, `cascaded`, `mpc`, `rl`, and `wind_options` may coexist as presets. Runtime reads only the selected planner, controller, and wind block; BMTP always renders its selected initialization and optimized trajectory.
- `rl.checkpoint` and `rl.device` are required only for `controller: rl`; the checkpoint metadata owns its control period.

Reject duplicate keys, unknown settings, missing XML/selected-checkpoint files, and control periods that are not integer multiples of the XML timestep. A planner may validate required scene metadata, but it must never replace the selected scene.

Training configuration remains independent. `configs/ppo_trajectory.yaml` and `configs/acmpc_trajectory.yaml` may contain bank, wind, PPO, and differentiable-MPC settings that are not valid in `flight.yaml`.

## Numerical and serialization contracts

Planning and control use NED world coordinates and FRD body coordinates; positive `z` points down. MuJoCo uses ENU/FLU, so conversion belongs at the simulation boundary.

Controller-ready trajectories contain `[x, y, z, vx, vy, vz, ax, ay, az, yaw, ...]`. Vehicle state has 13 elements: position, scalar-first quaternion, velocity, and body angular velocity. Controllers implement `reset()` and `step(quad, reference) -> ControlCommand`.

Preserve controller timing: cascaded feedback executes every physics step, the reference advances once per control interval, and RL actions are held between policy ticks. Vehicle physics and the physics timestep come from XML.

Robot classes, actuation parameters, and dynamics interfaces live under
`uav_ac/robot/`; reusable robot MJCF components live under
`uav_ac/simulation/model/`, and complete
mission scenes remain under `uav_ac/simulation/models/`. The aerial manipulator uses a
13-value NED/FRD base state plus four named arm-joint positions/rates and a
gripper opening state. The base controller uses total vehicle mass and a
configuration-dependent diagonal inertia approximation; MuJoCo integrates the
full coupled multibody dynamics. Generic link and gripper masses/dimensions are
simulation assumptions. Whole-body pick/place planning uses the
`aerial_manipulator_minco` planner. Static XML scenes provide planning bounds,
pick/place targets, joint poses, and gripper widths; box, sphere, cylinder, and
horizontal plane obstacles are converted to a shared conservative voxel map.

Whole-body planning code should use `simulation.robot` rather than MuJoCo data
arrays. Its 12-value configuration is NED position, scalar-first
FRD-to-NED quaternion, four arm angles and gripper gap; its 11-value tangent
velocity is world NED linear velocity, body FRD angular velocity, four arm
rates and gripper gap rate. FK and analytic Jacobians accept arbitrary
configuration; Jacobian twists are world NED. Dynamics and collision queries
use scratch MuJoCo data and leave the live simulation unchanged. The reduced
gripper dynamics assume ideal symmetric finger motion while the physical model
uses a soft equality constraint and a position servo. Apply rotor and arm
commands through `simulation.robot.apply(...)`; motor response advances once
per physics step. Keep full-body planners, controllers and task policies
independent of private simulation addresses.

Saved policies depend on observation/action layouts, physical parameters, control period, and metadata. Preserve serialized ACMPC class paths and keep `rl_config.json` beside checkpoints. BMTP executable results must remain collision-certified and retain their optimized timing; its viewer overlay shows the selected initial path dashed and the optimized path solid.

## Help a user run the project

Work from the repository root and inspect `git status --short` first. Preserve
existing user configuration and unrelated changes. Use Python 3.13:

```bash
uv sync --python 3.13
.venv/bin/python -m uav_ac.main --help
```

For a first viewer run, use the tracked configuration that needs neither acados
nor an RL checkpoint:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight_quickstart.yaml
```

It selects the lab-course scene, minimum-snap planning, and cascaded control.
The viewer requires a graphical desktop/OpenGL. A headless planning or controller
smoke check does not prove that a viewer opens; report which checks actually ran.
Do not invent a `--headless` CLI option. Preserve the user's requested task when
investigating its failures; the quickstart is only a baseline.

Before running another configuration:

1. Load it with `uav_ac.runners.config.load_config` to validate task selectors,
   scene/checkpoint paths, and options. This does not verify graphics or acados.
2. For `controller: mpc`, follow [README MPC setup](README.md#acados-mpc), check
   `ACADOS_SOURCE_DIR` and shared-library visibility, and verify importing
   `AcadosOcpSolver` in `.venv`. The default `configs/flight.yaml` selects MPC.
   Solver construction still needs code generation/compilation on a cache miss;
   import success alone does not prove solver availability.
3. For `controller: rl`, require an existing compatible `.zip` checkpoint and
   its metadata. Training artifacts under `runs/` are not shipped. Inspect the
   requested device if CUDA is unavailable rather than silently changing it.
4. For planning acceleration, run
   `.venv/bin/python -m uav_ac.planning.native status`. Missing kernels use
   reference implementations. Follow the [native guide](uav_ac/planning/native/README.md)
   for compiler requirements and rebuilding; do not compile inside a flight query.
5. Launch the selected configuration from the same project environment. Distinguish
   configuration, missing dependencies/artifacts, graphics errors, planning failure,
   and tracking failure from the actual traceback and measurements. Do not tune
   controllers or alter scenes merely to hide an environment problem.

Planning builds recreate `c_generated_code/cython/` from tracked sources. acados
creates its own solver output on first use or a signature change. Never request
another user's generated binaries or delete MPC caches as Cython maintenance.
For aerial pick/place, report failed trajectory validation as a diagnostic;
validation failure must not be turned into an execution refusal without a user
request. Do not claim successful flight from successful planning alone.

## Verification when changing code

Run focused tests first, then the full relevant suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q tests/unit
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q tests/integration
git diff --check
```

For configuration changes, test both valid selections and inactive-option failures. For controller or planner changes, verify a real trajectory and the nearest MuJoCo integration test. For checkpoint changes, verify train/save/load/deploy compatibility.

## Artifacts and safe changes

Trajectory banks, checkpoints, videos, generated acados/Cython code, caches, and local environments are artifacts and should not be committed. Keep generated work under `runs/`, `docs/videos/`, or ignored build directories as appropriate.

Inspect `git status --short` before editing. Do not use destructive cleanup commands, overwrite user runs, or change unrelated planner/controller behavior as part of a structural refactor.
