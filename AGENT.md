# LearnOpt-Motion-Planning — Agent Guide

Read this file before changing the repository. Work from the repository root,
preserve existing user changes, and keep changes focused on the requested behavior.

## Mission and architecture

This project plans and controls a quadrotor in MuJoCo using NED/FRD coordinates:
`x` forward, `y` right, and `z` down. Negative `z` represents positive altitude.

```text
MJCF scene → MujocoSimulation → waypoints/obstacles
           → RRT*/FIRI/GCS/Minimum Snap/GCOPTER
           → [position, velocity, acceleration, yaw] reference
           → TrajectoryController → cascaded or MPC → MuJoCo
```

Important locations:

- `uav_ac/main.py`: interactive simulation and planner/controller selection;
- `uav_ac/planning/pipeline/mission_planner.py`: mission-level planning composition;
- `uav_ac/simulation/mujoco_sim.py`: MuJoCo adapter, physics loop, and recording;
- `uav_ac/control/`: cascaded controller, MPC controller, and trajectory scheduler;
- `uav_ac/record_experiments.py`: deterministic offscreen videos and FIRI images;
- `uav_ac/simulation/models/*.xml`: vehicle, mission, and obstacle definitions;
- `tests/`: unit and integration tests.

The public control imports are:

```python
from uav_ac.control import CascadedController, TrajectoryController
tracker = TrajectoryController(controller, quad, trajectory, steps_per_reference)
```

Do not add the removed `uav_ac.control.controller` compatibility module or change
controller interfaces unless the task explicitly requests it.

## Required workflow

1. Run `git status --short` before editing; preserve unrelated user work.
2. Read the relevant entry point, implementation, configuration, and tests.
3. Reproduce the issue before changing code when practical.
4. Make the smallest change in the layer that owns the behavior.
5. Use `Path(__file__)` or repository-relative paths; never keep machine-specific
   absolute paths in source or generated configuration.
6. Run focused tests, then the nearest integration test or the full test suite.
7. Report changed files, commands, results, and remaining external dependencies.

Do not use `git reset --hard`, `git checkout --`, or broad deletion to clean the
workspace. Do not modify coordinate conventions, public controller APIs, or unrelated
planners/controllers without an explicit request.

## Dependencies: acados is required for MPC

Base requirements are Python 3.13+, `uv`, MuJoCo 3.3+, and FFmpeg for recording.
Install the Python environment with:

```bash
uv sync --group dev
```

MPC additionally requires CasADi and acados. The current `pyproject.toml` expects the
acados Python template at the repository-relative path `../../acados/interfaces/acados_template`.
Keep that sibling layout, or update `pyproject.toml` for the local installation.

Typical acados setup on a new machine:

```bash
git clone --recurse-submodules https://github.com/acados/acados.git /path/to/acados
cd /path/to/acados
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
cd /path/to/LearnOpt-Motion-Planning
uv sync --group dev
```

Before running MPC or GCS-MPC recording, verify that `casadi` and
`acados_template` import successfully. If acados is missing, explain the installation
blocker; do not silently substitute the cascaded controller.

MPC generates local artifacts in `c_generated_code/` and `*_ocp.json`. They are ignored
by Git and may be regenerated after a directory or environment change.

## Run and verify

```bash
# Interactive simulation
uv run python -m uav_ac.main

# Wind-disturbance simulation
uv run python -m uav_ac.wind

# GCS-MPC offscreen recording
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl \
  uv run python -m uav_ac.record_experiments \
  --experiment gcs-mpc --output-dir docs/videos

# Planning tests
uv run python -m pytest -q tests/unit/planning -o addopts=

# Full tests; disable unrelated auto-loaded ROS pytest plugins
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  uv run python -m pytest -q -o addopts=
```

Generated videos, FIRI images, MuJoCo logs, Python environments/caches, and acados
artifacts should not be committed. See `.gitignore`. Note that `.gitignore` does not
untrack files already in Git; use `git rm --cached` only for explicitly requested
untracking, and verify the staged diff afterward.
