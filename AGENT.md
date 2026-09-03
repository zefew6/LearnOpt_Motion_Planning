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

## Dependencies: acados is an optional local dependency for MPC

Base requirements are Python 3.13+, `uv`, MuJoCo 3.3+, and FFmpeg for recording.
Install the base Python environment with:

```bash
uv sync --group dev
```

MPC additionally requires CasADi and a locally built acados installation. CasADi is
part of the base project dependencies. The acados source tree is declared as the
optional `mpc` extra because its path is machine-specific and is not available in
CI. The default development environment does not install it. After building acados,
install the MPC extra from the repository root:

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
UV_PROJECT_ENVIRONMENT="$PWD/uav" uv sync --group dev --extra mpc
```

The checked-in `pyproject.toml` assumes the standard sibling layout
`../../acados/interfaces/acados_template`. If acados is stored elsewhere, update
the `tool.uv.sources.acados-template` path locally before syncing the `mpc` extra.

Use one project environment consistently. This checkout currently uses `uav/`; make
that choice explicit for `uv` commands:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/uav"
uv sync --group dev --extra mpc
```

The default `uv` project environment is `.venv` when `UV_PROJECT_ENVIRONMENT` is not
set. Do not mix an activated `(uav)` shell with commands that silently target `.venv`.
Confirm the interpreter with `python -c "import sys; print(sys.executable)"`.
Do not assume that an `acados_template` directory on `PYTHONPATH` is a complete
installation: its runtime dependencies include `Deprecated`, `numpy`, `scipy`,
`casadi`, `matplotlib`, and `cython`.

Verify the actual imports before running MPC or GCS-MPC recording:

```bash
ACADOS_SOURCE_DIR=/path/to/acados \
LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}" \
UV_PROJECT_ENVIRONMENT="$PWD/uav" uv run python -c \
  "import casadi; from acados_template import AcadosOcpSolver; print(casadi.__version__)"
```

If acados is missing, report the installation blocker; do not silently substitute the
cascaded controller. After a later base `uv sync` that removes the local interface,
repeat `uv sync --group dev --extra mpc`.

MPC generates local artifacts in `c_generated_code/` and `*_ocp.json`. They are ignored
by Git and may be regenerated after a directory or environment change.

## Run and verify

```bash
# Interactive simulation
UV_PROJECT_ENVIRONMENT="$PWD/uav" uv run python -m uav_ac.main

# Wind-disturbance simulation
UV_PROJECT_ENVIRONMENT="$PWD/uav" uv run python -m uav_ac.wind

# GCS-MPC offscreen recording
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl \
  UV_PROJECT_ENVIRONMENT="$PWD/uav" uv run python -m uav_ac.record_experiments \
  --experiment gcs-mpc --output-dir docs/videos

# Planning tests
UV_PROJECT_ENVIRONMENT="$PWD/uav" uv run python -m pytest -q tests/unit/planning -o addopts=

# Full tests; disable unrelated auto-loaded ROS pytest plugins while keeping coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  UV_PROJECT_ENVIRONMENT="$PWD/uav" uv run python -m pytest -q -p pytest_cov.plugin
```

Generated videos, FIRI images, MuJoCo logs, Python environments/caches, and acados
artifacts should not be committed. See `.gitignore`. Note that `.gitignore` does not
untrack files already in Git; use `git rm --cached` only for explicitly requested
untracking, and verify the staged diff afterward.
