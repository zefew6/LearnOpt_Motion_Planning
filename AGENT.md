# LearnOpt-Motion-Planning: AI and Contributor Guide

This file gives AI assistants and human contributors the minimum context needed
to understand, run, test, and safely modify the project. Work from the repository
root and keep changes limited to the requested task.

## Project purpose

The project plans and controls a quadrotor in MuJoCo. Planning and control use
the aerospace NED/FRD convention:

- `x`: north/forward;
- `y`: east/right;
- `z`: down, so negative `z` represents positive altitude.

The MuJoCo adapter converts between this convention and MuJoCo's ENU/FLU world
and body conventions.

The main pipeline is:

```text
MuJoCo scene
  -> mission waypoints and obstacles
  -> RRT*/FIRI/GCS/GCOPTER/Minimum Snap planning
  -> position, velocity, acceleration, and yaw reference
  -> trajectory scheduler
  -> cascaded or MPC controller
  -> MuJoCo simulation or recording
```

## Repository map

- `uav_ac/main.py`: interactive simulation and planner/controller selection;
- `uav_ac/planning/`: search, corridor, geometry, and trajectory planners;
- `uav_ac/control/`: cascaded control, MPC, and trajectory scheduling;
- `uav_ac/simulation/mujoco_sim.py`: MuJoCo adapter and simulation loop;
- `uav_ac/simulation/models/*.xml`: vehicle and environment scenes;
- `uav_ac/record_experiments.py`: deterministic offscreen recordings and figures;
- `tests/`: unit and integration tests;
- `docs/`: README media and generated documentation assets.

The public controller usage is:

```python
from uav_ac.control import CascadedController, TrajectoryController

tracker = TrajectoryController(controller, quad, trajectory, steps_per_reference)
```

Keep the controller interfaces and coordinate conventions stable unless the task
explicitly requires an API change. The removed
`uav_ac.control.controller` compatibility module should not be reintroduced.

## Environment setup

The supported project workflow uses uv, but the project metadata is standard
Python packaging and can also be installed with pip in a virtual environment.

```bash
uv sync
```

For a regular virtual environment, install the same project extras with:

```bash
python -m pip install -e ".[dev]"
```

The base environment provides the planners, MuJoCo, CasADi, and the cascaded
controller. MPC additionally requires a locally built acados installation. Use
the official [acados installation guide](https://docs.acados.org/installation/index.html)
or the [acados source repository](https://github.com/acados/acados), including
submodules. Then install the acados Python interface into the active environment:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
uv pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
```

Check the installation before running MPC:

```bash
uv run python -c "import casadi; from acados_template import AcadosOcpSolver; print(casadi.__version__)"
```

Do not add machine-specific acados paths to `pyproject.toml`, source files, or
committed configuration. If acados is unavailable, use the cascaded controller
for ordinary simulations and tests.

## Running and testing

Interactive simulation:

```bash
uv run python -m uav_ac.main
```

Optional wind is configured in `uav_ac/main.py` with `WIND_ENABLED` and uses
the same simulation entry point:

```bash
uv run python -m uav_ac.main
```

Planning tests:

```bash
uv run pytest -q tests/unit/planning -o addopts=
```

Full tests with coverage:

```bash
uv run pytest -q -p pytest_cov.plugin
```

For recording, use the project recording entry point and write outputs under
`docs/videos/`. The GCS recording camera is intentionally configured separately:
field of view 45, center `(12, -7, 3)`, azimuth `-5`, and elevation `-5`.
Other recorded experiments use field of view 45, azimuth `45`, and elevation
`-32`.

## Change workflow for AI assistants

Before editing:

1. Run `git status --short` and preserve unrelated user changes.
2. Read the relevant implementation, configuration, and tests.
3. Identify the narrowest module responsible for the requested behavior.

After editing:

1. Run focused tests for the changed module.
2. Run the nearest integration test or the full suite when practical.
3. Run `git diff --check` and inspect the final diff.
4. Report changed files, verification commands, and any external dependency that
   the user must install.

Do not use destructive cleanup such as `git reset --hard`, `git checkout --`, or
broad recursive deletion. Do not change unrelated APIs, planners, controllers,
coordinate conventions, or generated media while fixing a focused issue.

## Generated and local-only files

The following are local or generated artifacts and should remain ignored by Git:

- Python environments such as `.venv/` or `uav/`;
- acados output such as `c_generated_code/` and `*_ocp.json`;
- Python caches, test reports, logs, and build directories;
- local MuJoCo output files.

`.gitignore` does not remove files already tracked. If explicitly requested to
stop tracking an existing generated file, use `git rm --cached` and verify the
staged diff before committing. Do not use `git add -A` as a substitute for
reviewing the staged file list.
