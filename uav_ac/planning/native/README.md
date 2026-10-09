# Native planning extensions

Run all commands from the repository root. A fresh clone includes the `.pyx`
sources and build configuration; it does not need generated C or binaries from
another user's machine.

## Install and check

```bash
uv sync --python 3.13
.venv/bin/python -m uav_ac.planning.native status
```

`uv sync` supplies Cython in its isolated build environment and builds the
optional extensions. Building requires a C compiler (GCC/Clang on Linux/macOS,
or MSVC on Windows). `status` lists each available module's binary path; a null
entry means that kernel is absent. Python/NumPy reference implementations remain
available when extensions are absent.

## Rebuild after changing native source

The isolated installation tools are not necessarily installed in `.venv`.
For manual builds, install them there once:

```bash
uv pip install --python .venv/bin/python "Cython>=3.1" "setuptools>=68"
.venv/bin/python -m uav_ac.planning.native build
```

Restart the Python process after rebuilding. Compilation happens during
installation or an explicit build, never during a planning query.

To remove planning build output before a fresh build:

```bash
.venv/bin/python -m uav_ac.planning.native clean
.venv/bin/python -m uav_ac.planning.native build
```

## Source and generated output

Git tracks this guide, `.pyx` sources, Python wrappers, and build configuration.
Generated artifacts are machine-specific and ignored by Git:

| Location | Contents |
| --- | --- |
| `uav_ac/planning/native/*.pyx` | Maintained Cython source |
| `c_generated_code/cython/c/` | Generated C |
| `c_generated_code/cython/obj/` | Compiler objects grouped by platform/Python |
| `c_generated_code/cython/lib/` | Local binaries and package build output |
| `c_generated_code/cython/wheels/` | Built wheels |
| `c_generated_code/cython/archive/` | Retired local build artifacts |

Source checkouts load extensions from the local `lib/` directory; installed
wheels carry extensions in the package. Rebuild locally after changing Python
version or platform rather than copying another machine's binaries.

## Relationship to MPC

Planning extensions compile numerical kernels. acados separately generates an
MPC solver from its model and configuration; it requires its own installation.
Follow [MPC setup](../../../README.md#acados-mpc) before selecting that controller.

Existing MPC sources, libraries, signatures, and `quadrotor_wrench_nmpc_*`
directories keep their locations under `c_generated_code/`; solver JSON stays
at the project root. Planning `build` and `clean` operate only on the `cython/`
subtree. The controller generates its solver on first use and rebuilds when its
cache signature changes.
