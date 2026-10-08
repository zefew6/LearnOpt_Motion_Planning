# Generated code ownership

This directory is split by generator and lifetime. Generated files are ignored;
this ownership document is tracked.

- **Root and existing `quadrotor_wrench_nmpc_*` directories:** legacy acados MPC
  sources, objects, solver library and signature. Their locations are preserved
  because `MPCController` loads/caches them here. The solver JSON remains at the
  project root. Native planning tools do not move, clean or overwrite these files.
- **`cython/c/`:** Cython-generated C, grouped under source package/module paths.
- **`cython/obj/`:** compiler objects, separated by platform and Python cache tag.
- **`cython/lib/`:** local extension libraries and package build output. Source
  checkouts import native planning modules from this isolated path; installed
  wheels contain their extensions in the normal package location.
- **`cython/archive/`:** retired local binaries from earlier source-tree builds.

Cython source lives in `uav_ac/planning/native/`, not in this generated directory.
Build with `.venv/bin/python -m uav_ac.planning.native build`, inspect with `status`,
and clean only Cython output with `clean`. Cleaning Cython never deletes MPC files.
Rebuild/restart the Python process after editing native source; no runtime query
performs compilation. NumPy/Python fallbacks remain available when kernels are absent.
