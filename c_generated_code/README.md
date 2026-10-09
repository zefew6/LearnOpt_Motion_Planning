# Generated build output

Generated files are ignored by Git. This README is tracked through an explicit
`.gitignore` exception.

- `cython/` contains optional planning build products and can be rebuilt from the
  tracked `.pyx` sources.
- Existing files at this root and `quadrotor_wrench_nmpc_*` directories belong to
  acados MPC. Planning maintenance commands preserve them. Solver JSON remains
  at the project root.

See the [native planning guide](../uav_ac/planning/native/README.md) for installation,
verification, rebuilding, cleanup, and directory ownership. See
[MPC setup](../README.md#acados-mpc) for acados prerequisites; the controller
creates its solver on first use.
