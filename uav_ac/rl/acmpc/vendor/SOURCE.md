Source: https://github.com/uzh-rpg/mpc.pytorch_acmpc
Commit: 63732fa85ab2a151045493c4e67653210ca3d7ff
License: LICENSE.mit (retained verbatim).

Local changes: convergence diagnostics, batch-independent termination, modern
PyTorch boolean masks / linear algebra, and no unnecessary graph in inference.
The fixed-point backward is the upstream iLQR approximation, not an exact
nonlinear-program sensitivity including second derivatives of dynamics.
