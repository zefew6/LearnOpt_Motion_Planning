# Official YOPO references for the rolling infrastructure

Repository: https://github.com/TJU-Aerial-Robotics/YOPO
Branch: YOPO-MINCO
Pinned revision: `d5e1e60f83cc991602f142d9b6b96204a24d6b9f`
Upstream license: MIT (Copyright 2024 TJU-Aerial-Robotics, Tianjin University).

Inspected implementation:

- [state_transform.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/state_transform.py): one inner point, terminal PVA, two positive durations; primitive/body transforms.
- [poly_solver.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/poly_solver.py): CPU inference solver, two-piece quintic MINCO and C4 internal continuity.
- [loss_function.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/loss_function.py): separate safety/goal/feasibility/effort/time costs.
- [test_yopo_ros.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/test_yopo_ros.py): per-depth-frame replanning, reference or odometry starts, continuing reference derivatives.

The production implementation reuses LearnOpt's MINCO and robot interfaces. It extends configuration to eight dimensions, retains whole-body collision/physical gradients and adds starting angular-rate matching. Its 32-free-variable rolling layout is not the upstream 14-variable quadrotor layout. The existing shared local-frame encoding remains; upstream per-cell frames are not blindly substituted for whole-body anchors.

Only the upstream inference matrix is redistributed in `tests/fixtures/yopo_minco_reference.py` for coefficient comparison, with its MIT license in `tests/fixtures/YOPO_REFERENCE_LICENSE`. No upstream model weights, generated code, ROS stack or CUDA simulator dependencies are included.

Current reference-snapshot data and nominal reference handoffs do not establish real physics closed-loop performance. Training, learned candidate ranking and deployment remain later stages.
