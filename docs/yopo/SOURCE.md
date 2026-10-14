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

Current reference-snapshot data and nominal reference handoffs do not establish real physics closed-loop performance. P3/P4 training and map-free prediction are described below; real physics closed-loop deployment remains a later stage.

## P3/P4 direct_v3 training implementation

Additional primary sources inspected:

- [YOPO original paper](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf): independent depth/state inputs, primitive residuals and differentiable trajectory-cost guidance.
- [YOPO-MINCO paper](https://arxiv.org/abs/2608.15741): two-piece MINCO, trajectory-score ranking and safety weighting.
- [YOPOv2-Tracker](https://arxiv.org/abs/2505.06923): guidance learning and online execution; its tracking/control additions are outside this planning-only stage.
- Upstream [`yopo_network.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/yopo_network.py), [`yopo_dataset.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/yopo_dataset.py), [`yopo_trainer.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/yopo_trainer.py): spatial ResNet features, independent offline views with state/goal augmentation, detached generation weights/normalization and safety-prioritized cross-entropy ranking.

Our implementation is original repository code, not a copied pretrained network. It uses a reduced-width ResNet18-style backbone, depth plus validity mask, actual-camera-projected spatial tokens, global visual features, state/anchor encoders and a shared candidate head. It keeps one gravity-aligned start-yaw frame rather than imposing upstream per-cell rotations on arm states. Geometry projection accounts for backbone stride and receptive-field centers. There are no safety-corridor distribution heads or RodriguesNetwork.

The main learned profile is now `direct_v3`: eight-dimensional inner configuration, terminal PVA (24), and two durations = **34** outputs. The old 20-variable stationary and 32-variable rate-constrained rolling profiles remain compatible research baselines. Actual measured pose/gyro are accepted; starting angular-rate disagreement is a soft training cost. The simplified flatness model still requires a separate physical-execution assessment.

Self-collision training uses explicit zero additional clearance because authored sphere proxies include tangent joint connections, following the existing workcell convention. Collision masks and geometry remain unchanged. Sampled zero-penalty validity is a proxy diagnostic, not physical or continuous safety certification. P3/P4 checkpoints carry robot, camera, normalization, fixed gripper opening, score semantics and training settings; deployed selection uses only observations and learned scores.
