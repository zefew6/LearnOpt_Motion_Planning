# YOPO whole-body planning

YOPO owns the single-frame depth/state -> whole-body local trajectory workflow.
`main.py`, `runners` and `tasks` retain online physics and controller orchestration.

| Location | Responsibility |
| --- | --- |
| `contracts.py` | Depth calibration and immutable current-state/goal observations |
| `deployment.py` | `YopoPlanner`: checkpoint inference, candidate selection, deterministic decode |
| `model/` | Network variants, 3×3 primitive alignment, observation encoding, checkpoint IO and tensor FK |
| `trajectory/` | Two-piece MINCO decode and differentiable tensor solve; `legacy.py` retains earlier trajectory contracts |
| `data/` | Dataset schemas, independent-state sampling and generated scene artifacts |
| `training/` | CPU/CUDA guidance, losses, optimizer and checkpoint resume |
| `validation/` | Offline evaluation, candidate audit and closed-loop acceptance tools |

The primary `primitive_goal_v5` network predicts 27 candidates (3×3 directions,
three arm priors), each with 34 MINCO parameters: one intermediate configuration,
free terminal PVA and two durations. Existing `direct_v3` and `primitive_v4`
checkpoints remain loadable. Deployment imports no training or privileged-map modules.
Training uses complete scene geometry only to compute trajectory costs/gradients;
the model receives a single depth image, current state and eight-value goal.
Online execution repeatedly plans from the current robot state and tracks with
actual MuJoCo physics.

From the repository root:

```bash
# Dataset instances and XML copies are written beneath runs/.
MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.data.generate \
  --config configs/yopo/dataset/yopo_fixed_dense.yaml \
  --run-dir runs/yopo_whole_body/bamboo_data

.venv/bin/python -m uav_ac.yopo.training \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/bamboo_data/dataset \
  --run-dir runs/yopo_whole_body/bamboo_train

.venv/bin/python -m uav_ac.main --config configs/flight/yopo.yaml
```

The flight YAML selects the checkpoint and the single interactive XML. Dataset
artifacts and old checkpoints remain untouched when the scene/package is changed.
A working graphical desktop is required for the live viewer.

See [guidance design](../../docs/yopo/GUIDANCE_MINCO.md),
[CUDA training](../../docs/yopo/CUDA_TRAINING.md),
[primitive design](../../docs/yopo/PRIMITIVE_V4.md),
[source mapping](../../docs/yopo/SOURCE.md), and
[historical experiments](../../docs/yopo/experiments/).
