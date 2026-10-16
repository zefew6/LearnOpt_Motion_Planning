# Official YOPO-MINCO source boundary

Source: [TJU-Aerial-Robotics/YOPO](https://github.com/TJU-Aerial-Robotics/YOPO),
branch YOPO-MINCO, pinned commit
`d5e1e60f83cc991602f142d9b6b96204a24d6b9f`. The MIT license and torchvision
license are retained beside this file. `provenance.json` records upstream and
local file hashes. No source is loaded from generated runs at runtime.

The official policy/network, MINCO, dataset sampling and trajectory losses are
copied from this revision. `policy/yopo_trainer.py` retains the upstream
`forward_and_compute_loss`, body/world lifting and diagnostics. Its candidate
loss and radius NLL methods delegate to functions extracted from the same file
in `loss/selection_loss.py`; both workflows call these functions.

Local adaptations:

- Imports are namespaced; construction uses a scoped configuration/device.
- Dataset paths and training run paths are explicit. Maps, PNGs and poses remain
  the official data format. Pillow plus floor-index nearest resize replaces
  OpenCV, and NumPy's RandomState permutation reproduces sklearn ShuffleSplit.
  State/goal sampling formulas and normalization are retained.
- A shared scalar PLY reader replaces Open3D XYZ loading. Official voxel bounds,
  occupancy, SciPy EDT, trilinear lookup and trajectory losses are retained.
- `uav.training` owns CLI/config, epoch iteration, partial batches, metrics,
  TensorBoard, optimizer/RNG resume and strict deployment metadata. Defaults
  (50 epochs, batch 16, AdamW, 1.5e-4, seed 0) come from upstream
  [train_yopo.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/train_yopo.py).
  Rich progress bars, automatic log-folder numbering and atexit writes are not
  retained. Missing requested CUDA is an error. CPU uses unfused AdamW.

## Articulated extension

`whole_body.model.network.OfficialWholeBodyPlannerNet` directly uses this
ResNet18 and YopoHead, changing input channels to depth/mask and head dimensions
to 58 state features plus an 8-value arm embedding, 34 MINCO parameters, score
and 20 clearance channels. Camera calibration and body/gravity/primitive frame
adapters remain repository-owned. Nine spatial cells times three arm modes
replace the UAV's fifteen cells. The fixed gripper is outside learned outputs.

Whole-body MINCO and geometry-dependent costs remain under `whole_body/`:
eight axes, FK, body sphere/environment ESDF distances, self-collision and
vehicle/joint feasibility. Those quantities cannot use the upstream point-UAV
safety loss unchanged. Official generation/ranking/inner-radius/NLL functions
are reused directly, including detached score weights and detached labels.
The radius label is sampled minimum environmental clearance over all robot
spheres. It is not an articulated free-ball certificate. Near-goal stopping
and heading-gated shaping remain absent from the articulated trajectory cost.

Schema 4 / `official_whole_body_v1` separates these weights from earlier custom
whole-body architectures. Their loaders, datasets and flight presets remain
supported; old weights are not silently converted to the new architecture.
