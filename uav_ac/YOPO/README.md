# YOPO whole-body planning: P0–P2

This package prepares ideal-depth observations and provides training-only MINCO guidance. Network training and learned deployment (P3+) are not implemented yet.

Run from the repository root with the existing Python 3.13 environment:

```bash
.venv/bin/python -m uav_ac.YOPO.generate \
  --config configs/yopo_whole_body.yaml \
  --run-dir runs/yopo_whole_body/my_dataset
```

The output directory must not already exist. Defaults produce 12 XML scene instances and 384 samples, split by scene into 8/2/2 train/validation/test instances. Each sample has valid, collision-checked stationary endpoints; no path solver selects samples, so reachability and fixed-3-second trajectory feasibility remain unconfirmed. Ground height and obstacle geometry vary with the recorded seed. A dataset includes a copy of the robot XML, numeric NPZ shards, checksums, a JSON manifest and rejection/range statistics.

## Observation and coordinate boundary

`planning/end2end/contracts.py` defines `DepthSpec` and immutable `PlannerObservation`. Observations contain optical-z depth and a validity mask, robot configuration (12), tangent velocity (11), start PVA (3×8) and goal configuration (8). They do not contain a scene ID, ESDF, obstacle map or simulation handle. Raw physical state uses NED/FRD. The shared local frame uses the start position/yaw and retains the gravity direction. `encode_observation` emits a two-channel depth batch and a 58-value state feature vector; `STATE_LAYOUT` documents its order. FRD angular velocity remains in its body frame. Both start and goal joint positions are retained and normalized by actual limits.

`simulation/depth_observation.py` provides a reusable offscreen `IdealDepthObserver`. It validates focal lengths, centered principal point, robot-relative optical mounting and clipping range. A render-only model/data copy excludes planning sites and debug markers by render groups without modifying live physics. Physical robot geometry still occludes the depth sensor. Out-of-range/background returns have mask=false and depth=far_m; near returns clip to near_m but remain valid. The sensor is a first-surface observation, not a complete local map. World-mounted cameras are supported for standalone rendering probes; body-frame observation construction requires the declared robot-mounted camera.

## Deterministic decoding

`InteriorMap(head_pva, tail_pva, durations)` operates in the coordinates supplied by its caller. It binds an affine 24-node-variable to 20-free-variable map for four quintic segments. Both horizontal jerk endpoint constraints are eliminated analytically; default durations are `[.75,.75,.75,.75]` seconds. Intermediate joint/physical limits are penalties, not hard guarantees.

The public `decode_trajectory` and `PrivilegedEvaluator.evaluate` consistently accept **start-yaw local** free parameters. Their head/goal/observation state arguments are in world NED. The decoder restores world coefficients and returns the existing aerial reference type, with empty RRT path and explicitly unknown validation status. It performs neither search nor online optimization.

For example:

```python
import numpy as np
from uav_ac.planning.end2end.encoding import LocalFrame
from uav_ac.planning.end2end.parameterization import InteriorMap
from uav_ac.planning.end2end.decoder import decode_trajectory

frame = LocalFrame.from_observation(obs)
tail = np.vstack((frame.local_goal(obs.goal), np.zeros((2, 8))))
mapping = InteriorMap(frame.local_pva(obs.start_pva), tail)
trajectory = decode_trajectory(mapping.seed(), obs.start_pva, obs.goal,
                               gripper_gap=obs.configuration[11])
```

## Privileged guidance and data loading

`PrivilegedEvaluator.from_simulation` creates occupancy/ESDF once for that scene, then reuses the robot's scratch queries. Guidance combines the existing weighted jerk energy, whole-body environment/self collision and physical constraints with exact MINCO adjoints. At least 16 quadrature intervals per segment are used. Components are reported separately. Fixed duration has no trainable time cost, and terminal goals are hard boundaries. Reference-v1 accepts only stationary starts whose actual attitude agrees with the simplified flatness boundary.

`guidance_loss` supplies custom analytic backward to PyTorch; `score_loss` detaches its oracle target. `calibrate_cost_scale` computes a positive common frozen normalization. These helpers do not implement a training loop or launch learning.

```python
from uav_ac.YOPO.dataset import PlanningDataset

dataset = PlanningDataset('runs/yopo_whole_body/my_dataset/dataset', split='train')
obs, privileged = dataset[0]
# Only obs goes to the future network. privileged.scene_path reconstructs the oracle.
```

The loader rejects checksum corruption, invalid numeric shapes, escaping paths and repeated scene IDs/XML identities/shard aliases. `allow_pickle=False` is always used. Public observations are read-only; copy arrays before handing them to legacy simulation reset APIs that normalize inputs in place.

## Verification and limitations

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q \
  tests/unit/planning/end2end tests/unit/yopo \
  tests/integration/test_yopo_depth.py tests/integration/test_yopo_trajectory.py
```

Real rendering needs a working OpenGL backend. Configuration and numerical tests alone do not prove rendering. Full-body flatness/physical penalties use the current simplified model, not a proof for complete coupled MuJoCo dynamics. Dense exact-geometry sampling is an independent diagnostic, not continuous collision certification or flight tracking. No payload, grasp/place events, neural network, online ESDF access or fallback RRT planner is added here.
