# YOPO whole-body planning: P0–P2 rolling revision

This package prepares ideal-depth observations and provides training-only MINCO guidance. Network training and learned deployment (P3+) are not implemented yet.

Run from the repository root with the existing Python 3.13 environment:

```bash
.venv/bin/python -m uav_ac.YOPO.generate \
  --config configs/yopo_whole_body.yaml \
  --run-dir runs/yopo_whole_body/my_dataset
```

The output directory must not already exist. The tracked configuration selects `rolling_v2`; `stationary_v1` remains supported. Defaults produce 12 XML scene instances and 384 samples, split by scene into 8/2/2 train/validation/test instances. Rolling samples have a dynamically consistent reference-sampled starting configuration/rates and a collision-checked global goal; no path solver selects samples, so reachability and fixed-3-second trajectory feasibility remain unconfirmed. Ground height and obstacle geometry vary with the recorded seed. A dataset includes a copy of the robot XML, numeric NPZ shards, checksums, a JSON manifest and rejection/range statistics.

## Rolling profile and official reference

The rolling profile follows official [YOPO-MINCO](https://github.com/TJU-Aerial-Robotics/YOPO/tree/d5e1e60f83cc991602f142d9b6b96204a24d6b9f) variable roles: **one inner waypoint, free terminal PVA and two variable segment durations**. The repository's MINCOQuintic coefficients are compared against the pinned official inference matrix in a licensed test fixture. Production continues to use this repository's robot/solver/ESDF infrastructure; no ROS/CUDA simulator stack is installed.

Eight-dimensional expansion gives 34 physical parameters. Two terminal linear-velocity components are eliminated to match the observed starting body angular velocity, leaving 30 spatial variables plus two times: **32 free parameters**, not 32 waypoints. Times are physical seconds in [0.1,3] per segment; future network activation/bounded primitive transforms are not implemented here. The old four-piece/20-variable stationary decoder remains a compatibility baseline.

```python
from uav_ac.planning.end2end.rolling import RollingMap, decode_rolling
from uav_ac.planning.end2end.encoding import LocalFrame

mapping = RollingMap.from_observation(obs)
free = mapping.seed(LocalFrame.from_observation(obs).local_goal(obs.goal))
trajectory = decode_rolling(free, obs)
```

Use `RollingEvaluator` with rolling observations/32-value free parameters; use `PrivilegedEvaluator` only with the old stationary profile. Rolling guidance adds a terminal goal-progress term and a trainable duration cost; analytic gradients include terminal derivatives and the time-dependent boundary elimination. The global goal is an input, not a hard terminal constraint on each local plan.

`ReferenceSchedule` and `observation_at` support nominal-reference handoff checks, including C2 and FRD angular-rate continuity, monotonically increasing time and explicit horizon exhaustion. They do not implement a trained policy, asynchronous perception or physical execution. `ObservationHistory` retains only observed timed frames and does not reconstruct hidden geometry.

Schema v2 stores timestamp, start jerk and derivative source. Dynamic samples come from existing reference trajectories, with physical snapshot and collision checks; they are explicitly labeled `reference`, **not physics rollout data**. Four scene families include staggered dense obstacles. Schema v1/static datasets remain readable. Old 384-sample data is preserved; new acceptance data is in `runs/yopo_whole_body/p2_rolling_acceptance/`.

For actual offscreen rendering in environments where GLFW is unavailable, run with `MUJOCO_GL=egl` set before Python starts. This verifies rendering, not the interactive viewer.

## Observation and coordinate boundary

`planning/end2end/contracts.py` defines `DepthSpec` and immutable `PlannerObservation`. Observations contain optical-z depth and a validity mask, robot configuration (12), tangent velocity (11), start PVA (3×8), start jerk (8), timestamp/derivative source and goal configuration (8). They do not contain a scene ID, ESDF, obstacle map or simulation handle. Raw physical state uses NED/FRD. The shared local frame uses the start position/yaw and retains the gravity direction. `encode_observation` emits a two-channel depth batch and a 58-value state feature vector; `STATE_LAYOUT` documents its order. FRD angular velocity remains in its body frame. Both start and goal joint positions are retained and normalized by actual limits.

`simulation/depth_observation.py` provides a reusable offscreen `IdealDepthObserver`. It validates focal lengths, centered principal point, robot-relative optical mounting and clipping range. A render-only model/data copy excludes planning sites and debug markers by render groups without modifying live physics. Physical robot geometry still occludes the depth sensor. Out-of-range/background returns have mask=false and depth=far_m; near returns clip to near_m but remain valid. The sensor is a first-surface observation, not a complete local map. World-mounted cameras are supported for standalone rendering probes; body-frame observation construction requires the declared robot-mounted camera.

## Deterministic decoding

`InteriorMap(head_pva, tail_pva, durations)` operates in the coordinates supplied by its caller. It binds an affine 24-node-variable to 20-free-variable map for four quintic segments. Both horizontal jerk endpoint constraints are eliminated analytically; default durations are `[.75,.75,.75,.75]` seconds. Intermediate joint/physical limits are penalties, not hard guarantees.

The public `decode_trajectory` and `PrivilegedEvaluator.evaluate` consistently accept **start-yaw local** free parameters. Their head/goal/observation state arguments are in world NED. This paragraph describes the stationary_v1 API; rolling_v2 uses the separate RollingMap/decode_rolling API above. The decoder restores world coefficients and returns the existing aerial reference type, with empty RRT path and explicitly unknown validation status. It performs neither search nor online optimization.

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

`guidance_loss` supplies custom analytic backward to PyTorch; `score_loss` detaches its oracle target. `calibrate_cost_scale` computes a positive common frozen normalization. The P0–P2 helpers are used by the P3/P4 training loop described below.

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

Real rendering needs a working OpenGL backend. Configuration and numerical tests alone do not prove rendering. Full-body flatness/physical penalties use the current simplified model, not a proof for complete coupled MuJoCo dynamics. Dense exact-geometry sampling is an independent diagnostic, not continuous collision certification or flight tracking. The P0–P2 infrastructure adds no payload or grasp/place events. P3/P4 below add the neural network without online ESDF access or a fallback RRT planner.

## P3/P4 learned planning (direct_v3)

The current primary network lives under `planning/end2end`; dataset generation,
privileged costs and training live here. Existing v1/v2 APIs above remain
research baselines. The learned two-piece trajectory has 34 outputs: inner
configuration (8), terminal PVA (24), and two durations. P3 uses one candidate;
P4 uses nine spatial directions × three arm templates. Anchors are geometric
priors, not a promise of distinct homotopies or safety. They require no map.

The reduced-width ResNet18-style depth/mask backbone retains spatial features.
Camera-projected anchor tokens and pooled visual features are conditioned on
58 state features and the anchor descriptor. A shared head predicts bounded
residuals and candidate scores. Candidate order is exchangeable. Projection
uses actual body attitude, camera mounting and CNN stride. All eight-dimensional
trajectory coordinates share the start-yaw NED frame; arm angles stay in their
physical coordinates. The decoder accepts actual pose/rate model residuals.

Guidance uses existing MINCO analytic adjoints for whole-body environment/self
collision, physical constraints, jerk energy and duration; terminal goal progress
and initial angular-rate mismatch are differentiable soft penalties. Optional
log-mean-exp emphasizes peak sampled safety cost. This is an additive peak term,
not a reproduction of upstream first-collision truncation or curvature limits.
P3 minimizes `mean(J/C) + SmoothL1(score, stopgrad(J/C))`. P4 uses per-observation
detached cost normalization and detached positive score weights for generation,
plus cross entropy for the safest lowest-cost candidate. All candidates receive
guidance. Ranking uses greatest logit; P3 cost regression uses smallest score.
Weights, coefficients and sampling remain configurable research choices.

No expert trajectories are used as labels. The stored reference snapshots supply
observations; scene ESDF and robot geometry are privileged only to the training
oracle. Optional independent goal/state jitter preserves images and actual pose,
and is marked `augmentation`, not physics rollout. Scene-disjoint splits are
preserved. Default smoke configurations use no augmentation to isolate learning.

Run from the repository root with a prepared dataset:

```bash
.venv/bin/python -m uav_ac.YOPO.training --config configs/yopo_p3.yaml \
  --dataset runs/yopo_whole_body/p2_rolling_v2_verified/dataset \
  --run-dir runs/yopo_whole_body/my_p3
.venv/bin/python -m uav_ac.YOPO.training --config configs/yopo_p4.yaml \
  --dataset runs/yopo_whole_body/p2_rolling_v2_verified/dataset \
  --run-dir runs/yopo_whole_body/my_p4
.venv/bin/python -m uav_ac.YOPO.evaluation \
  --dataset runs/yopo_whole_body/p2_rolling_v2_verified/dataset \
  --checkpoints runs/yopo_whole_body/my_p3/last runs/yopo_whole_body/my_p4/last \
  --split validation --samples 8 --output runs/yopo_whole_body/comparison.json
```

Run directories must be new. `--resume <checkpoint-dir>` restores weights,
optimizer and RNG into a new run; extend `steps` while retaining data and training
settings. Explicit CPU smoke configs reflect the currently available hardware.
`best_<step>` selection prioritizes sampled constraint validity, then selected
cost; `last` is the terminal checkpoint. Offline comparison applies one common
objective to both models and reports selected/random/oracle costs and ranks.

```python
from uav_ac.planning.end2end.planner import End2EndPlanner
planner = End2EndPlanner(checkpoint, device='cpu', expected_robot_hash=robot_hash)
plan = planner.plan(observation)  # call again with a fresh actual-state observation
```

Deployment imports no training oracle. The checkpoint checks robot/camera/layout/
normalization and fixed gripper opening. The resulting trajectory is explicitly
uncertified. This API enables later rolling integration; P3/P4 do not wire a learned
policy into the viewer/controller or prove flight tracking. Initial attitude is
not independently controlled by the flat-output model; the soft rate penalty
cannot eliminate actual attitude/model discrepancies. Physical closed-loop
execution belongs to P5. Zero sampled weighted penalties are only a proxy metric.
Use `MUJOCO_GL=egl` before Python for offscreen dataset/integration verification
on machines without a working display. Source comparisons are in [SOURCE.md](SOURCE.md).

## Motion-only coupled corridor/bamboo prototype

`coupled_v3` adds lower obstacles of different heights and front gaps of different
widths. Corridor sides, a physical ceiling and an explicit base altitude band
prevent unrestricted climbing. Bamboo is represented by physical cylinders of
varying height/radius, with low central obstacles and high paired trunks. No
payload, grasp/contact objective or gripper action is included.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.generate \
  --config configs/yopo_coupled_dataset.yaml \
  --run-dir runs/yopo_whole_body/my_coupled_data
```

The default bounded prototype is six physical XML scenes, 24 observations each:
one corridor and one bamboo layout in each train/validation/test split. Each
layout has two source modes: a down-hanging start requiring height reduction,
and a laterally extended start requiring width reduction. These are authored
six-piece MINCO references using the existing eight-dimensional whole-body
trajectory and flatness interfaces, **not optimized planner demonstrations**.
They supply independent state observations, not expert trajectory labels.

Every source is independently checked by the existing aerial-manipulator dense
validator (nominal dt .025 s plus refinement/joint extrema), .05 m environmental
clearance, zero added self-clearance and physical constraints. Freezing the arm
at its actual initial configuration while retaining the entire base trajectory
must cause physical collision, with the base geometry clear. The two modes must
collectively produce both lower-obstacle and front-gap collision witnesses.
This demonstrates coupling along those base routes, not global impossibility
of every fixed-arm detour or continuous collision certification.

The unchanged schema2 observation interface feeds existing `direct_v3` training;
source segments do not change the network's two-piece decoder. Snapshots are
stratified across approach, traversal and restoration; old `snapshot_min/max`
settings apply only to rolling_v2. The coupled source schedule is authored and
fixed, and frames from the same source/layout never cross splits. Numeric
source coefficients and exact-check diagnostics are separate privileged,
checksummed resources under `dataset/sources/`; they are not network inputs.
A rejected source raises a diagnostic and leaves the partial run for inspection.

The verified 144-frame run is `runs/yopo_whole_body/coupled_corridor_bamboo_v2`.
Its `inspection/` contains scene/arm counterfactual figures, true stored depth,
a segmentation-based critical-object visibility audit and a reproducible script.
Only explanatory scene overviews hide ceilings/sidewalls; sensor depth retains
all physical geometry and self-occlusion. All 48 approach frames show both lower
and front obstacles. Visibility at every later frame is not guaranteed.

This is a limited first prototype: narrow variations around two templates,
small endpoint-joint variation and an authored speed/fold schedule. It has not
been used for retraining or physical flight. Review candidate expression range,
speed bounds and arm-template coverage before expecting the P3/P4 smoke models
to solve it. See [coupled dataset acceptance](COUPLED_DATA_REPORT.md).

## Closed-room wall apertures with interior bamboo/tree obstacles

`portal_v4` extends the motion-only prototype with three families: rectangular
wall openings, stepped openings whose available width depends on height, and
two consecutive laterally offset stepped openings. The walls are exact unions
of physical boxes, 28–40 cm thick; the room has physical sides/end walls, floor
and ceiling. Generated XML stores aperture geometry and authored NED source-route
knots. No mesh approximations or sensor mounting changes are introduced.

Between the portal walls, and in approach/exit rooms, thin bamboo-like cylinders,
thicker tree trunks and solid spherical crowns have varied heights. A low central
tree obstacle between the two offset walls additionally constrains downward arm
extension. These are conservative collision primitives, not realistic botanical
meshes or visual-only decorations. Robot dimensions and physics remain unchanged.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.generate \
  --config configs/yopo_portal_dataset.yaml \
  --run-dir runs/yopo_whole_body/my_portal_data
```

The bounded acceptance run is `runs/yopo_whole_body/wall_portals_forest_v4`:
9 independent layouts ×24 observations=216, 72 each train/validation/test. Every
split has all three aperture families, and both source modes stay with their
layout. Source references are still authored six-piece MINCO, not expert labels
or optimized demonstrations. Source validation uses the existing dense exact/
physical validator plus a **strict minimum world distance ≥.05 m** generation
gate, rather than relying on the validator's 3 mm tolerance. Frozen-arm witnesses
retain the same full base trajectory, including lateral travel between portals.

Snapshot allocation keeps approach/restoration samples and explicitly root-finds
one sample at the center of **every wall**, for each source mode. `portal_v4`
requires at least eight frames per scene and split scene counts divisible by
three. Observation schema2, separate checksummed privileged sources, `direct_v3`
network decoding and deployment isolation remain unchanged. The dataset is not
used to retrain a policy in this prototype.

`inspection/portal_forest_gallery.png` shows wall shapes, actual depth and arm
counterfactual envelopes. The explanatory overview hides the room shell and
makes portal walls translucent; depth/visibility checks keep every physical
surface opaque and retain self-occlusion. The complete room is present in XML.
There is no guarantee that every obstacle is visible from every snapshot.
See [portal/forest acceptance](PORTAL_DATA_REPORT.md) for evidence and limitations.

First bounded training on the portal/forest prototype uses
`configs/yopo_portal_train.yaml`: existing width16 network,27 candidates, all72
eligible train frames,12 validation frames,200 steps with batch1,CPU, no state/
goal augmentation. It starts from fresh weights; old models/runs are preserved.
Use a new run directory:

```bash
.venv/bin/python -m uav_ac.YOPO.training --config configs/yopo_portal_train.yaml \
  --dataset runs/yopo_whole_body/wall_portals_forest_v4/dataset \
  --run-dir runs/yopo_whole_body/my_portal_training
```

Checkpoint selection prioritizes sampled validity, then cost on validation;
held-out test data is used only after training. Neither that sampled metric nor
a successful training run establishes physical closed-loop performance.
