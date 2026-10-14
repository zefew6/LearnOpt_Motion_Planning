# YOPO planning

YOPO owns independent UAV and whole-body single-frame depth/state planning workflows.
`main.py`, `runners` and `tasks` retain online physics and controller orchestration.

| Location | Responsibility |
| --- | --- |
| `uav/` | Official pure-UAV model adapter, checkpoints and deployment |
| `whole_body/` | Articulated network, datasets, MINCO, guidance training and validation |
| `common/depth.py` | Shared depth calibration and immutable array validation |
| `vendor/yopo_minco/` | Pinned upstream algorithms, source hashes and licenses |
| `_compat.py` | Lazy aliases for previous whole-body imports and CLI entrypoints |

The whole-body `primitive_goal_v5` network predicts 27 candidates (3×3 directions,
three arm priors), each with 34 MINCO parameters: one intermediate configuration,
free terminal PVA and two durations. Existing `direct_v3` and `primitive_v4`
checkpoints remain loadable. Deployment imports no training or privileged-map modules.
Training uses complete scene geometry only to compute trajectory costs/gradients;
the model receives a single depth image, current state and eight-value goal.
Online execution repeatedly plans from the current robot state and tracks with
actual MuJoCo physics.

Whole-body modules now have an explicit `uav_ac.yopo.whole_body` namespace.
Previous imports (`uav_ac.yopo.model`, `data`, `training`, `trajectory`,
`validation`, `deployment`, `contracts`) and module commands remain lazy aliases
to the same implementation. The former `legacy` modules are now named
`data.sampling`, `training.oracle` and `trajectory.parameterization`.
Checkpoint schemas, normalization, dataset paths and weights are unchanged.
Vendor code remains separate from `common`: provenance/namespace adaptations
are third-party integration details, while `common` holds genuinely shared interfaces.

## Official YOPO-MINCO pure UAV

The isolated UAV reproduction lives in `uav/`; pinned upstream modules live in
`vendor/yopo_minco/`, with licenses and per-file hashes in `provenance.json`.
It uses official revision `d5e1e60f83cc991602f142d9b6b96204a24d6b9f`:
a single-channel ResNet18, nine body-frame state/goal inputs and **3×5 = 15**
primitives. Each primitive predicts 14 MINCO parameters, a score and a
20-channel safe-corridor estimate. The highest admitted score is selected;
when no corridor is admitted, the controller holds position and turns toward
the goal. This UAV path does not use the whole-body network or privileged maps.

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/yopo_uav.yaml
```

The preset uses a pure quadrotor in `uav_yopo_bamboo.xml`, an independent
`yopo_uav` block and the official pretrained weights under
`runs/yopo_uav/official_minco_v1/`. Weights are local artifacts and are not
shipped in Git. This directory requires `weights.pth` and `metadata.json`
containing the profile, upstream revision, SHA256 and official configuration;
incompatible whole-body weights are rejected. The existing whole-body preset,
network, datasets and weights remain available through `configs/flight/yopo.yaml`.

The roughly 60×20 m UAV scene has 96 varied-height trees with slender trunks
and closed hemispherical canopies (radii 0.7/1.1 m), staggered central passages,
initial/start/goal heights of 0.9 m, and a goal at x=58 m. Canopies participate
in optical depth and MuJoCo contacts. Scene planning metadata conservatively
covers each mesh with a box; UAV inference does not consume that metadata.

The native optical depth is 90×160 with fx=fy=80 pixels and a 20 m range,
resized with nearest-neighbour sampling to 96×160 and quantized as upstream.
The render copy excludes robot geometry. Explicit adapters convert repository
NED/FRD states to upstream NWU/FLU. Current-state single-frame replanning runs
asynchronously while cascaded feedback runs each physics step; the live viewer
shows actual tracked motion and updated candidate paths. Ordinary viewer runs
write no artifacts. Outdoor overflight is allowed: XML height bounds define a
finite map extent, while runtime enforces horizontal limits and minimum height.

U0 covers source/checkpoint and numerical parity; U1 covers this deployment
interface. Official dataset generation and guidance training (U2) are not yet
implemented. CPU/CUDA core parity and bounded controller-tracked motion have
been checked; reaching the goal in the bamboo scene remains an acceptance
item. Mission arrival uses a 0.5 m position tolerance, low speed and a settling
interval, which is stricter than upstream's 5 m completion notification.

## Whole-body quickstart

From the repository root:

```bash
# Dataset instances and XML copies are written beneath runs/.
MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.whole_body.data.generate \
  --config configs/yopo/dataset/yopo_fixed_dense.yaml \
  --run-dir runs/yopo_whole_body/bamboo_data

.venv/bin/python -m uav_ac.yopo.whole_body.training \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/bamboo_data/dataset \
  --run-dir runs/yopo_whole_body/bamboo_train

.venv/bin/python -m uav_ac.main --config configs/flight/yopo.yaml
```

The flight YAML selects the checkpoint and `simulation/models/yopo_bamboo.xml`: a
24×12 m outdoor bamboo map with varied pole heights and an open winding passage. Dataset
artifacts and old checkpoints remain untouched when the scene/package is changed.
A working graphical desktop is required for the live viewer.

## Detailed guide

- [Network and primitives](#primitives)
- [Guidance objective](#guidance)
- [CUDA training and recovery](#cuda-training)
- [Live benchmark](#online)
- [Paper and code references](#references)

<a id="primitives"></a>

## Network and primitive contracts

The active profile is `primitive_goal_v5`. It keeps the 3×3 intermediate
primitive alignment described below, uses zero minimum intermediate reach, and
predicts a free terminal vector in a bounded ball about the body origin. It
rotates that vector from the primitive frame into the local gravity frame.
The older `primitive_v4` angular terminal contract remains supported when
loading an existing v4 checkpoint. The following geometric details document
that earlier contract and the shared camera/primitive interfaces.

This profile implements the user's selected YOPO-style 3×3 motion-primitive
partition and primitive-frame input/output alignment. It preserves independent
single-frame training and actual-state online replanning. ESDF remains privileged
training/evaluation information; deployment does not use it to filter candidates.

### Representation

The complete calibrated 96×160 depth/mask image enters one CNN. The nine primitive
cells cover left/centre/right and up/centre/down camera angles. Backbone features
are sampled at the projected centres of these cells, then each cell uses the same
1×1 convolutional head. CNN feature-map dimensions are not the primitive count:
the existing stride-32 backbone still produces 3×5 spatial features. We retain
camera intrinsics/extrinsics and match nine prediction sites to the nine angular
cells without cropping independent images or changing the depth dataset.

`primitive.geometry` constructs a right-handed orthonormal primitive frame for
each camera direction in FRD. The camera-frame ray is proportional to
`[tan(horizontal angle), tan(vertical angle), 1]`. This respects the rectangular
pinhole camera projection; it is not an uncalibrated spherical XYZ offset.

For each cell, current base PVA spatial vectors, linear velocity and goal direction
are rotated from the gravity-local frame through body FRD into the primitive
frame. Body angular velocity and gravity are rotated from FRD directly. Actual
attitude metadata, scalar yaw semantics and arm joints/rates are retained. Joint
angles are not spatial vectors. The goal position feature is clipped as
`goal_delta / max(goal_clip, norm(goal_delta))`, with goal_clip=3 m. The actual
8-D final goal remains unchanged in the observation and training goal cost.

Each cell has three conditioned arm modes. A learned mode embedding feeds the
same prediction head; initial arm configurations use direct/folded/extended
priors. Mode order is **cell-major, arm-minor**: `candidate = cell*3 + arm_mode`,
with cell order top-to-bottom and left-to-right. All 27 candidates get independent
raw predictions and scores. This is an initialization prior, not hard collision
certification or a requirement that three modes remain distinct after learning.

The raw 34 channels predict:

| Channels | Meaning |
| --- | --- |
| 0–2 | Inner camera angular offsets and radial distance |
| 3–7 | Inner yaw and four arm angles |
| 8–10 | Independent terminal camera direction and radial distance |
| 11–15 | Terminal yaw and four arm angles |
| 16–23 | Terminal base velocity in the primitive frame, yaw rate and arm rates |
| 24–31 | Terminal base acceleration in the primitive frame, yaw acceleration and arm accelerations |
| 32–33 | Two positive segment durations |

The inner angular half-range is 1.1 times the cell half-width, allowing modest
overlap; camera-edge angles are clipped to the full FOV. Inner radial range is
0.2–2.5 m. The endpoint is independently predicted across the full camera FOV,
with radius 0.2–3.0 m, rather than restricted to a ±0.25 m residual about a shared
endpoint. Its range is not tied to the inner cell. Joint outputs can explore the
full joint bounds through a tanh map around mode priors. Derivative components
are bounded, while full trajectory feasibility remains a training/evaluation
constraint. Segment durations lie in 0.1–3 s.

Spatial derivatives are rotated primitive→FRD→gravity-local. The differentiable
decoder then emits the existing **physical local MINCO34** layout (inner8,
terminal PVA24, durations2). Existing DirectEvaluator gradients and MINCO solver
are reused; actual current head PVA is preserved exactly. This does not enforce
exact actual attitude/body-rate consistency or certify the full trajectory.

### Contracts and commands

Network and geometric encoding live in `uav_ac/yopo/whole_body/model/{network,primitives}.py`.
Dataset generation, oracle guidance and training stay under `uav_ac/yopo`.
Training selects `profile: primitive_v4` and requires 27 candidates.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.whole_body.training \
  --config configs/yopo/train/yopo_primitive_pilot.yaml \
  --dataset runs/yopo_whole_body/wall_portals_forest_v4/dataset \
  --run-dir runs/yopo_whole_body/my_primitive_pilot
```

Saved checkpoints use schema 2, profile primitive_v4, a distinct 58-value state
layout contract and serialized PrimitiveSpec. Geometry/normalization buffers are
verified against metadata, including all spec values. Legacy schema-1 direct_v3
checkpoints still load through their original network/encoding; cross-profile
training resume is rejected. Both evaluation and deployment use the saved spec.
Existing depth/current-state datasets can be reused; old network weights cannot
be treated as trained primitive_v4 weights.

Experimental live entry point:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/yopo.yaml
```

The current single flight preset uses the CUDA-trained `primitive_goal_v5`
checkpoint `best_17000`, rather than the historical 32-step v4 pilot. To test a
different compatible checkpoint, edit `yopo.checkpoint` in that YAML. Historical
weights and datasets remain available. Ordinary viewer runs write no artifacts.
The entry point uses the existing controller, scene and live overlays, without
a classical map-based fallback.

### Validation limits

The tests cover primitive/camera alignment, independent endpoints, current-head
MINCO continuity, cost gradients, checkpoint round trips and training resume.
These checks establish interface and numerical consistency; they do not establish
successful obstacle avoidance or controller-tracked navigation. The current
flight configuration is for online testing. See the [online task guide](#online).

<a id="guidance"></a>

## Guidance objective

All tracked `configs/yopo/train/*.yaml` now select `guidance_profile:
yopo_minco_v1`. This recipe excludes terminal stopping and initial gyro-matching
costs. Each training item is one current
state and depth image. Online replanning is a deployment operation.

### Reference and adaptations

The reference is official YOPO-MINCO revision
[`d5e1e60`](https://github.com/TJU-Aerial-Robotics/YOPO/tree/d5e1e60f83cc991602f142d9b6b96204a24d6b9f), especially
[`safety_loss.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/safety_loss.py),
[`guidance_loss.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/guidance_loss.py),
[`feasible_loss.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/feasible_loss.py) and
[`yopo_trainer.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/yopo_trainer.py).

This is a whole-body adaptation, not a literal quadrotor reproduction. Actual
robot geometry supplies privileged signed pair distances and analytic witness
Jacobians, with conservative geometry covers pruning far environment pairs.
All queries use the robot's public scratch-state interfaces. Static planning
spheres do not decide contact. The existing ESDF/context remains available for
offline trajectory validation. No privileged geometry enters network inputs
or deployment candidate selection.

The point-robot free-ball-chain certificate is not applied to articulated body
motion. Contact classification uses 17 samples per piece including endpoints;
it is a discrete whole-body check, not a continuous safety certificate. Safe
corridor prediction, geodesic guidance, heading-gated speed shaping and the
optional inner-radius regularizer are outside this four-component baseline.
The upstream near-goal stopping term is explicitly disabled by user request.

### Trajectory cost

The cost is safety + smoothness/acceleration energy + time + feasibility +
goal progress. No `goal_stop` or `continuity_and_peak` component exists in this
recipe. Every weight and threshold is configurable in the `guidance` YAML
block, validated strictly, and expanded into saved checkpoint metadata.

Safety uses shifted exponential barriers on the minimum environment and valid
self-pair distances at each sample. Environment defaults are activation gap
0.15 m, exponential origin 0.05 m and decay 0.05 m. Self activation/decay are
0.015 m. These are explicit whole-body baseline parameters, not the upstream
6 m/s point-robot clearances. An exponent cap of 50 bounds already-invalid
initial-contact costs. Contact means signed distance <=0, independently of
barrier activation or its weight.

For collision-free trajectories, the aggregate is
`(logsumexp(beta*c)-log(M))/beta`, beta=4. For a contacting trajectory it is the
cost at the last sample before first contact (sample zero if already in
contact). There is one focused safety cost; no duplicate safety integral or
additional duration-scaled peak is included. Safety diagnostics allocate the
aggregate to environment/self contributions without changing the total.

Smoothness is exact eight-dimensional minimum-jerk energy using the existing
per-axis weights, multiplied by 0.002. Acceleration energy is an exact
polynomial integral, weighted by 0.002 after position/yaw/joint scale
normalization. Time weight is 1.5. Gradients pass through both waypoint and
free-terminal PVA, and both segment durations.

Physical feasibility sums each constraint's peak positive normalized
overshoot: base speed/acceleration/jerk, yaw rate/acceleration, joint
position/rate/acceleration, thrust, tilt, flatness body rate, and workspace
bounds. Joint constraints and thrust/attitude constraints are whole-body
extensions; this does not certify full coupled actuator-torque feasibility.
Weight is 1.5. The base curvature cap follows the upstream speed-only gradient:
curvature determines a detached speed limit, so its gradient slows motion
without changing path curvature. There is no target-distance speed penalty.

Goal position cost uses parallel/perpendicular vector Smooth-L1 relative to
the start-to-goal ray; perpendicular weight is 0.3 and total goal weight 0.6.
Wrapped yaw and four joint Smooth-L1 preferences extend the target to eight
variables (scales 0.7 rad and 0.8 rad). These are soft preferences, not equality
constraints on every local endpoint. For a contacting candidate, the goal
value remains available for ranking but its entire goal gradient is stopped.

### Generation, selection and gradient audits

For 27-candidate recipes, the positive score `s=softplus(raw_score)` is used
consistently for cross-entropy and detached generation weights. Each image's
candidate costs are normalized by their detached mean. The classification
label is the least-cost collision-free candidate, or the least-cost candidate
when all contact. Physical limits affect the total cost, not the collision
label. Deployment's argmax of raw scores is equivalent to argmax of positive
scores, so weights/network architecture and inference contracts need no
migration. The single-candidate P3 ablation retains cost regression.

Validity metrics use raw physical violations and actual contact flags;
zeroing a cost weight cannot make an infeasible trajectory appear valid.
Margin-only barriers are not contacts. Legacy metric behavior is preserved
for legacy objectives.

The generated backward is intentionally a surrogate when goal gradients or
curvature geometry are detached. `GuidanceBatch.detached_gradients` records
those excluded derivatives. The audit compares full-cost FD with
`guidance_gradient + detached_gradient`; its neural audit separately verifies
the actual surrogate backward and subtracts the excluded contribution from
raw-decoder FD. It still detects accidentally detached terminal outputs.

### Compatibility and reproduction

`TrainingConfig` defaults to `legacy_direct_v3` only for historical API and
checkpoint compatibility. Legacy `DirectEvaluator`/rolling/static objectives
remain unchanged. The CPU reference backend uses `collision_model: exact_geometry`; the CUDA
backend uses `collision_model: esdf_spheres`. Both reject nonzero legacy
`goal_stop_weight` or `peak_weight`. Resume requires
matching recipe/settings; warm-start transfers compatible network weights
and resets optimizer/calibration.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.whole_body.training \
  --config configs/yopo/train/yopo_stage_free.yaml \
  --dataset runs/yopo_whole_body/staged_repair_v1/curriculum_closed_loop_states/dataset \
  --run-dir runs/yopo_whole_body/<new-run>

MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.whole_body.validation.audit \
  --dataset runs/yopo_whole_body/staged_repair_v1/curriculum_closed_loop_states/dataset \
  --checkpoint runs/yopo_whole_body/<new-run>/last \
  --samples 3 --epsilon 1e-5 --output <new-audit.json>
```

Use `python -m uav_ac.yopo.whole_body.validation.evaluation --guidance-profile yopo_minco_v1` for an explicitly common
new objective when comparing checkpoints. The existing acceptance tool runs
selected dense validation plus actual MuJoCo/controller tracking. Training or
audit success does not promote a checkpoint to the live benchmark.

The `local_motion_v1` acceptance gate requires >=90% selected dense validity
per family and actual reaching without collision. The former 0.5 rad/s gyro
handoff gate has been removed; that quantity remains an execution diagnostic.
Historical acceptance reports retain their original rules and results.

<a id="cuda-training"></a>

## CUDA training and recovery

This recipe trains independent single-frame observations, not motion rollouts.
Every fixed map supplies many different collision-free current
states and one fixed eight-dimensional goal from XML. Full geometry supplies
training-only whole-body costs/gradients. No expert trajectory is required.
Start-to-goal global reachability is not certified for every random state.

### Data and model

`configs/yopo/dataset/yopo_fixed_dense.yaml` creates 20,000 frames in ten fixed
maps: 14,000 train, 4,000 validation, 2,000 test. New generation uses the
bamboo-only `yopo_bamboo.xml` benchmark. Existing formal_cuda artifacts retain
the original wall/tree geometry copied into their datasets; they are not rebuilt
by this change.
Small static map variations belong to different splits; maps stay fixed while
current states change. Frames retain self-occlusion. Current positions span the
map, headings point roughly towards the goal with variation, arm configurations
span unfolded/partly folded/folded, and independently sampled PVA includes slow
and moving states. Pose changes always re-render the depth image.

`configs/yopo/train/yopo_cuda_large.yaml` selects `primitive_goal_v5`, 27 candidates
(3x3 directions x three arm priors), 34 MINCO parameters per candidate, width64,
batch16, CUDA0. The validated `yopo_minco_v1` guidance weights are unchanged.
Small numerical cost weights are not smoke-test budgets and are not enlarged
merely because the dataset is larger.

### Optimization and throughput

30,000 updates, approximately34.3 shuffled epochs over14,000 states. Each epoch
visits every selected sample once, including the final partial batch. No
32-state training cap. AdamW lr3e-4, weight decay1e-4, clip norm1; 500-step warmup
then cosine decay to3e-5.

`guidance_backend: torch_esdf` and `collision_model: esdf_spheres` explicitly
select the CUDA training representation. The entire predicted trajectory solve,
polynomial states, full-body FK, ESDF sampling, self-sphere pair distances,
physical constraints, energy and goal cost backward execute in batched torch.
The two-piece MINCO solve/energy use float64 for numerical stability; the network,
FK and distance queries use float32. There is no iterative trajectory optimizer
and no per-predicted-state MuJoCo query in this hot path. CPU is used once per
map to build the privileged0.05m ESDF and public robot kinematic description.
There are zero CPU guidance workers with this backend.

Actual geometry is conservatively covered by spheres, including base, all arm
links and gripper. These spheres plus trilinear voxel distances are a training
approximation; they are not numerically equal to MuJoCo exact geom distances.
The voxel-diagonal uncertainty margin increases safety penalties only; contact
flags use raw signed sphere gaps. Non-adjacent self-sphere pairs and gripper-gap
exclusions come from the same whole-body geometry context. Store the collision
representation and resolution in every checkpoint. Actual-geometry checks remain
available for independent validation through `evaluation --guidance-profile
 yopo_minco_v1` and the online acceptance workflow.

Focused safety aggregation, collision-detached goal guidance, jerk/acceleration
energy, peak feasibility and curvature-detached speed guidance keep the prior
recipe. FD audits use a float64 reference of the **saved representation** and
report intentionally detached derivatives separately. CPU/CUDA same-representation
cost/gradient equivalence and public robot FK equivalence are tested.

The loader bounds shard caching and subsets store indices rather than copies
of every image. Training caches up to eight shards/maps, validation two; no
unbounded oracle cache. Logs separate step and guidance wall times.


### Evaluation and recovery

Initial/final train evaluation uses the same64 evenly distributed states; this
is a sampled metric, not a full-training-set average. Every500 updates evaluate
128 held-out-map validation states. `best_<step>` prioritizes selected valid
fraction then selected cost. Save `step_<step>` at the first update and every100
updates; `latest` is an atomically updated relative symlink to a complete model,
metadata, optimizer and RNG checkpoint. Keep the last three periodic snapshots to bound disk use; best-validation
snapshots are retained. Each retained snapshot is immutable. Test frames
remain untouched by training. Formal offline and real controller navigation
acceptance still need to be evaluated after training.

Generate data:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uav_ac.yopo.whole_body.data.generate \
  --config configs/yopo/dataset/yopo_fixed_dense.yaml \
  --run-dir runs/yopo_whole_body/<data-run>
```

Train:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uav_ac.yopo.whole_body.training \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/<data-run>/dataset \
  --run-dir runs/yopo_whole_body/<train-run>
```

Resume into a **new** run directory, preserving the existing run:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uav_ac.yopo.whole_body.training \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/<data-run>/dataset \
  --resume runs/yopo_whole_body/<train-run>/latest \
  --run-dir runs/yopo_whole_body/<resume-run>
```

For an interrupted run, keep the original target steps so the learning-rate
curve and shuffled epoch position continue unchanged. Extending completed
training changes the cosine schedule. Flight YAML checkpoint selections are
not changed automatically by this recipe.

<a id="online"></a>

## Live benchmark

Launch from the repository root:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/yopo.yaml
```

The preset uses CUDA inference and the fixed-map training run’s `best_17000`
checkpoint (`primitive_goal_v5`). It is an online test configuration; the offline
validation result does not establish successful navigation.

This task opens the MuJoCo viewer before submitting the first plan. Green shows
the selected base trajectory; pale gray shows five diverse alternative candidates,
orange the selected arm grasp-frame path (not another base candidate), purple the preceding selected path,
and blue actual base motion. Alternatives are a diagnostic subset of all 27
candidates, not collision-certified paths. The console prints the selected
candidate index, horizon, candidate positional spread and actual speed. The
viewer remains open at success/failure, and the terminal immediately prints the
outcome. Closing the viewer ends the task. Normal viewer runs write no artifacts.
A graphical desktop/OpenGL and the YAML-selected local checkpoint are required.

### Ownership and configuration

`main.py` dispatches `task: whole_body_planning` to
`runners/whole_body_planning.py`; task execution and actual-state sampling belong
in `tasks/whole_body_reaching.py`. Neural modeling, decoding and deployment live in `yopo/whole_body/model`,
`yopo/whole_body/trajectory` and `yopo/whole_body/deployment.py`; data and training live in
`yopo/whole_body/data` and `yopo/whole_body/training`.

The task selects `planner: yopo`, `controller: cascaded`, and `wind: none`.
The whole-body adapter adds inverse-dynamics arm tracking to cascaded base
control. Its feedback and integral period equal the XML physics timestep;
reference sampling follows YAML `control_dt` (default 10 ms). No robot resets
occur during motion. Viewer reset reinitializes the episode.

Scene geometry and start/goal live in `simulation/models/yopo_bamboo.xml`.
The benchmark starts at 1.2 m above ground; the goal remains at 1.9 m.
The outdoor YAML sets `whole_body.enforce_height_ceiling: false`, so crossing
the XML planning grid's 3.4 m upper extent does not stop flight. Horizontal
bounds, the minimum flight height and physical collision checks remain active.
The existing checkpoint does not receive boundary margins as input. Changing
this execution setting does not change its original training objective.

XML `whole_body_start` and `whole_body_goal` are
`[x,y,z,yaw,q1,q2,q3,q4]` in NED/radians. `whole_body_gripper` fixes the commanded
opening; no manipulation action or object target is present. `planning_bounds`
sets the task workspace. The scene covers about 24 × 12 m and contains 282
bamboo cylinders, 1.09–3.30 m high, with radii 0.045–0.10 m. Poles are spread
across the map. Shorter poles can intersect the arm below the base; taller poles
require base avoidance. A winding passage retains whole-body clearance.
The base height workspace spans 0.8–3.4 m. XML owns all geometry.

`yopo.checkpoint` resolves relative to the YAML; `device` selects CPU/CUDA.
Checkpoint camera/robot contracts are checked. The physical gripper servo may
slightly deviate from its command: this YAML explicitly allows 2 mm through
`gripper_tolerance`; observations retain the actual measured gap. The planner's
default tolerance is 0.5 mm, and all tolerances are bounded by 5 mm.

`yopo.speed_scale` is a deployment adjustment, default 1.0 and bounded to
0.5–2.0. This preset uses 1.0 after the shape/jerk investigation. It divides predicted piece durations by the
factor (clipped to the decoder's 0.1–3.0 s bounds), scales predicted terminal
velocities/accelerations, and re-solves MINCO from the unchanged actual start
PVA. It increases commanded physical motion rather than viewer playback speed.
The original network ranking is retained; this is not speed-conditioned training
or a new collision/dynamics certificate. Nonzero actual start derivatives mean
the resulting path need not be a uniform time-rescaling of the original curve.

`whole_body` options (defaults in `runners/whole_body_config.py`):

| Setting | Meaning | Default |
| --- | --- | --- |
| enforce_height_ceiling | Enforce XML maximum altitude during execution; outdoor preset disables it | true |
| replan_dt | Simulated interval between single-frame planning requests | 0.2 s |
| planning_timeout | Maximum wall time per inference | 1 s |
| max_plan_age | Maximum simulated age of an accepted snapshot | 0.5 s |
| position_tolerance / yaw_tolerance / joint_tolerance | Final configuration tolerances | 0.1 m / 0.1 rad / 0.1 rad |
| velocity_tolerance / body_rate_tolerance / joint_rate_tolerance | Actual settling speed tolerances | 0.1 m/s / 0.2 rad/s / 0.1 rad/s |
| settle_time | Continuous time within configuration and speed tolerances | 0.5 s |
| tracking_error_limit | Maximum base reference error | 2 m |
| stall_time / progress_distance | Detect lack of actual position movement | 12 s / 0.05 m |
| log_dt / threads | Console interval / PyTorch CPU threads | 1 s / 2 |

Control and replanning periods must be integer multiples of their inner periods.
`duration` limits simulated episode time. `visualize: false` runs the same physics
and controller loop without the viewer for diagnostics; it is not playback.

### Online execution

On each request the simulation supplies one current depth image, mask, actual
configuration/velocity and the fixed final goal. Acceleration and jerk use
filtered finite differences of actual measured velocity. Rendering forwards a
scratch copy to align camera pose with current qpos without altering live data.
The model ranks and decodes its local candidates without ESDF/map filtering.
No temporal training input is added by online replanning.

A single daemon worker performs inference while the controller keeps tracking
its previous reference. New trajectories are sampled at their snapshot age,
including the time spent computing them. Stale/late plans, exhausted horizons,
physical collisions, enabled workspace violations and tracking divergence report explicit
failure. Success requires the actual full configuration and speeds to settle.
This is soft real time: Python, rendering and the desktop viewer do not guarantee
hard deadlines. A timed-out native inference cannot be forcibly killed inside a
thread; closing/resetting does not wait for it, and its result is discarded.

### Validation limits

The bamboo scene has a ten-piece whole-body collision-free witness in
`tests/integration/test_whole_body_online.py`. This checks scene feasibility;
the witness is never supplied to the neural planner. Passing it or successfully
loading a checkpoint does not establish successful online navigation.
The selected checkpoint was trained on the earlier wall/tree dataset; changing
the interactive scene does not retrain it. Observe actual base/arm tracking,
collisions and final success/failure in the live viewer.

<a id="references"></a>

## Paper and code references

Repository: https://github.com/TJU-Aerial-Robotics/YOPO
Branch: YOPO-MINCO
Pinned revision: `d5e1e60f83cc991602f142d9b6b96204a24d6b9f`
Upstream license: MIT (Copyright 2024 TJU-Aerial-Robotics, Tianjin University).

Inspected implementation:

- [state_transform.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/state_transform.py): one inner point, terminal PVA, two positive durations; primitive/body transforms.
- [poly_solver.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/poly_solver.py): CPU inference solver, two-piece quintic MINCO and C4 internal continuity.
- [loss_function.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/loss_function.py): separate safety/goal/feasibility/effort/time costs.
- [test_yopo_ros.py](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/test_yopo_ros.py): per-depth-frame replanning, reference or odometry starts, continuing reference derivatives.

The pure-UAV adapter uses the pinned modules under `vendor/yopo_minco/`;
their licenses and per-file provenance are retained there. Whole-body model,
trajectory and guidance implementations live separately under `whole_body/`.
The reference matrix in `tests/fixtures/yopo_minco_reference.py` retains its
MIT license in `tests/fixtures/YOPO_REFERENCE_LICENSE`. Weights, datasets and
runtime reports remain local artifacts under `runs/`.

Primary papers:

- [YOPO original paper](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf): independent depth/state inputs, primitive residuals and differentiable trajectory-cost guidance.
- [YOPO-MINCO paper](https://arxiv.org/abs/2608.15741): two-piece MINCO, trajectory-score ranking and safety weighting.
- [YOPOv2-Tracker](https://arxiv.org/abs/2505.06923): guidance learning and online execution; its tracking/control additions are outside this planning-only stage.
