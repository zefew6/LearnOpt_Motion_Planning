# Formal CUDA whole-body guidance training

This recipe trains independent single-frame observations, not motion rollouts.
Every fixed forest/factory map supplies many different collision-free current
states and one fixed eight-dimensional goal from XML. Full geometry supplies
training-only whole-body costs/gradients. No expert trajectory is required.
Start-to-goal global reachability is not certified for every random state.

## Data and model

`configs/yopo/dataset/yopo_fixed_dense.yaml` creates 20,000 frames in ten fixed
maps: 14,000 train, 4,000 validation, 2,000 test. Each map contains the large
multi-portal walls and varying-height trees from the authored outdoor benchmark.
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

## Optimization and throughput

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

Measured dense-map pilot(width64, batch16, RTX4060): warm steps approximately
0.14s with CUDA guidance, versus9.8s with CPU exact guidance. These use different
collision representations, so this is a throughput comparison, not a claim of
identical cost. Multiple maps per batch and full dataset loading add overhead;
use the formal run's metrics for a runtime estimate.

## Evaluation and recovery

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
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uav_ac.yopo.data.generate \
  --config configs/yopo/dataset/yopo_fixed_dense.yaml \
  --run-dir runs/yopo_whole_body/<data-run>
```

Train:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uav_ac.yopo.training.runner \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/<data-run>/dataset \
  --run-dir runs/yopo_whole_body/<train-run>
```

Resume into a **new** run directory, preserving the existing run:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uav_ac.yopo.training.runner \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/<data-run>/dataset \
  --resume runs/yopo_whole_body/<train-run>/latest \
  --run-dir runs/yopo_whole_body/<resume-run>
```

For an interrupted run, keep the original target steps so the learning-rate
curve and shuffled epoch position continue unchanged. Extending completed
training changes the cosine schedule. Flight YAML checkpoint selections are
not changed automatically by this recipe.
