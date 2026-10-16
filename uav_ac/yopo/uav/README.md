# Pure-UAV YOPO

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
interface. The official dataset collector source now lives in
`vendor/yopo_simulator/`; pure-UAV guidance training (U2) is not yet
integrated into this package. CPU/CUDA core parity and bounded controller-tracked motion have
been checked; reaching the goal in the bamboo scene remains an acceptance
item. Mission arrival uses a 0.5 m position tolerance, low speed and a settling
interval, which is stricter than upstream's 5 m completion notification.

## Official UAV dataset collection

The official collector and its map/rendering dependencies are tracked under
`vendor/yopo_simulator/`, pinned to the same revision as the UAV model.
`provenance.json` records upstream and local hashes. Dataset-generation algorithms
are unchanged. Local adaptations add a standalone build, explicit configuration,
refusal to overwrite an existing output directory, and an early CUDA availability
check. Unused ROS viewer nodes and their resources are excluded; map algorithms
and collection dependencies retain separate source provenance.

From the repository root, prepare a new collection run without rendering data:

```bash
.venv/bin/python -m uav_ac.yopo.uav.dataset \
  --config configs/yopo/dataset/yopo_uav.yaml \
  --run-dir runs/yopo_uav/<new-data-run> --prepare-only
```

This writes only `collection.yaml`, resolving the bundled tree resource and
setting the output to `<run-dir>/dataset/`. The default recipe is ten maps with
10,000 images each; preparing it does not generate those images. For a small
collection smoke check, copy the YAML beneath `runs/` and set `env_num: 1` and
`image_num: 2` before running the collector.

Build separately (CUDA toolkit, a compatible C++17 compiler, Eigen3, PCL, OpenCV
and yaml-cpp development packages are required; ROS/catkin is not required):

```bash
cmake -S uav_ac/yopo/vendor/yopo_simulator \
  -B runs/yopo/build -DYOPO_BUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runs/yopo/build --target dataset_generator -j2
```

On systems requiring explicit CUDA architecture selection, add the appropriate
`-DCMAKE_CUDA_ARCHITECTURES=<architecture>` to the configure command. Compilation
and dataset outputs stay under `runs/`.

To collect, use a **new** run directory and omit `--prepare-only`; the launcher
uses `runs/yopo/build/dataset_generator`, overridable with `--binary`:

```bash
.venv/bin/python -m uav_ac.yopo.uav.dataset \
  --config configs/yopo/dataset/yopo_uav.yaml \
  --run-dir runs/yopo_uav/<another-new-data-run>
```

The official output is `dataset/<map-id>/img_<frame-id>.png`,
`dataset/pose-<map-id>.csv` and `dataset/pointcloud-<map-id>.ply`.
These are pure-UAV data, not compatible whole-body numeric shards. Keep both poses
and point clouds for official guidance learning. This migration does not add a
pure-UAV trainer or convert the data into the retired whole-body generator format.
