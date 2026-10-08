## References

This module implements motion planning and trajectory optimization methods
based on the following works:

[1] P. Werner, T. Marcucci, and D. Rus,
**"Biconvex Optimization for Smooth Minimum-Time Trajectories around Convex Obstacles,"**
arXiv preprint arXiv:2608.02834, 2026.

[2] T. Marcucci, M. Petersen, D. von Wrangel, and R. Tedrake,
**"Motion Planning around Obstacles with Convex Optimization,"**
*Science Robotics*, vol. 8, no. 84, eadf7843, 2023.

[3] Z. Wang, X. Zhou, C. Xu, and F. Gao,
**"Geometrically Constrained Trajectory Optimization for Multicopters,"**
*IEEE Transactions on Robotics*, vol. 38, no. 5, pp. 3259–3278, 2022.

[4] Q. Wang, Z. Wang, M. Wang, J. Ji, Z. Han, T. Wu, R. Jin, Y. Gao,
C. Xu, and F. Gao,
**"Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions
of Obstacle-Free Space,"**
*IEEE Transactions on Robotics*, vol. 41, pp. 3223–3243, 2025.

## 3D FIRI API

Create one planner per static scene, then reuse it for individual regions,
path corridors, or sampled free-space covers:

```python
from uav_ac.planning.corridor.firi import FIRI3D, FIRIConfig

firi = FIRI3D(
    obstacle_points,
    lower_bound,
    upper_bound,
    FIRIConfig(max_iterations=2),
)

corridor = firi.build_safe_flight_corridor(rrt_or_astar_path)
cover = firi.cover_free_space(collision_checked_free_samples)
```

Each returned `FIRIRegion` owns its half-spaces and visualization geometry:
`region.contains(points)`, `region.vertices()`, and `region.edges()`.

## Search library boundaries

`uav_ac.planning.search` contains reusable algorithms only:

- `astar_search(grid_map, start_world, goal_world, ...)` consumes a
  `GridMap` and returns a world-coordinate `AStarResult`. It does not know
  about robots, ESDFs, MuJoCo, or vehicle clearance.
- `RRTStar` receives state, edge-validity, sampling, and distance callbacks;
  mission-level obstacle geometry is adapted by `pipeline/mission_planner.py`.
- `plan_rrt_connect` receives a `StateSpaceAdapter` protocol implementation;
  the core owns the OMPL planner lifecycle while each planner supplies its
  state-space representation and sampling-bound conversion.

The aerial manipulator MINCO planner keeps its GridMap/ESDF inflation,
clearance costs, guide diagnostics, and `R3 × SO2 × R4` state-space conversion
in its local `search_adapter.py`. This keeps the generic package independent
of aerial-specific dimensions and lets other planners provide their own
adapters.

## Aerial-manipulator pick and place

The whole-body manipulation task plans each leg in
`[x, y, z, yaw, q1, q2, q3, q4]` using OMPL `RRTConnect` on `R3 × SO2 × R4`, an
8-D quintic MINCO spline, and analytic-gradient L-BFGS refinement. The initial
MINCO knots are sampled by equivalent 8-D arc length (0.25 m by default), with
all RRT corners retained; the segment count follows the sampled path. One
positive total-time variable is shared equally across the segments.

At flight initialization, `PickPlacePlanner` voxelizes scene primitives and
ground once and constructs the ESDF. The shared occupancy grid feeds RRT
collision checks and the `edt` package generates the signed Euclidean distance
field; SciPy's first-order `NdBSpline` provides distance queries and analytic
gradients. The RRT searches nominal horizontal body geometry, while MINCO and
dense validation use the full roll/pitch/yaw recovered from flatness. Their
collision costs differentiate through acceleration and yaw, and exact MuJoCo
distances confirm close geometry contacts.

RRT occupancy, self/payload sphere tests and world AABB candidate selection share
an optional Cython broad-phase kernel. Normal package builds include the extension;
for local source edits, rebuild with `.venv/bin/python setup.py build_ext --inplace`.
There is no query-time compilation. Without the extension, a NumPy reference path
remains available. `rrt_broadphase_backend` identifies the active implementation;
`rrt_broadphase_seconds` reports the fused work rather than attributing it to the
older separate occupancy/sphere/AABB timers. Exact MuJoCo distance checks remain
unchanged. Batch candidates reuse their coarse results during exact refinement.

A three-dimensional, base-center grid A* route is computed from that same
occupancy grid and used only to shape the early OMPL position region. Its primary
and fallback search grids, inflation and clearance costs are prepared once in
`AerialAStarMaps`. Both legs and repeated `.plan()` calls reuse them; endpoint
selection and search do not construct maps. Execution reset reuses the planned
trajectories and maps. A changed scene or planner configuration requires a new
`PickPlacePlanner`, with no global cache. Yaw stays
free; arm-joint bounds start around the endpoint configurations, widen for the
A*-guided stage, and then expand to the full limits on one continuing
`RRTConnect` tree. The OMPL 2.0.1 Python bindings do not expose the C++
state-sampler allocator, and OMPL has no classical occupancy-grid A* planner,
so this staged-bound strategy avoids a Python callback sampler while leaving a
global exploration stage. The weighted A* route is a hint, not an optimality
certificate. The `--search-only` benchmark measures pick and place
independently of MINCO; each seed runs in a fresh process by default.

Run the deterministic headless demo with:

```bash
.venv/bin/python -m uav_ac.main --config configs/aerial_manipulator_workcell.yaml
```

Set `visualize: true` to open the existing MuJoCo viewer. The V1 payload is a
kinematic marker: it follows the grasp frame after closure and freezes at the
release pose. Its mass and contact forces do not enter the robot dynamics.
Planning and execution results separately report optimizer convergence,
dense trajectory validation, mission state, collision status, and failure
reason. The final collision check is dense sampling; it does not certify every
continuous-time point between samples.

The aerial manipulator task reads targets and gripper widths from the selected
XML scene. Static boxes, spheres, cylinders, and horizontal ground are shared
by occupancy search, ESDF optimization, and final MuJoCo collision checks.
Adding a compatible robot environment therefore requires a scene XML and a
configuration that selects its filename, not a scene-specific code branch.

## Package structure

The implementation is split by responsibility:

```text
geometry/              shared polytope, ellipsoid, collision and sampling tools
search/                generic GridMap A*, callback RRT*, and adapter RRT-Connect
corridor/firi/         FIRI configuration, separation, MVIE and corridor planning
trajectory/minimum_snap.py
trajectory/gcopter/    MINCO, mappings, penalties, L-BFGS and planner orchestration
trajectory/gcs/        CVXPY perspective SOCP, flow rounding and Bezier restriction
trajectory/bmtp/       independent Bernstein BMTP trajectory/plane alternation
pipeline/              mission-level composition of RRT*, FIRI and trajectory planning
```

Public imports follow the package hierarchy directly; no duplicate flat-module
compatibility layer is maintained.

## BMTP API and reproduction demo

BMTP is implemented independently of the upstream `pybmtp` package and Drake.
It alternates a CVXPY/Clarabel minimum-time trajectory SOCP with maximum-margin
time-varying separating-plane SOCPs, and certifies each Bézier segment with
recursive de Casteljau collision checking:

```python
from uav_ac.planning.geometry.polytope import ConvexPolytope
from uav_ac.planning.trajectory.bmtp import BMTPConfig, BMTPLimits, BMTPPlanner

result = BMTPPlanner(BMTPConfig()).plan(
    collision_free_initial_path, convex_obstacles, planning_domain,
    BMTPLimits(velocity=3.0, acceleration=3.0, jerk=15.0, snap=30.0),
)
assert result.success
samples = result.trajectory.sample(0.01)
```

The interactive entry selects `scene: bmtp_village` and `planner: bmtp` in
`configs/flight.yaml`; its optional `bmtp:` section controls the initial route,
segment count, clearance, and solver overrides. To reproduce the multi-initialization experiment and export `summary.png`,
`summary.svg`, `convergence.png`, `outcomes.png`, `iterations.gif`, JSON, and
NPZ records:

```bash
MPLCONFIGDIR=/tmp/mpl-bmtp .venv/bin/python \
  -m uav_ac.planning.trajectory.bmtp.demo
```

Saved runs can be plotted or checked without re-solving with
`--replay runs/bmtp/<timestamp>`.  `--check-flight` additionally tracks each
certified fixed initialization in MuJoCo and writes `flight_checks.json`.
`bmtp_village.xml` reproduces the paper's deterministic FPP village benchmark:
521 convex boxes in a 34×34×10 workspace and an eight-segment initialization
that naively goes around the village. The dashed path is that poor initial
route; the solid curve is BMTP's certified minimum-time trajectory, which cuts
through the village as collision-triggered separating planes are added.
Red/orange animation states identify rejected collision candidates and newly
tagged obstacles.

## GCS API

The GCS implementation is independent of Drake and MuJoCo. CVXPY formulates
the perspective SOCP and Clarabel is the default numerical backend:

```python
from uav_ac.planning.trajectory.gcs import GCSConfig, GCSPlanner

trajectory = GCSPlanner(GCSConfig()).plan(
    start, goal, firi_regions)
points = trajectory.sample(samples_per_segment=40)
```

GCS constrains only `start` and `goal`; intermediate mission waypoints belong to
the separate RRT/FIRI/GCOPTER workflow.  Its default restriction uses quintic,
C2 Bézier segments and the exact integrated squared-acceleration objective.  A
linear/C0 Bézier surrogate is used for the full-graph relaxation to keep route
selection small; high-order variables are introduced only after deterministic
flow rounding.

Controller samples use one curvature-aware arc-length clock over the complete
selected path.  Analytic Bézier derivatives limit normal and tangential
acceleration, while a separate vertical-speed bound keeps the vehicle inside
the southwest stair opening until it has cleared the intermediate slab.  Long
straight sections still cruise near the configured velocity, and region
boundaries do not each receive the old worst-case segment duration.

The runnable GCS example uses `simulation/models/gcs_building.xml`, a compact
two-storey apartment maze with a different staggered-wall pattern on each
floor.  Its graph is built from a complete collision-free voxel cover, not from
path samples: every safe voxel is greedily merged into a maximal axis-aligned
convex box.  The 0.5 x 0.5 x 0.25 m cover uses a 0.15 m obstacle clearance and
typically needs 38 regions for this scene.  The intermediate slab is closed
outside the stairwell, so every start-to-goal path must climb in the southwest
corner.  This full-space cover is intentionally separate from the ordered
RRT/FIRI corridor used by the GCOPTER laboratory-course example.
