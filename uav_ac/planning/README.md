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

## Package structure

The implementation is split by responsibility:

```text
geometry/              shared polytope, ellipsoid, collision and sampling tools
search/                geometric path search (RRT*)
corridor/firi/         FIRI configuration, separation, MVIE and corridor planning
trajectory/minimum_snap.py
trajectory/gcopter/    MINCO, mappings, penalties, L-BFGS and planner orchestration
trajectory/gcs/        CVXPY perspective SOCP, flow rounding and Bezier restriction
pipeline/              mission-level composition of RRT*, FIRI and trajectory planning
```

Public imports follow the package hierarchy directly; no duplicate flat-module
compatibility layer is maintained.

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
