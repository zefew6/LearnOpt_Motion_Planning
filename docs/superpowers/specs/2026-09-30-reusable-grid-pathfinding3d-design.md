# Reusable 3-D Grid and pathfinding3D A* Design

## Goal

Replace the current one-off occupancy-grid and hand-written A* guide with a reusable 3-D grid map and the `pathfinding3D` A* implementation. Keep the map correct for the existing RRT collision checks and trajectory optimizer, and rename the OMPL adapter to `RRT_connect.py` as requested.

## Existing constraints

- Planning coordinates are NED; positive z points down.
- Static MuJoCo geometry is already exposed as scene geometry and includes boxes, spheres, cylinders, and horizontal planes.
- The current `InflatedOccupancyGrid` is shared by the aerial-manipulator RRT collision checker and the ESDF builder. The ESDF itself is also consumed by trajectory optimization and remains a separate signed-distance representation.
- The A* route is a guide for the whole-body planner. It does not certify the manipulator route; RRT and final trajectory collision validation retain that responsibility.
- The installed project targets Python 3.13. `pathfinding3d` will be a declared, locked project dependency.

## Chosen approach

Create one reusable `GridMap` in `uav_ac/planning/geometry/grid_map.py`, migrate all current occupancy-grid consumers to it, and remove the old `InflatedOccupancyGrid` implementation. Keep `ESDF` in `esdf.py`, building it from `GridMap` data. Place the library integration in `uav_ac/planning/search/A_star.py` and remove the hand-written `aerial_astar_guide.py` search implementation.

This replaces the shared occupancy representation in one migration instead of maintaining parallel old and new maps. The alternative of limiting the new representation to A* would leave RRT and optimization coupled to the old map, contrary to the request for a reusable grid class.

## Components and interfaces

### `GridMap`

`GridMap` owns a read-only 3-D boolean occupancy array, a finite NED origin, and a positive voxel resolution. It provides:

- validated construction from voxel data, axis-aligned boxes, and parsed MuJoCo scene geometries;
- world-to-index and index-to-world conversion with a documented rounding convention;
- bounds and shape properties;
- batched conservative `collision_mask(points, radii, margin)` queries for existing RRT and optimizer callers;
- conversion to a pathfinding3D walkability matrix at a requested search resolution, using conservative collision queries so downsampling cannot erase obstacles.

The MuJoCo geometry rasterizer remains an application-owned converter from parsed geometry to planning voxels; no second XML parser or MuJoCo-specific pathfinding dependency is introduced.

### ESDF

`ESDF` remains a signed-distance field for trajectory optimization. Its occupancy constructor changes to accept `GridMap`; occupancy-specific code is removed from `esdf.py`. Existing distance queries and gradients retain their behavior.

### A* adapter

The `uav_ac/planning/search/A_star.py` adapter uses `pathfinding3d.core.grid.Grid` and `pathfinding3d.finder.a_star.AStarFinder`. It maps NED start and goal positions to free grid nodes, runs 3-D A*, and converts returned node identifiers back to NED world coordinates. It keeps the current guide result and diagnostics contract consumed by the whole-body planner.

Diagonal movement uses the library's `only_when_no_obstacle` policy. Non-grid-aligned endpoint connections and every returned/simplified segment are checked against `GridMap`; an invalid connection is rejected rather than returned as a route. Route resampling and clearance annotations remain guide post-processing, not part of the A* implementation. Search time limits and no-path outcomes are reported through the existing guide metrics.

### RRT-Connect module name

Rename `uav_ac/planning/search/ompl_rrt_connect.py` to `uav_ac/planning/search/RRT_connect.py`. Update planner, package exports, and test imports to the case-sensitive new module path; do not keep a compatibility shim under the old path.

## Dependency and migration

Add `pathfinding3d` to `pyproject.toml` and regenerate `uv.lock`. Migrate imports and types in the pick/place task, MINCO evaluator/planner, geometry and search exports, and existing occupancy/A* test modules. Delete the old occupancy-grid class and custom A* module once all callers use the new implementations.

## Failure behavior and safety

- Reject malformed grids, invalid resolutions, unsupported geometry, and endpoints outside the map using explicit errors or the existing no-guide result contract.
- Preserve conservative obstacle handling during rasterization, downsampling, endpoint attachment, and route simplification.
- The A* guide remains a base-position hint. Whole-body state-space search and trajectory validation continue to provide the final feasibility checks.

## Acceptance criteria

1. A `GridMap` can be constructed from scene geometry and provides stable coordinate/index conversions and conservative batched collision queries.
2. The `pathfinding3D` A* adapter returns world-coordinate routes through free voxels and reports a clear failure when no route or time remains.
3. Existing RRT collision checks and ESDF construction consume `GridMap` without changing their safety semantics.
4. The old occupancy-grid and hand-written A* implementations are removed; no imports reference their old modules or names.
5. The OMPL adapter is available at `RRT_connect.py` and all in-repository imports use that path.
6. The A* adapter is available at `A_star.py` and planner imports use that path.
7. `pathfinding3d` is present in both project dependency metadata and the lock file.
