"""Explicit MJCF loading and optional NED scene metadata; no planner selection."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ENU_TO_NED = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class SceneMetadata:
    """Scene-owned geometry; missing mission fields remain explicitly absent."""

    model_path: Path
    start_position: np.ndarray
    goal_position: np.ndarray | None
    mission_waypoints: np.ndarray
    space_limits: np.ndarray | None
    obstacles: np.ndarray
    gcs_guide_paths: list[np.ndarray]


def load_scene(model_path: str | Path) -> SceneMetadata:
    """Read exactly the supplied XML path, including MJCF relative includes."""
    path = Path(model_path).expanduser().resolve(strict=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return extract_scene_metadata(path, model, data)


def extract_scene_metadata(model_path, model, data) -> SceneMetadata:
    """Read metadata from an already compiled simulation without recompiling."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    if body_id < 0:
        raise ValueError("MuJoCo scene is missing required element 'quadrotor'")
    start = ENU_TO_NED @ data.xpos[body_id]
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
    goal = None if goal_id < 0 else ENU_TO_NED @ data.site_xpos[goal_id]
    bounds_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "planning_bounds")
    bounds = None
    if bounds_id >= 0:
        if model.numeric_size[bounds_id] != 6:
            raise ValueError("MuJoCo numeric 'planning_bounds' must contain 6 values")
        address = model.numeric_adr[bounds_id]
        bounds = model.numeric_data[address:address + 6].copy().reshape(2, 3)
    return SceneMetadata(
        Path(model_path).expanduser().resolve(), start, goal,
        _extract_mission_waypoints(model, data, start, goal), bounds,
        _extract_obstacles(model, data), _extract_gcs_guide_paths(model, data),
    )


def _extract_obstacles(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    obstacles = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not name.startswith("obstacle_"):
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"MuJoCo planning obstacle '{name}' must be an axis-aligned box")
        if not np.allclose(data.geom_xmat[geom_id].reshape(3, 3), np.eye(3)):
            raise ValueError(f"MuJoCo planning obstacle '{name}' must be axis-aligned")

        center_ned = ENU_TO_NED @ data.geom_xpos[geom_id]
        half_size = model.geom_size[geom_id]
        obstacles.append(np.array([
            center_ned[0] - half_size[0], center_ned[0] + half_size[0],
            center_ned[1] - half_size[1], center_ned[1] + half_size[1],
            center_ned[2] - half_size[2], center_ned[2] + half_size[2],
        ]))
    return np.asarray(obstacles, dtype=float).reshape(-1, 6)


def _extract_mission_waypoints(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        start_position: np.ndarray,
        goal_position: np.ndarray | None,
) -> np.ndarray:
    waypoint_ids = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name is not None and name.startswith("waypoint_"):
            waypoint_ids.append((name, site_id))

    waypoint_ids.sort()
    expected_names = [f"waypoint_{index:02d}" for index in range(len(waypoint_ids))]
    if [name for name, _ in waypoint_ids] != expected_names:
        raise ValueError("MuJoCo mission waypoints must be consecutively numbered from waypoint_00")
    if not waypoint_ids:
        return np.empty((0, 3))

    mandatory_waypoints = np.array([
        ENU_TO_NED @ data.site_xpos[site_id] for _, site_id in waypoint_ids
    ])
    if goal_position is None:
        return np.vstack((start_position, mandatory_waypoints))
    return np.vstack((start_position, mandatory_waypoints, goal_position))


def _extract_gcs_guide_paths(
        model: mujoco.MjModel,
        data: mujoco.MjData,
) -> list[np.ndarray]:
    """Read consecutively numbered GCS corridor guide paths from scene sites."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    prefix = "gcs_route_"
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name is None or not name.startswith(prefix):
            continue
        parts = name[len(prefix):].split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(
                "GCS guide sites must use names gcs_route_<route>_<point>")
        route_index, point_index = map(int, parts)
        grouped.setdefault(route_index, []).append((point_index, site_id))
    if not grouped:
        return []
    if sorted(grouped) != list(range(len(grouped))):
        raise ValueError("GCS guide routes must be consecutively numbered from zero")
    routes = []
    for route_index in range(len(grouped)):
        points = sorted(grouped[route_index])
        if [index for index, _ in points] != list(range(len(points))):
            raise ValueError(
                f"GCS guide route {route_index} points must be consecutively numbered")
        if len(points) < 2:
            raise ValueError("each GCS guide route must contain at least two points")
        routes.append(np.array([
            ENU_TO_NED @ data.site_xpos[site_id] for _, site_id in points
        ]))
    return routes
