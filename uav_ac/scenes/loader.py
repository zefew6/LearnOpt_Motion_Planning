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
    collision_geometries: tuple["SceneGeometry", ...]
    pick_place: "PickPlaceMetadata | None"


@dataclass(frozen=True)
class SceneGeometry:
    """Static primitive expressed in NED coordinates."""

    name: str
    kind: str
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    contype: int
    conaffinity: int

    @property
    def half_extents(self):
        if self.kind == "sphere":
            return np.full(3, self.size[0])
        if self.kind == "cylinder":
            axis = self.rotation[:, 2]
            return (self.size[1]*np.abs(axis)
                    + self.size[0]*np.sqrt(np.maximum(0., 1.-axis**2)))
        if self.kind == "box":
            return np.abs(self.rotation) @ self.size[:3]
        raise ValueError(f"geometry '{self.name}' has no finite half extents")


@dataclass(frozen=True)
class PickPlaceMetadata:
    """Typed mission targets owned by an aerial-manipulator scene XML."""

    pick_position_ned: tuple[float, float, float]
    place_position_ned: tuple[float, float, float]
    pick_yaw: float
    place_yaw: float
    pick_nominal_joints: tuple[float, float, float, float]
    place_nominal_joints: tuple[float, float, float, float]
    gripper_open: float
    gripper_closed: float

    def as_mapping(self) -> dict:
        return {
            "pick_position_ned": self.pick_position_ned,
            "place_position_ned": self.place_position_ned,
            "pick_yaw": self.pick_yaw,
            "place_yaw": self.place_yaw,
            "pick_nominal_joints": self.pick_nominal_joints,
            "place_nominal_joints": self.place_nominal_joints,
            "gripper_open": self.gripper_open,
            "gripper_closed": self.gripper_closed,
        }


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
        if not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("MuJoCo numeric 'planning_bounds' must be finite and ordered")
    pick_place = _extract_pick_place(model)
    geometries = _extract_collision_geometries(
        model, data, validate_static=pick_place is not None)
    if pick_place is not None:
        if bounds is None:
            raise ValueError("scene pick/place metadata requires planning_bounds")
        _validate_pick_place(model, bounds, pick_place)
    return SceneMetadata(
        Path(model_path).expanduser().resolve(), start, goal,
        _extract_mission_waypoints(model, data, start, goal), bounds,
        _geometry_aabbs(geometries), _extract_gcs_guide_paths(model, data),
        geometries, pick_place,
    )


def _extract_collision_geometries(model: mujoco.MjModel, data: mujoco.MjData, *,
                                  validate_static=False):
    geometries = []
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    robot_bodies = set()
    if validate_static:
        for body in range(model.nbody):
            parent = body
            while parent > 0 and parent != base:
                parent = int(model.body_parentid[parent])
            if parent == base:
                robot_bodies.add(body)
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        body = int(model.geom_bodyid[geom_id])
        if validate_static and body not in robot_bodies:
            if name is None or not name.startswith(("obstacle_", "ground")):
                raise ValueError(
                    f"static planning geometry '{name or geom_id}' must be named "
                    "obstacle_* or ground")
            if model.body_jntnum[body] or model.body_mocapid[body] >= 0:
                raise ValueError(
                    f"dynamic planning geometry '{name}' is unsupported; use static XML geoms")
        if name is None or not name.startswith(("obstacle_", "ground")):
            continue
        geom_type = int(model.geom_type[geom_id])
        kinds = {
            mujoco.mjtGeom.mjGEOM_BOX: "box",
            mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
            mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
            mujoco.mjtGeom.mjGEOM_PLANE: "plane",
        }
        if geom_type not in kinds:
            raise ValueError(
                f"MuJoCo planning geometry '{name}' has unsupported type {geom_type}; "
                "use box, sphere, cylinder, or plane")
        center = ENU_TO_NED @ data.geom_xpos[geom_id]
        rotation = ENU_TO_NED @ data.geom_xmat[geom_id].reshape(3, 3) @ ENU_TO_NED
        geometries.append(SceneGeometry(
            name, kinds[geom_type], center, rotation.copy(),
            model.geom_size[geom_id].copy(), int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id])))
    return tuple(geometries)


def _geometry_aabbs(geometries):
    obstacles = []
    for geometry in geometries:
        if geometry.kind == "plane":
            continue
        half_extent = geometry.half_extents
        obstacles.append(np.column_stack((geometry.center-half_extent,
                                           geometry.center+half_extent)).reshape(-1))
    return np.asarray(obstacles, dtype=float).reshape(-1, 6)


def _extract_pick_place(model: mujoco.MjModel) -> PickPlaceMetadata | None:
    names = {
        "pick_position_ned": 3, "place_position_ned": 3,
        "pick_yaw": 1, "place_yaw": 1,
        "pick_nominal_joints": 4, "place_nominal_joints": 4,
        "gripper_open": 1, "gripper_closed": 1,
    }
    present = {name for name in names
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name) >= 0}
    if not present:
        return None
    missing = names.keys()-present
    if missing:
        raise ValueError(f"scene pick/place metadata is missing '{sorted(missing)[0]}'")
    result = {}
    for name, size in names.items():
        numeric = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if model.numeric_size[numeric] != size:
            raise ValueError(f"scene numeric '{name}' must contain {size} values")
        address = model.numeric_adr[numeric]
        values = model.numeric_data[address:address+size].copy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"scene numeric '{name}' must be finite")
        result[name] = tuple(map(float, values)) if size > 1 else float(values[0])
    if not (.02 <= result["gripper_closed"] < result["gripper_open"] <= .07):
        raise ValueError("scene gripper widths must satisfy 0.02 <= closed < open <= 0.07")
    return PickPlaceMetadata(**result)


def _validate_pick_place(model, bounds, targets):
    required = [
        (mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_motor_{i}") for i in range(4)
    ] + [
        (mujoco.mjtObj.mjOBJ_JOINT, f"arm_joint_{i}") for i in range(4)
    ] + [
        (mujoco.mjtObj.mjOBJ_JOINT, "quadrotor_freejoint"),
        (mujoco.mjtObj.mjOBJ_SITE, "tool_frame"),
        (mujoco.mjtObj.mjOBJ_SITE, "grasp_frame"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_position"),
        (mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_joint"),
        (mujoco.mjtObj.mjOBJ_JOINT, "gripper_right_joint"),
        (mujoco.mjtObj.mjOBJ_BODY, "payload_marker"),
        (mujoco.mjtObj.mjOBJ_BODY, "pick_target_marker"),
        (mujoco.mjtObj.mjOBJ_BODY, "place_target_marker"),
    ]
    for object_type, name in required:
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id < 0:
            raise ValueError(f"aerial pick/place XML is missing required element '{name}'")
        if (name == "quadrotor_freejoint"
                and model.jnt_type[object_id] != mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("aerial pick/place requires a free 'quadrotor_freejoint'")
        if (name.endswith("_marker")
                and model.body_mocapid[object_id] < 0):
            raise ValueError(f"aerial pick/place element '{name}' must be a mocap body")
    payload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_marker_geom")
    if payload < 0 or int(model.geom_type[payload]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError("aerial pick/place XML requires spherical 'payload_marker_geom'")
    for name in ("pick_position_ned", "place_position_ned"):
        point = np.asarray(getattr(targets, name))
        if np.any(point < bounds[0]) or np.any(point > bounds[1]):
            raise ValueError(f"scene target '{name}' lies outside planning_bounds")
    joint_limits = np.array([
        model.jnt_range[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_joint_{i}")]
        for i in range(4)
    ])
    for name in ("pick_nominal_joints", "place_nominal_joints"):
        joints = np.asarray(getattr(targets, name))
        if np.any(joints < joint_limits[:, 0]) or np.any(joints > joint_limits[:, 1]):
            raise ValueError(f"scene target '{name}' exceeds arm joint limits")


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
