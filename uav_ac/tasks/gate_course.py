"""Seeded, static six-gate courses built from the scene's gate bodies."""

from copy import deepcopy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from uav_ac.scenes.loader import ENU_TO_NED

COURSE_VERSION = 2
COURSE_DEFAULTS = {
    "mode": "fixed", "spacing": [5., 8.], "height": [2., 6.],
    "height_step": 1., "turn_degrees": 60., "width_ratio": [1.3, 4.],
    "height_ratio": [1.3, 4.], "yaw_degrees": 15., "tilt_degrees": 15.,
}
FRAME_HALF_THICKNESS = .08
CLEARANCE = .25


def vehicle_diameter(simulation):
    """Conservative origin-centered collision sphere, including rotor disks."""
    model = simulation.model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    descendants = {body}
    for i in range(body+1, model.nbody):
        if int(model.body_parentid[i]) in descendants:
            descendants.add(i)
    ids = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in descendants
           and (model.geom_contype[i] or model.geom_conaffinity[i])]
    # qpos0 geometry, independent of the current attitude and episode state.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return 2 * max(float(np.linalg.norm(data.geom_xpos[i]-data.xpos[body]) + model.geom_rbound[i])
                   for i in ids)


def course_settings(values=None):
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values) - set(COURSE_DEFAULTS):
        raise ValueError("invalid course settings")
    result = deepcopy(COURSE_DEFAULTS)
    result.update(values)
    if not isinstance(result["mode"], str) or result["mode"] not in {"fixed", "random"}:
        raise ValueError("course.mode must be fixed or random")
    for key in ("spacing", "height", "width_ratio", "height_ratio"):
        value = result[key]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)
                or not np.all(np.isfinite(value)) or not 0 < value[0] <= value[1]):
            raise ValueError(f"invalid course.{key} range")
    for key in ("height_step", "turn_degrees", "yaw_degrees", "tilt_degrees"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid course.{key}")
    if (result["turn_degrees"] > 100 or result["yaw_degrees"] > 30
            or result["tilt_degrees"] > 30 or result["height_step"] >= result["spacing"][0]
            or min(result["width_ratio"][0], result["height_ratio"][0]) < 1.3):
        raise ValueError("course ranges cannot provide a forward, clear approach")
    return result


def frame_boxes(gate):
    """Local centers and half sizes, with the site defining the clear aperture."""
    w, h = gate.half_size
    t = FRAME_HALF_THICKNESS
    return [(np.array(p), np.array(s)) for p, s in (
        ([0, -w-t, 0], [t, t, h+2*t]), ([0, w+t, 0], [t, t, h+2*t]),
        ([0, 0, -h-t], [t, w, t]), ([0, 0, h+t], [t, w, t]))]


def segment_hits_box(start, end, center, rotation, size, padding=CLEARANCE):
    a = rotation.T @ (start-center)
    delta = rotation.T @ (end-start)
    low, high = 0., 1.
    for x, dx, radius in zip(a, delta, size + padding):
        if abs(dx) < 1e-12:
            if abs(x) > radius:
                return False
        else:
            enter, leave = sorted(((-radius-x)/dx, (radius-x)/dx))
            low, high = max(low, enter), min(high, leave)
            if low > high:
                return False
    return True


def valid_course(gates, start, bounds, clearance=CLEARANCE):
    """Conservative geometry checks, not a dynamics feasibility certificate."""
    low, high = bounds
    signs = np.array(np.meshgrid(*[[-1, 1]]*3)).T.reshape(-1, 3)
    for i, gate in enumerate(gates):
        previous = start if i == 0 else gates[i-1].center
        if (gate.rotation.T @ (previous-gate.center))[0] >= -clearance:
            return False
        outer = np.r_[FRAME_HALF_THICKNESS, gate.half_size + 2*FRAME_HALF_THICKNESS]
        vertices = gate.center + (signs * outer) @ gate.rotation.T
        if np.any(vertices < low + clearance) or np.any(vertices > high - clearance):
            return False
        # Disjoint enclosing spheres guarantee nonoverlapping gate frames.
        for other in gates[:i]:
            other_outer = np.r_[FRAME_HALF_THICKNESS, other.half_size + 2*FRAME_HALF_THICKNESS]
            if np.linalg.norm(gate.center-other.center) <= np.linalg.norm(outer)+np.linalg.norm(other_outer)+clearance:
                return False
    # Narrow, rotated apertures need an entry/exit corridor along their normals;
    # a center-to-center chord can cut through a frame even for a flyable layout.
    points = [start]
    for gate in gates:
        offset = 2*clearance*gate.rotation[:, 0]
        points.extend((gate.center-offset, gate.center, gate.center+offset))
    # Include clearance beyond the finish plane.
    points.append(gates[-1].center + gates[-1].rotation[:, 0] * .5)
    if np.any(points[-1] < low+clearance) or np.any(points[-1] > high-clearance):
        return False
    for a, b in zip(points, points[1:]):
        for gate in gates:
            for center, size in frame_boxes(gate):
                if segment_hits_box(a, b, gate.center + gate.rotation @ center, gate.rotation, size, clearance):
                    return False
    return True


def sample_course(rng, start, bounds, settings, diameter=.45):
    from .gate_racing import Gate

    midpoint = (bounds[0] + bounds[1]) / 2
    if (max(settings["height"][0], -start[2]-settings["height_step"])
            > min(settings["height"][1], -start[2]+settings["height_step"])):
        raise ValueError("course.height cannot be reached from the start within course.height_step")
    base_heading = np.arctan2(midpoint[1]-start[1], midpoint[0]-start[0])
    for _ in range(100):
        gates = []
        previous = start.copy()
        heading = base_heading + rng.uniform(-np.pi/6, np.pi/6)
        for index in range(6):
            if index:
                heading += np.deg2rad(rng.uniform(-settings["turn_degrees"], settings["turn_degrees"]))
            distance = rng.uniform(*settings["spacing"])
            altitude = rng.uniform(max(settings["height"][0], -previous[2]-settings["height_step"]),
                                   min(settings["height"][1], -previous[2]+settings["height_step"]))
            dz = -altitude-previous[2]
            horizontal = np.sqrt(distance**2-dz**2)
            center = previous + [horizontal*np.cos(heading), horizontal*np.sin(heading), dz]
            yaw = heading + np.deg2rad(rng.uniform(-settings["yaw_degrees"], settings["yaw_degrees"]))
            pitch, roll = np.deg2rad(rng.uniform(-settings["tilt_degrees"], settings["tilt_degrees"], 2))
            rotation = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
            size = diameter * np.array([rng.uniform(*settings["width_ratio"]), rng.uniform(*settings["height_ratio"])]) / 2
            gates.append(Gate(center, rotation, size))
            previous = center
        if valid_course(gates, start, bounds, diameter/2+.02):
            return gates
    raise ValueError("could not generate a valid six-gate course after 100 attempts; check course ranges and scene bounds")


def apply_course(simulation, gates):
    # Compile from a pristine template: this also rebuilds MuJoCo's collision BVH.
    spec = simulation.scene_specification.copy()
    for index, gate in enumerate(gates):
        body = spec.body(f"gate_{index:02d}")
        if body is None or len(body.geoms) != 4 or len(body.sites) != 1:
            raise ValueError("random courses require six four-sided gate bodies")
        body.pos = ENU_TO_NED @ gate.center
        body.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        body.quat = Rotation.from_matrix(ENU_TO_NED @ gate.rotation).as_quat(scalar_first=True)
        body.sites[0].size = np.r_[.04, gate.half_size]
        for geom, (position, size) in zip(body.geoms, frame_boxes(gate)):
            if geom.type != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("random gate frames must use box geoms")
            geom.pos, geom.size = position, size
    simulation.replace_static_scene(spec)


def describe_course(gates):
    return [{"center_ned": g.center.tolist(), "rotation_ned": g.rotation.tolist(),
             "half_size": g.half_size.tolist()} for g in gates]
