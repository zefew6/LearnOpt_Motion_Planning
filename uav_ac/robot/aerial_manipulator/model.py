"""Numerical MuJoCo model interface used by aerial-manipulator planners."""

from dataclasses import dataclass
import mujoco
import numpy as np

S = np.diag([1.0, -1.0, -1.0])  # ENU/FLU coordinates to NED/FRD coordinates
Q_SIGN = np.array([1.0, 1.0, -1.0, -1.0])
GAP_MIN, GAP_MAX = .020, .070
NV_PUBLIC = 11
NQ_PUBLIC = 12


@dataclass(frozen=True)
class ConfigurationLimits:
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    gripper_opening: tuple[float, float]
    rotor_thrust: tuple[float, float]
    joint_torque_lower: np.ndarray
    joint_torque_upper: np.ndarray

    def __post_init__(self):
        for name in ("joint_lower", "joint_upper", "joint_torque_lower", "joint_torque_upper"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gripper_opening", tuple(map(float, self.gripper_opening)))
        object.__setattr__(self, "rotor_thrust", tuple(map(float, self.rotor_thrust)))


class AerialManipulatorModel:
    """Maps public NED/FRD configuration space onto MuJoCo's native state."""

    def __init__(self, model, data_getter):
        self.model = model
        self._live_data = data_getter
        self.base = _id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self.free_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "quadrotor_freejoint")
        self.base_qpos = int(model.jnt_qposadr[self.free_joint])
        self.base_dof = int(model.jnt_dofadr[self.free_joint])
        self.arm_joints = np.array([_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_joint_{i}")
                                    for i in range(4)])
        self.arm_qpos = model.jnt_qposadr[self.arm_joints].copy()
        self.arm_dof = model.jnt_dofadr[self.arm_joints].copy()
        self.gripper_joints = np.array([_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_{s}_joint")
                                        for s in ("left", "right")])
        self.grip_qpos = model.jnt_qposadr[self.gripper_joints].copy()
        self.grip_dof = model.jnt_dofadr[self.gripper_joints].copy()
        self.arm_actuators = np.array([_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_motor_{i}")
                                       for i in range(4)])
        self.grip_actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_position")
        self.rotor_sites = np.array([_id(model, mujoco.mjtObj.mjOBJ_SITE, f"rotor_{i}")
                                     for i in range(4)])
        self.tool_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_frame")
        self.grasp_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_frame")
        self.rotor_directions = model.site_user[self.rotor_sites, 0].copy()
        joint_ranges = model.jnt_range[self.arm_joints]
        numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "rotor_thrust_limits")
        numeric_adr, numeric_size = model.numeric_adr[numeric_id], model.numeric_size[numeric_id]
        rotor_limits = model.numeric_data[numeric_adr:numeric_adr+numeric_size]
        self.limits = ConfigurationLimits(
            joint_ranges[:, 0].copy(), joint_ranges[:, 1].copy(), (GAP_MIN, GAP_MAX),
            tuple(np.asarray(rotor_limits, float)),
            model.actuator_ctrlrange[self.arm_actuators, 0].copy(),
            model.actuator_ctrlrange[self.arm_actuators, 1].copy())
        self._scratch = mujoco.MjData(model)
        self._descendant_bodies = self._find_descendants()
        self._robot_geoms = [g for g in range(model.ngeom)
                             if model.geom_bodyid[g] in self._descendant_bodies
                             and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self._environment_geoms = [g for g in range(model.ngeom)
                                   if model.geom_bodyid[g] not in self._descendant_bodies
                                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self._body_ids = {
            _name(model, mujoco.mjtObj.mjOBJ_BODY, body): body
            for body in range(model.nbody)
        }
        self._body_joint_ancestors = np.zeros((model.nbody, len(self.arm_joints)), dtype=bool)
        for body in range(model.nbody):
            for joint_index, joint in enumerate(self.arm_joints):
                self._body_joint_ancestors[body, joint_index] = self._joint_is_ancestor(
                    body, int(model.jnt_bodyid[joint]))
        self._self_collision_pairs = tuple(
            (a, b) for index, a in enumerate(self._robot_geoms)
            for b in self._robot_geoms[index+1:]
            if not self._adjacent_geoms(a, b) and self._collision_masks_allow(a, b))
        self._environment_collision_pairs = tuple(
            (a, b) for a in self._robot_geoms for b in self._environment_geoms
            if self._collision_masks_allow(a, b))
        payload_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_marker")
        self._payload_mocap = (int(model.body_mocapid[payload_body])
                               if payload_body >= 0 else -1)
        self._payload_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "payload_marker_geom")
        self._gripper_body_ids = {
            self._body_ids[name] for name in
            ("gripper_palm_body", "gripper_left_body", "gripper_right_body")
            if name in self._body_ids
        }

    def _find_descendants(self):
        result = set()
        for body in range(self.model.nbody):
            parent = body
            while parent > 0 and parent != self.base:
                parent = int(self.model.body_parentid[parent])
            if parent == self.base:
                result.add(body)
        return result

    def _configuration(self, value=None, *, check_limits=True):
        if value is None:
            return self.configuration()
        q = np.asarray(value, dtype=float)
        if q.shape != (NQ_PUBLIC,) or not np.all(np.isfinite(q)):
            raise ValueError("configuration must contain 12 finite values")
        norm = np.linalg.norm(q[3:7])
        if norm < 1e-12:
            raise ValueError("configuration quaternion cannot be zero")
        q = q.copy()
        q[3:7] /= norm
        if (check_limits and (np.any(q[7:11] < self.limits.joint_lower)
                              or np.any(q[7:11] > self.limits.joint_upper))):
            raise ValueError("configuration exceeds an arm joint limit")
        if check_limits and not GAP_MIN <= q[11] <= GAP_MAX:
            raise ValueError("configuration gripper gap must be within [0.020, 0.070] m")
        return q

    @staticmethod
    def _velocity(value):
        v = np.asarray(value, dtype=float)
        if v.shape != (NV_PUBLIC,) or not np.all(np.isfinite(v)):
            raise ValueError("velocity must contain 11 finite values")
        return v.copy()

    def configuration(self):
        d = self._live_data()
        bq = self.base_qpos
        left, right = d.qpos[self.grip_qpos]
        return np.r_[S @ d.qpos[bq:bq+3], d.qpos[bq+3:bq+7] * Q_SIGN,
                     d.qpos[self.arm_qpos], GAP_MIN + left + right]

    def velocity(self):
        d = self._live_data()
        bv = self.base_dof
        left, right = d.qvel[self.grip_dof]
        return np.r_[S @ d.qvel[bv:bv+3], S @ d.qvel[bv+3:bv+6],
                     d.qvel[self.arm_dof], left + right]

    def state(self):
        from . import AerialManipulatorState
        d = self._live_data()
        config, velocity = self.configuration(), self.velocity()
        base = np.r_[config[:7], velocity[:6]]
        left, right = d.qpos[self.grip_qpos]
        left_v, right_v = d.qvel[self.grip_dof]
        return AerialManipulatorState(base, config[7:11].copy(), velocity[6:10].copy(),
                                      config[11], velocity[10], float(left-right))

    def _prepare(self, configuration=None, velocity=None, *, check_limits=True):
        live = self._live_data()
        d = self._scratch
        d.qpos[:] = live.qpos
        d.qvel[:] = 0.0
        d.mocap_pos[:] = live.mocap_pos
        d.mocap_quat[:] = live.mocap_quat
        q = self._configuration(configuration, check_limits=check_limits)
        v = np.zeros(NV_PUBLIC) if velocity is None else self._velocity(velocity)
        bq, bv = self.base_qpos, self.base_dof
        d.qpos[bq:bq+3] = S @ q[:3]
        d.qpos[bq+3:bq+7] = q[3:7] * Q_SIGN
        d.qpos[self.arm_qpos] = q[7:11]
        d.qpos[self.grip_qpos] = (q[11]-GAP_MIN)*.5
        d.qvel[bv:bv+3] = S @ v[:3]
        d.qvel[bv+3:bv+6] = S @ v[3:6]
        d.qvel[self.arm_dof] = v[6:10]
        d.qvel[self.grip_dof] = .5*v[10]
        mujoco.mj_forward(self.model, d)
        return d, q, v

    def integrate(self, configuration, tangent_delta):
        q = self._configuration(configuration)
        delta = self._velocity(tangent_delta)
        native_q = np.zeros(self.model.nq)
        native_v = np.zeros(self.model.nv)
        native_q[self.base_qpos:self.base_qpos+3] = S @ q[:3]
        native_q[self.base_qpos+3:self.base_qpos+7] = q[3:7]*Q_SIGN
        native_q[self.arm_qpos] = q[7:11]
        native_q[self.grip_qpos] = .5*(q[11]-GAP_MIN)
        native_v[self.base_dof:self.base_dof+3] = S @ delta[:3]
        native_v[self.base_dof+3:self.base_dof+6] = S @ delta[3:6]
        native_v[self.arm_dof] = delta[6:10]
        native_v[self.grip_dof] = .5*delta[10]
        mujoco.mj_integratePos(self.model, native_q, native_v, 1.0)
        result = np.r_[S @ native_q[self.base_qpos:self.base_qpos+3],
                       native_q[self.base_qpos+3:self.base_qpos+7]*Q_SIGN,
                       native_q[self.arm_qpos], GAP_MIN + native_q[self.grip_qpos].sum()]
        return self._configuration(result)

    def difference(self, configuration_from, configuration_to):
        qa, qb = self._configuration(configuration_from), self._configuration(configuration_to)
        a = np.zeros(self.model.nq); b = np.zeros(self.model.nq)
        a[self.base_qpos:self.base_qpos+3] = S @ qa[:3]
        b[self.base_qpos:self.base_qpos+3] = S @ qb[:3]
        a[self.base_qpos+3:self.base_qpos+7] = qa[3:7]*Q_SIGN
        b[self.base_qpos+3:self.base_qpos+7] = qb[3:7]*Q_SIGN
        a[self.arm_qpos], b[self.arm_qpos] = qa[7:11], qb[7:11]
        a[self.grip_qpos] = .5*(qa[11]-GAP_MIN)
        b[self.grip_qpos] = .5*(qb[11]-GAP_MIN)
        native = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, native, 1.0, a, b)
        return np.r_[S @ native[self.base_dof:self.base_dof+3],
                     S @ native[self.base_dof+3:self.base_dof+6], native[self.arm_dof],
                     native[self.grip_dof].sum()]

    def forward_kinematics(self, configuration=None, frame="tool", *, check_limits=True):
        d, _, _ = self._prepare(configuration, check_limits=check_limits)
        site = self.tool_site if frame == "tool" else self.grasp_site if frame == "grasp" else None
        if site is None:
            raise ValueError("frame must be 'tool' or 'grasp'")
        p = S @ d.site_xpos[site]
        r = S @ d.site_xmat[site].reshape(3, 3) @ S
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, r.ravel())
        return p, quat

    def jacobian(self, configuration=None, frame="tool", *, check_limits=True):
        d, _, _ = self._prepare(configuration, check_limits=check_limits)
        site = self.tool_site if frame == "tool" else self.grasp_site if frame == "grasp" else None
        if site is None:
            raise ValueError("frame must be 'tool' or 'grasp'")
        point = d.site_xpos[site]
        basep = d.xpos[self.base]
        rb = d.xmat[self.base].reshape(3, 3)
        j = np.zeros((6, NV_PUBLIC))
        j[:3, :3] = S @ S
        j[:3, 3:6] = S @ (-_skew(point-basep) @ rb @ S)
        j[3:, 3:6] = S @ rb @ S
        for k, joint in enumerate(self.arm_joints):
            j[:3, 6+k] = S @ np.cross(d.xaxis[joint], point-d.xanchor[joint])
            j[3:, 6+k] = S @ d.xaxis[joint]
        # Gap velocity moves the symmetric fingers; both requested frames are upstream.
        return j

    def point_positions_and_jacobians(
            self, configuration, body_names, local_points, *, check_limits=True):
        """Query NED positions and 3x8 planner-state Jacobians for body-local points.

        Planner state columns are base translation, world yaw, and four arm joints.
        This kinematic-only query leaves live MuJoCo data unchanged and can opt out
        of joint/gap range validation for smooth penalty evaluation.
        """
        names = tuple(body_names)
        points = np.asarray(local_points, dtype=float)
        if points.shape != (len(names), 3) or not np.all(np.isfinite(points)):
            raise ValueError("local_points must be a finite (len(body_names), 3) array")
        try:
            body_ids = np.asarray([self._body_ids[name] for name in names], dtype=int)
        except KeyError as error:
            raise ValueError(f"unknown body name: {error.args[0]}") from error
        d, q, _ = self._prepare(configuration, check_limits=check_limits)
        rotations = d.xmat[body_ids].reshape(-1, 3, 3)
        native_points = d.xpos[body_ids] + np.einsum("nij,nj->ni", rotations, points)
        positions = native_points @ S
        jacobians = np.zeros((len(points), 3, 8))
        jacobians[:, :, :3] = np.eye(3)
        jacobians[:, :, 3] = np.cross(
            np.array([0.0, 0.0, 1.0]), positions-q[:3])
        axes = d.xaxis[self.arm_joints]
        anchors = d.xanchor[self.arm_joints]
        columns = np.cross(axes[None, :, :],
                           native_points[:, None, :]-anchors[None, :, :]) @ S
        columns *= self._body_joint_ancestors[body_ids, :, None]
        jacobians[:, :, 4:] = columns.transpose(0, 2, 1)
        return positions, jacobians

    def point_positions(self, configuration, body_names, local_points, *, check_limits=True):
        """Return NED positions for batches of body-local points without Jacobians."""
        names = tuple(body_names)
        points = np.asarray(local_points, dtype=float)
        if points.shape != (len(names), 3) or not np.all(np.isfinite(points)):
            raise ValueError("local_points must be a finite (len(body_names), 3) array")
        try:
            body_ids = np.asarray([self._body_ids[name] for name in names], dtype=int)
        except KeyError as error:
            raise ValueError(f"unknown body name: {error.args[0]}") from error
        d, _, _ = self._prepare(configuration, check_limits=check_limits)
        rotations = d.xmat[body_ids].reshape(-1, 3, 3)
        return (d.xpos[body_ids] + np.einsum("nij,nj->ni", rotations, points)) @ S

    def _joint_is_ancestor(self, body_id, joint_body_id):
        while body_id > 0:
            if body_id == joint_body_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def collision_geometries(self):
        """Describe active robot collision geoms using public names and local poses."""
        result = []
        for geom_id in self._robot_geoms:
            name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            ancestors = []
            parent = int(self.model.body_parentid[body_id])
            while parent > 0:
                ancestors.append(_name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent))
                parent = int(self.model.body_parentid[parent])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, self.model.geom_quat[geom_id])
            rotation = rotation.reshape(3, 3)
            center = self.model.geom_pos[geom_id].copy()
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_length = float(self.model.geom_size[geom_id, 1])
                result.append({"name": name, "body": body_name, "ancestors": tuple(ancestors),
                               "parent": ancestors[0] if ancestors else None,
                               "contype": int(self.model.geom_contype[geom_id]),
                               "conaffinity": int(self.model.geom_conaffinity[geom_id]),
                               "local_start": center-rotation[:, 2]*half_length,
                               "local_end": center+rotation[:, 2]*half_length,
                               "radius": float(self.model.geom_size[geom_id, 0])})
            else:
                result.append({"name": name, "body": body_name, "ancestors": tuple(ancestors),
                               "parent": ancestors[0] if ancestors else None,
                               "contype": int(self.model.geom_contype[geom_id]),
                               "conaffinity": int(self.model.geom_conaffinity[geom_id]),
                               "local_start": center, "local_end": center,
                               "radius": float(np.linalg.norm(self.model.geom_size[geom_id]))})
        return tuple(result)

    def frame_point(self, frame="grasp"):
        """Return a named frame as a body name and body-local point."""
        site = self.tool_site if frame == "tool" else self.grasp_site if frame == "grasp" else None
        if site is None:
            raise ValueError("frame must be 'tool' or 'grasp'")
        body_id = int(self.model.site_bodyid[site])
        return (_name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                self.model.site_pos[site].copy())

    def mass_properties(self, configuration=None):
        """Return total mass, NED center of mass, and COM inertia in NED axes."""
        d, _, _ = self._prepare(configuration)
        mass = float(self.model.body_subtreemass[self.base])
        com = d.subtree_com[self.base].copy()
        inertia = np.zeros((3, 3))
        for body in self._descendant_bodies:
            m = self.model.body_mass[body]
            if m <= 0:
                continue
            rot = d.ximat[body].reshape(3, 3)
            inertia += rot @ np.diag(self.model.body_inertia[body]) @ rot.T
            offset = d.xipos[body]-com
            inertia += m*((offset@offset)*np.eye(3)-np.outer(offset, offset))
        return {"mass": mass, "center_of_mass": S@com,
                "inertia_com": S@inertia@S}

    def dynamics(self, configuration=None, velocity=None):
        """Return reduced numerical dynamics at a configuration and tangent velocity.

        With no contact forces, ``M @ acceleration + bias = B @ actuator_inputs
        + passive``. ``bias`` is MuJoCo's gravity/Coriolis/centrifugal term;
        contact constraint forces are not part of this output. The two physical
        gripper sliders are reduced to one ideal symmetric gap coordinate.
        """
        d, _, _ = self._prepare(configuration, velocity)
        # Lift reduced tangent velocities into the native 12-DOF model.
        lift = np.zeros((self.model.nv, NV_PUBLIC))
        lift[self.base_dof:self.base_dof+3, :3] = S
        lift[self.base_dof+3:self.base_dof+6, 3:6] = S
        lift[self.arm_dof, 6:10] = np.eye(4)
        lift[self.grip_dof, 10] = .5
        full_mass = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, d, full_mass)
        mass = lift.T @ full_mass @ lift
        bias = lift.T @ d.qfrc_bias
        passive = lift.T @ d.qfrc_passive
        actuation = np.zeros((NV_PUBLIC, 9))
        for k, site in enumerate(self.rotor_sites):
            force = d.site_xmat[site].reshape(3, 3)[:, 2].copy()
            torque = d.xmat[self.base].reshape(3, 3) @ np.array(
                [0., 0., self.rotor_directions[k]*_rotor_kappa(self.model)])
            native = np.zeros(self.model.nv)
            mujoco.mj_applyFT(self.model, d, force, torque, d.site_xpos[site], self.base, native)
            actuation[:, k] = lift.T @ native
        for k, dof in enumerate(self.arm_dof):
            actuation[6+k, 4+k] = 1.0
        actuation[10, 8] = .5
        return {"mass_matrix": mass, "bias_forces": bias, "passive_forces": passive,
                "actuation_matrix": actuation,
                "actuation_inputs": ("rotor_0_N", "rotor_1_N", "rotor_2_N", "rotor_3_N",
                                     "arm_joint_0_Nm", "arm_joint_1_Nm", "arm_joint_2_Nm",
                                     "arm_joint_3_Nm", "left_gripper_servo_N"),
                "gripper_reduction": "ideal symmetric fingers; servo input is left actuator force"}

    def check_collision(self, configuration=None, clearance=0.0, *,
                        self_clearance=None, payload_position_ned=None,
                        payload_radius=None, payload_attached=False):
        """Return signed distances and named pairs for one static configuration.

        This is a point-configuration check; it does not certify the swept volume
        between two configurations.
        """
        clearance = float(clearance)
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("clearance must be finite and non-negative")
        self_clearance = clearance if self_clearance is None else float(self_clearance)
        if not np.isfinite(self_clearance) or self_clearance < 0:
            raise ValueError("self_clearance must be finite and non-negative")
        d, _, _ = self._prepare(configuration)
        pairs, minimum = [], np.inf
        minimum_self = minimum_world = np.inf
        for a, b in self._self_collision_pairs:
            line = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(self.model, d, a, b, 1e6, line))
            minimum = min(minimum, distance)
            minimum_self = min(minimum_self, distance)
            if distance <= self_clearance:
                pairs.append({"geoms": (_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, a),
                                         _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, b)),
                              "distance": distance, "points_ned": (S@line[:3], S@line[3:])})
        for a, b in self._environment_collision_pairs:
            line = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(self.model, d, a, b, 1e6, line))
            minimum = min(minimum, distance)
            minimum_world = min(minimum_world, distance)
            if distance <= clearance:
                pairs.append({"geoms": (_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, a),
                                         _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, b)),
                              "distance": distance, "points_ned": (S@line[:3], S@line[3:])})
        if payload_attached:
            if payload_position_ned is not None:
                raise ValueError("provide either payload_position_ned or payload_attached")
            if self.grasp_site < 0:
                raise ValueError("robot model does not provide grasp_frame")
            payload_position_ned = S@d.site_xpos[self.grasp_site]
        if payload_position_ned is not None:
            if self._payload_mocap < 0 or self._payload_geom < 0:
                raise ValueError("scene does not provide payload_marker and payload_marker_geom")
            payload = np.asarray(payload_position_ned, dtype=float)
            radius = (float(self.model.geom_size[self._payload_geom, 0])
                      if payload_radius is None else float(payload_radius))
            if (payload.shape != (3,) or not np.all(np.isfinite(payload))
                    or not np.isfinite(radius) or radius <= 0):
                raise ValueError("payload position and radius must be finite and valid")
            if not np.isclose(radius, self.model.geom_size[self._payload_geom, 0]):
                raise ValueError("payload radius must match the scene payload marker sphere")
            d.mocap_pos[self._payload_mocap] = S@payload
            mujoco.mj_forward(self.model, d)
            payload_pairs = [(self._payload_geom, geom) for geom in
                             self._robot_geoms+self._environment_geoms
                             if int(self.model.geom_bodyid[geom]) not in self._gripper_body_ids]
            for a, b in payload_pairs:
                line = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(self.model, d, a, b, 1e6, line))
                minimum = min(minimum, distance)
                is_robot = int(self.model.geom_bodyid[b]) in self._descendant_bodies
                threshold = self_clearance if is_robot else clearance
                if is_robot:
                    minimum_self = min(minimum_self, distance)
                else:
                    minimum_world = min(minimum_world, distance)
                if distance <= threshold:
                    pairs.append({"geoms": (_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, a),
                                             _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, b)),
                                  "distance": distance,
                                  "points_ned": (S@line[:3], S@line[3:])})
        pairs.sort(key=lambda x: x["distance"])
        return {"collision": any(p["distance"] <= 0 for p in pairs),
                "minimum_distance": minimum, "minimum_world_distance": minimum_world,
                "minimum_self_distance": minimum_self, "pairs": pairs}

    def _adjacent_geoms(self, ga, gb):
        ba, bb = int(self.model.geom_bodyid[ga]), int(self.model.geom_bodyid[gb])
        if ba == bb:
            return True
        for ancestor, descendant in ((ba, bb), (bb, ba)):
            body = descendant
            for _ in range(2):
                body = int(self.model.body_parentid[body])
                if body == ancestor:
                    return True
                if body == 0:
                    break
        return False

    def _collision_masks_allow(self, ga, gb):
        return bool((self.model.geom_contype[ga] & self.model.geom_conaffinity[gb])
                    or (self.model.geom_contype[gb] & self.model.geom_conaffinity[ga]))


def _id(model, typ, name):
    result = mujoco.mj_name2id(model, typ, name)
    if result < 0:
        raise ValueError(f"model is missing required element '{name}'")
    return result


def _name(model, typ, i):
    return mujoco.mj_id2name(model, typ, i) or f"geom_{i}"


def _skew(x):
    a, b, c = x
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]], float)


def _rotor_kappa(model):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "rotor_drag_to_thrust")
    return float(model.numeric_data[model.numeric_adr[i]])
