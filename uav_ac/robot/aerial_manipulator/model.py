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
        # RRT narrow-phase queries repeatedly use these same geometry names
        # and allowed pairs. Resolve the static model metadata once instead
        # of rebuilding dictionaries and calling mj_name2id for each state.
        self._geom_names_by_id = tuple(
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for geom in range(model.ngeom))
        self._geom_ids_by_name = {
            name: geom for geom, name in enumerate(self._geom_names_by_id)
            if name is not None
        }
        self._allowed_collision_pairs = {
            tuple(sorted(pair))
            for pair in self._self_collision_pairs+self._environment_collision_pairs
        }
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
        if self._payload_geom >= 0:
            robot_payload = [g for g in self._robot_geoms
                             if int(model.geom_bodyid[g]) not in self._gripper_body_ids]
            self._payload_collision_pairs = tuple(
                tuple(sorted((self._payload_geom, geom)))
                for geom in robot_payload+self._environment_geoms)
        else:
            self._payload_collision_pairs = ()
        self._allowed_collision_pairs_with_payload = (
            self._allowed_collision_pairs.union(self._payload_collision_pairs))

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

    def _prepare_kinematic(self, configuration=None, *, check_limits=True):
        """Load an isolated configuration and update transforms only.

        Collision point queries need body transforms, not mass matrices,
        constraints, or the rest of MuJoCo's forward dynamics pipeline.
        Keeping this path separate avoids mutating live simulation data and
        skips work that cannot affect forward kinematics.
        """
        live = self._live_data()
        d = self._scratch
        d.qpos[:] = live.qpos
        d.mocap_pos[:] = live.mocap_pos
        d.mocap_quat[:] = live.mocap_quat
        q = self._configuration(configuration, check_limits=check_limits)
        bq = self.base_qpos
        d.qpos[bq:bq+3] = S @ q[:3]
        d.qpos[bq+3:bq+7] = q[3:7] * Q_SIGN
        d.qpos[self.arm_qpos] = q[7:11]
        d.qpos[self.grip_qpos] = (q[11]-GAP_MIN)*.5
        mujoco.mj_kinematics(self.model, d)
        return d, q

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
        d, q = self._prepare_kinematic(configuration, check_limits=check_limits)
        rotations = d.xmat[body_ids].reshape(-1, 3, 3)
        native_points = d.xpos[body_ids] + np.einsum("nij,nj->ni", rotations, points)
        positions = native_points @ S
        jacobians = np.zeros((len(points), 3, 8))
        jacobians[:, :, :3] = np.eye(3)
        jacobians[:, :, 3] = np.cross(
            np.array([0.0, 0.0, 1.0]), positions-q[:3])
        axes = d.xaxis[self.arm_joints]
        anchors = d.xanchor[self.arm_joints]
        columns = _cross3(axes[None, :, :],
                          native_points[:, None, :]-anchors[None, :, :]) @ S
        columns *= self._body_joint_ancestors[body_ids, :, None]
        jacobians[:, :, 4:] = columns.transpose(0, 2, 1)
        return positions, jacobians

    def point_positions_and_pose_jacobians(
            self, configuration, body_names, local_points, *, check_limits=True):
        """Return position Jacobians for translation, world rotation, and joints.

        Columns are ``[position(3), world-angle tangent(3), arm-joints(4)]``.
        The rotation tangent is a left perturbation of the complete base
        orientation, including flatness-recovered roll and pitch.
        """
        names = tuple(body_names)
        points = np.asarray(local_points, dtype=float)
        if points.shape != (len(names), 3) or not np.all(np.isfinite(points)):
            raise ValueError("local_points must be a finite (len(body_names), 3) array")
        try:
            body_ids = np.asarray([self._body_ids[name] for name in names], dtype=int)
        except KeyError as error:
            raise ValueError(f"unknown body name: {error.args[0]}") from error
        d, q = self._prepare_kinematic(configuration, check_limits=check_limits)
        rotations = d.xmat[body_ids].reshape(-1, 3, 3)
        native_points = d.xpos[body_ids]+np.einsum("nij,nj->ni", rotations, points)
        positions = native_points @ S
        jacobians = np.zeros((len(points), 3, 10))
        jacobians[:, :, :3] = np.eye(3)
        relative = positions-q[:3]
        jacobians[:, :, 3:6] = np.asarray([-_skew(value) for value in relative])
        axes, anchors = d.xaxis[self.arm_joints], d.xanchor[self.arm_joints]
        joint_columns = _cross3(
            axes[None, :, :], native_points[:, None, :]-anchors[None, :, :]) @ S
        joint_columns *= self._body_joint_ancestors[body_ids, :, None]
        jacobians[:, :, 6:10] = joint_columns.transpose(0, 2, 1)
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
        d, _ = self._prepare_kinematic(configuration, check_limits=check_limits)
        rotations = d.xmat[body_ids].reshape(-1, 3, 3)
        return (d.xpos[body_ids] + np.einsum("nij,nj->ni", rotations, points)) @ S

    def point_positions_batch(self, configurations, body_names, local_points,
                              *, check_limits=True):
        """Return ``(B, S, 3)`` NED positions using kinematics-only updates.

        Body names and local points are resolved once per batch. Each state is
        evaluated in the model's independent scratch ``MjData`` object with
        ``mj_kinematics``; the live simulation state is never changed.
        """
        configurations = np.asarray(configurations, dtype=float)
        names = tuple(body_names)
        points = np.asarray(local_points, dtype=float)
        if (configurations.ndim != 2 or configurations.shape[1] != NQ_PUBLIC
                or not np.all(np.isfinite(configurations))):
            raise ValueError("configurations must be a finite (B, 12) array")
        if points.shape != (len(names), 3) or not np.all(np.isfinite(points)):
            raise ValueError("local_points must be a finite (len(body_names), 3) array")
        try:
            body_ids = np.asarray([self._body_ids[name] for name in names], dtype=int)
        except KeyError as error:
            raise ValueError(f"unknown body name: {error.args[0]}") from error
        result = np.empty((len(configurations), len(names), 3), dtype=float)
        d = self._scratch
        for index, configuration in enumerate(configurations):
            d, _ = self._prepare_kinematic(configuration, check_limits=check_limits)
            rotations = d.xmat[body_ids].reshape(-1, 3, 3)
            native = d.xpos[body_ids] + np.einsum("nij,nj->ni", rotations, points)
            result[index] = native @ S
        return result

    def exact_collision_distances(
            self, configuration=None, pairs=None, *, payload_attached=False,
            with_jacobians=False, with_pose_jacobians=False,
            jacobian_distance_thresholds=None, check_limits=True):
        """Batch MuJoCo signed distances for configured robot geometry pairs.

        ``pairs`` contains public geom-name pairs.  By default all configured
        non-adjacent self pairs and robot/environment pairs are queried.  A
        Payload pairs are enabled when ``payload_attached`` is true. Jacobians
        are gradients of the signed distance with respect to
        ``[position(3), yaw, arm_joints(4)]``
        by default. ``with_pose_jacobians`` selects the full-pose tangent
        ``[position(3), world-angle tangent(3), arm-joints(4)]``. Both use
        MuJoCo's closest-point witnesses and public point Jacobians. All work uses the
        scratch data object and leaves the live simulation untouched.
        """
        # ``mj_geomDistance`` needs current geometry transforms, not dynamics
        # products such as the mass matrix or constraint Jacobians.
        d, q = self._prepare_kinematic(configuration, check_limits=check_limits)
        payload_id = int(self._payload_geom)
        payload_enabled = payload_attached
        if payload_enabled:
            if payload_id < 0 or self._payload_mocap < 0:
                raise ValueError("scene does not provide payload_marker and payload_marker_geom")
            if self.grasp_site < 0:
                raise ValueError("robot model does not provide grasp_frame")
            payload_ned = S @ d.site_xpos[self.grasp_site]
            # The attached payload follows the grasp frame in the isolated
            # mocap state used by this distance query.
            d.mocap_pos[self._payload_mocap] = S @ payload_ned
            mujoco.mj_kinematics(self.model, d)

        allowed = (self._allowed_collision_pairs_with_payload if payload_enabled
                   else self._allowed_collision_pairs)

        if pairs is None:
            selected = list(allowed)
        else:
            selected = []
            for pair in pairs:
                if len(pair) != 2:
                    raise ValueError("collision pairs must contain two geom names")
                if not all(isinstance(name, str) for name in pair):
                    raise ValueError("collision pair names must be strings")
                ids = tuple(sorted((self._geom_ids_by_name.get(pair[0], -1),
                                    self._geom_ids_by_name.get(pair[1], -1))))
                if -1 in ids or ids not in allowed:
                    raise ValueError(f"collision pair is not an allowed configured pair: {pair}")
                selected.append(ids)

        distances = np.empty(len(selected), dtype=float)
        points = np.empty((len(selected), 2, 3), dtype=float)
        jacobian_width = 10 if with_pose_jacobians else 8
        jacobians = (np.zeros((len(selected), jacobian_width), dtype=float)
                     if with_jacobians and jacobian_distance_thresholds is not None else
                     np.empty((len(selected), jacobian_width), dtype=float)
                     if with_jacobians else None)

        def body_point_jacobian(body_id, native_point):
            """Differentiate a MuJoCo witness point using the prepared state.

            Calling ``point_positions_and_jacobians`` here would prepare and
            forward the scratch model once per witness point.  The exact
            distance query already has the same scratch state, so reproduce
            that public Jacobian formula directly and keep one forward pass
            per batch of collision pairs.
            """
            positions = S @ native_point
            jacobian = np.zeros((3, jacobian_width))
            jacobian[:, :3] = np.eye(3)
            if with_pose_jacobians:
                jacobian[:, 3:6] = -_skew(positions-q[:3])
                joint_offset = 6
            else:
                jacobian[:, 3] = np.cross(
                    np.array([0.0, 0.0, 1.0]), positions-q[:3])
                joint_offset = 4
            axes = d.xaxis[self.arm_joints]
            offsets = native_point-d.xanchor[self.arm_joints]
            columns = np.empty_like(axes)
            columns[:, 0] = axes[:, 1]*offsets[:, 2]-axes[:, 2]*offsets[:, 1]
            columns[:, 1] = axes[:, 2]*offsets[:, 0]-axes[:, 0]*offsets[:, 2]
            columns[:, 2] = axes[:, 0]*offsets[:, 1]-axes[:, 1]*offsets[:, 0]
            columns = columns @ S
            columns *= self._body_joint_ancestors[body_id, :, None]
            jacobian[:, joint_offset:] = columns.T
            return jacobian

        def point_jacobian(geom_id, native_point):
            return body_point_jacobian(
                int(self.model.geom_bodyid[geom_id]), native_point)

        grasp_jacobian = None
        if with_jacobians and payload_enabled and payload_attached:
            body, local = self.frame_point("grasp")
            body_id = self._body_ids[body]
            rotation = d.xmat[body_id].reshape(3, 3)
            grasp_native = d.xpos[body_id]+rotation@local
            grasp_jacobian = body_point_jacobian(body_id, grasp_native)

        for index, (first, second) in enumerate(selected):
            line = np.zeros(6)
            distances[index] = float(mujoco.mj_geomDistance(
                self.model, d, first, second, 1e6, line))
            native_points = (line[:3].copy(), line[3:].copy())
            points[index] = np.stack((S@native_points[0], S@native_points[1]))
            if with_jacobians:
                pair_names = frozenset((self._geom_names_by_id[first],
                                        self._geom_names_by_id[second]))
                threshold = (None if jacobian_distance_thresholds is None else
                             jacobian_distance_thresholds.get(pair_names))
                # Smooth collision penalties have zero derivative once the
                # signed distance reaches the required clearance. Avoid
                # building witness-point Jacobians for those inactive pairs.
                if (threshold is not None
                        and distances[index] >= float(threshold)):
                    continue
                gradients = []
                for geom_id, native_point in zip((first, second), native_points):
                    if geom_id == payload_id:
                        if payload_attached:
                            gradients.append(grasp_jacobian[:3])
                        else:
                            gradients.append(np.zeros((3, jacobian_width)))
                    elif geom_id in self._robot_geoms:
                        gradients.append(point_jacobian(geom_id, native_point))
                    else:
                        gradients.append(np.zeros((3, jacobian_width)))
                delta = points[index, 0]-points[index, 1]
                norm = max(float(np.linalg.norm(delta)), 1e-10)
                jacobians[index] = (delta/norm) @ (gradients[0]-gradients[1])
        result = {
            "pairs": tuple((self._geom_names_by_id[first],
                             self._geom_names_by_id[second])
                            for first, second in selected),
            "distances": distances,
            "points_ned": points,
        }
        if jacobians is not None:
            result["jacobians"] = jacobians
        return result

    def collision_pairs(self, kind="self"):
        """Return configured collision geom-name pairs for planner narrow phases."""
        if kind == "self":
            pairs = self._self_collision_pairs
        elif kind == "world":
            pairs = self._environment_collision_pairs
        else:
            raise ValueError("collision pair kind must be 'self' or 'world'")
        return tuple((_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, first),
                      _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, second))
                     for first, second in pairs)

    def collision_environment_aabbs(self):
        """Return conservative NED AABBs for active static environment geoms.

        A horizontal plane is represented by zero thickness along its normal
        and infinite extent in tangent directions. Unknown geom types return
        an unbounded box so broad-phase filtering stays conservative.
        """
        data = self._live_data()
        bounds = {}
        for geom_id in self._environment_geoms:
            name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            center = S @ data.geom_xpos[geom_id]
            rotation = (S @ data.geom_xmat[geom_id].reshape(3, 3) @ S)
            geom_type = self.model.geom_type[geom_id]
            size = self.model.geom_size[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_extent = np.abs(rotation) @ size[:3]
                lower, upper = center-half_extent, center+half_extent
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                half_extent = np.full(3, size[0])
                lower, upper = center-half_extent, center+half_extent
            elif geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE,
                               mujoco.mjtGeom.mjGEOM_CYLINDER):
                half_extent = size[0]+size[1]*np.abs(rotation[:, 2])
                lower, upper = center-half_extent, center+half_extent
            elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                normal = rotation[:, 2]
                if np.max(np.abs(normal[:2])) <= 1e-8:
                    lower = np.array([-np.inf, -np.inf, center[2]])
                    upper = np.array([np.inf, np.inf, center[2]])
                else:
                    lower, upper = np.full(3, -np.inf), np.full(3, np.inf)
            else:
                lower, upper = np.full(3, -np.inf), np.full(3, np.inf)
            bounds[name] = np.stack((lower, upper))
        return bounds

    def base_inscribed_collision_radius(self):
        """Radius of a centered sphere contained inside the base collision box."""
        candidates = []
        for geom_id in self._robot_geoms:
            if (int(self.model.geom_bodyid[geom_id]) == self.base
                    and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                    and np.linalg.norm(self.model.geom_pos[geom_id]) <= 1e-12):
                candidates.append(float(np.min(self.model.geom_size[geom_id, :3])))
        if not candidates:
            return 0.0
        return max(candidates)

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

    @property
    def payload_radius(self):
        """Radius of the scene's payload sphere, in meters."""
        geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "payload_marker_geom")
        if geom < 0 or self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError("aerial manipulator scene requires spherical payload_marker_geom")
        return float(self.model.geom_size[geom, 0])

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
                        self_clearance=None, payload_attached=False):
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
        result = self.exact_collision_distances(
            configuration, payload_attached=payload_attached)
        distances = result["distances"]
        pairs = result["pairs"]
        world_pair_names = self.collision_pairs("world")
        self_pair_names = self.collision_pairs("self")
        world_pairs = {frozenset(pair) for pair in world_pair_names}
        pair_order = {
            **{frozenset(pair): pair for pair in world_pair_names},
            **{frozenset(pair): pair for pair in self_pair_names},
        }
        robot_names = {
            _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for geom in self._robot_geoms
        }
        world_mask = np.asarray(
            [frozenset(pair) in world_pairs for pair in pairs], dtype=bool)
        payload_mask = np.asarray(
            ["payload_marker_geom" in pair for pair in pairs], dtype=bool)
        thresholds = np.where(world_mask, clearance, self_clearance)
        for index, pair in enumerate(pairs):
            if payload_mask[index]:
                other = pair[1] if pair[0] == "payload_marker_geom" else pair[0]
                is_robot = other in robot_names
                world_mask[index] = not is_robot
                thresholds[index] = self_clearance if is_robot else clearance
        hit = distances <= thresholds
        selected = []
        for index, (pair, distance, points) in enumerate(
                zip(pairs, distances, result["points_ned"])):
            if not hit[index]:
                continue
            if payload_mask[index]:
                other = pair[1] if pair[0] == "payload_marker_geom" else pair[0]
                ordered_pair = ("payload_marker_geom", other)
            else:
                ordered_pair = pair_order.get(frozenset(pair), pair)
            if ordered_pair != pair:
                points = points[::-1]
            selected.append({
                "geoms": ordered_pair, "distance": float(distance),
                "points_ned": tuple(points),
            })
        selected.sort(key=lambda item: item["distance"])
        self_mask = ~world_mask
        return {
            "collision": bool(np.any(distances <= 0.0)),
            "minimum_distance": float(np.min(distances, initial=np.inf)),
            "minimum_world_distance": float(np.min(distances[world_mask], initial=np.inf)),
            "minimum_self_distance": float(np.min(distances[self_mask], initial=np.inf)),
            "pairs": selected,
        }

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


def _cross3(first, second):
    """Low-overhead vectorized cross product for arrays ending in 3."""
    first, second = np.asarray(first), np.asarray(second)
    result = np.empty(np.broadcast_shapes(first.shape, second.shape), dtype=float)
    result[..., 0] = first[..., 1]*second[..., 2]-first[..., 2]*second[..., 1]
    result[..., 1] = first[..., 2]*second[..., 0]-first[..., 0]*second[..., 2]
    result[..., 2] = first[..., 0]*second[..., 1]-first[..., 1]*second[..., 0]
    return result


def _rotor_kappa(model):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "rotor_drag_to_thrust")
    return float(model.numeric_data[model.numeric_adr[i]])
