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

    def _find_descendants(self):
        result = set()
        for body in range(self.model.nbody):
            parent = body
            while parent > 0 and parent != self.base:
                parent = int(self.model.body_parentid[parent])
            if parent == self.base:
                result.add(body)
        return result

    def _configuration(self, value=None):
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
        if np.any(q[7:11] < self.limits.joint_lower) or np.any(q[7:11] > self.limits.joint_upper):
            raise ValueError("configuration exceeds an arm joint limit")
        if not GAP_MIN <= q[11] <= GAP_MAX:
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

    def _prepare(self, configuration=None, velocity=None):
        live = self._live_data()
        d = self._scratch
        d.qpos[:] = live.qpos
        d.qvel[:] = 0.0
        q = self._configuration(configuration)
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

    def forward_kinematics(self, configuration=None, frame="tool"):
        d, _, _ = self._prepare(configuration)
        site = self.tool_site if frame == "tool" else self.grasp_site if frame == "grasp" else None
        if site is None:
            raise ValueError("frame must be 'tool' or 'grasp'")
        p = S @ d.site_xpos[site]
        r = S @ d.site_xmat[site].reshape(3, 3) @ S
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, r.ravel())
        return p, quat

    def jacobian(self, configuration=None, frame="tool"):
        d, _, _ = self._prepare(configuration)
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

    def check_collision(self, configuration=None, clearance=0.0):
        """Return signed distances and named pairs for one static configuration.

        This is a point-configuration check; it does not certify the swept volume
        between two configurations.
        """
        clearance = float(clearance)
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("clearance must be finite and non-negative")
        d, _, _ = self._prepare(configuration)
        pairs, minimum = [], np.inf
        candidates = [(a, b) for i, a in enumerate(self._robot_geoms)
                      for b in self._robot_geoms[i+1:]
                      if not self._adjacent_geoms(a, b) and self._collision_masks_allow(a, b)]
        candidates += [(a, b) for a in self._robot_geoms for b in self._environment_geoms
                       if self._collision_masks_allow(a, b)]
        for a, b in candidates:
            line = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(self.model, d, a, b, 1e6, line))
            minimum = min(minimum, distance)
            if distance <= clearance:
                pairs.append({"geoms": (_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, a),
                                         _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, b)),
                              "distance": distance, "points_ned": (S@line[:3], S@line[3:])})
        pairs.sort(key=lambda x: x["distance"])
        return {"collision": any(p["distance"] <= 0 for p in pairs),
                "minimum_distance": minimum, "pairs": pairs}

    def _adjacent_geoms(self, ga, gb):
        ba, bb = int(self.model.geom_bodyid[ga]), int(self.model.geom_bodyid[gb])
        return ba == bb or self.model.body_parentid[ba] == bb or self.model.body_parentid[bb] == ba

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
