"""Task-space residuals evaluated through the same flatness map as execution."""

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from .flatness import recover_state
from ..gcopter.mappings import polynomial_basis_matrix, smoothed_l1_array
from ..gcopter.optimization import integrate_samples
from .collision import WholeBodyCollision
from .validation import TrajectoryValidation

try:
    from ...native import _aerial_constraints as _native_aerial
except ImportError:
    _native_aerial = None

@dataclass(frozen=True)
class TaskWaypoint:
    position: np.ndarray
    orientation: np.ndarray | None = None
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    position_mask: np.ndarray | None = None
    orientation_mask: np.ndarray | None = None
    linear_velocity_mask: np.ndarray | None = None
    angular_velocity_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.position is None:
            raise ValueError('task position is required')
        for name, size in (('position', 3), ('orientation', 4),
                           ('linear_velocity', 3), ('angular_velocity', 3)):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.array(value, dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be a finite {size}-vector')
            if name == 'orientation':
                norm = np.linalg.norm(value)
                if norm < 1e-12:
                    raise ValueError('orientation quaternion must be nonzero')
                value /= norm
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            mask_name = name + '_mask'
            mask_value = getattr(self, mask_name)
            mask = np.ones(3, dtype=bool) if mask_value is None else np.array(mask_value, dtype=bool)
            if mask.shape != (3,):
                raise ValueError(f'{mask_name} must contain three selections')
            mask.setflags(write=False)
            object.__setattr__(self, mask_name, mask)


def task_residual(robot, waypoint, derivatives, gripper_opening, gravity=9.81):
    q, velocity = recover_state(derivatives, gripper_opening, gravity)
    position, orientation = robot.forward_kinematics(q, frame='grasp', check_limits=False)
    values = [(position - waypoint.position)[waypoint.position_mask]]
    if waypoint.orientation is not None:
        target = Rotation.from_quat(np.r_[waypoint.orientation[1:], waypoint.orientation[0]])
        # Robot FK returns scalar-first quaternion.
        current = Rotation.from_quat(np.r_[orientation[1:], orientation[0]])
        values.append((target.inv() * current).as_rotvec()[waypoint.orientation_mask])
    twist = None
    if waypoint.linear_velocity is not None or waypoint.angular_velocity is not None:
        twist = robot.jacobian(q, frame='grasp', check_limits=False) @ velocity
    if waypoint.linear_velocity is not None:
        values.append((twist[:3] - waypoint.linear_velocity)[waypoint.linear_velocity_mask])
    if waypoint.angular_velocity is not None:
        values.append((twist[3:] - waypoint.angular_velocity)[waypoint.angular_velocity_mask])
    return np.concatenate(values)


class TaskEventConstraint:
    """Fixed task residual at one knot, with coefficient and duration Jacobians.

    Local robot-map derivatives use central differences; spline/time pullbacks
    are exact. This reference implementation supports arbitrary task twist.
    """

    def __init__(self, robot, waypoint, *, knot, gripper_opening, gravity=9.81):
        self.robot, self.waypoint = robot, waypoint
        self.knot, self.gap, self.gravity = knot, gripper_opening, gravity

    def linearize(self, durations, coefficients):
        from ..gcopter.trajectory import polynomial_basis_matrix
        if not 0 <= self.knot <= len(durations):
            raise ValueError('task knot is out of range')
        piece = max(0, self.knot-1)
        local = 0. if self.knot == 0 else durations[piece]
        bases = np.stack([polynomial_basis_matrix(np.array([local]), d)[0] for d in range(5)])
        derivatives = bases @ coefficients[piece]
        def residual(values):
            return task_residual(self.robot, self.waypoint, values, self.gap, self.gravity)
        values = residual(derivatives[:4])
        jacobian = np.zeros((len(values), 4, 8))
        for order in range(4):
            for dimension in range(8):
                step = 1e-6 * max(1., abs(derivatives[order, dimension]))
                plus, minus = derivatives[:4].copy(), derivatives[:4].copy()
                plus[order, dimension] += step; minus[order, dimension] -= step
                jacobian[:, order, dimension] = (residual(plus)-residual(minus))/(2*step)
        gc = np.zeros((len(values), *coefficients.shape))
        gc[:, piece] = np.einsum('rdc,dk->rkc', jacobian, bases[:4])
        gt = np.zeros((len(values), len(durations)))
        if self.knot:
            gt[:, piece] = np.einsum('rdc,dc->r', jacobian, derivatives[1:5])
        return values, gc, gt


def _flatness_body_rate_squared_python(acceleration, jerk, yaw, yaw_rate, gravity):
    """Return full flatness body-rate norm and its analytic input gradient.

    Forward derivatives are propagated through the normalized thrust axis and
    heading construction; the input order is acceleration, jerk, yaw, yaw rate.
    """
    dimensions = 8

    def vector(values, offset):
        values = np.asarray(values, dtype=float)
        gradient = np.zeros((3, dimensions))
        gradient[np.arange(3), offset+np.arange(3)] = 1.
        return values, gradient

    def constant_vector(values):
        return np.asarray(values, dtype=float), np.zeros((3, dimensions))

    def scalar(value, index=None):
        gradient = np.zeros(dimensions)
        if index is not None:
            gradient[index] = 1.
        return float(value), gradient

    def vscale(value, factor):
        return (value[0]*factor[0],
                value[1]*factor[0]+value[0][:, None]*factor[1])

    def vadd(first, second):
        return first[0]+second[0], first[1]+second[1]

    def vsub(first, second):
        return first[0]-second[0], first[1]-second[1]

    def dot(first, second):
        return (float(np.dot(first[0], second[0])),
                first[1].T@second[0]+second[1].T@first[0])

    def cross(first, second):
        derivative = (np.cross(first[1].T, np.broadcast_to(second[0], (dimensions, 3)))
                      +np.cross(np.broadcast_to(first[0], (dimensions, 3)),
                                second[1].T))
        return np.cross(first[0], second[0]), derivative.T

    def sqrt(value):
        root = np.sqrt(max(value[0], 1e-16))
        return float(root), value[1]/(2*root)

    def multiply(first, second):
        return first[0]*second[0], first[1]*second[0]+second[1]*first[0]

    def divide(first, second):
        return first[0]/second[0], (first[1]*second[0]
                                     -second[1]*first[0])/(second[0]**2)

    acceleration = vector(acceleration, 0)
    jerk = vector(jerk, 3)
    yaw = scalar(yaw, 6)
    yaw_rate = scalar(yaw_rate, 7)
    force = vscale(acceleration, scalar(-1.))
    force = (force[0]+np.array([0., 0., gravity]), force[1])
    force_rate = vscale(jerk, scalar(-1.))
    magnitude = sqrt(dot(force, force))
    body_z = vscale(force, divide(scalar(1.), magnitude))
    force_along_body = dot(body_z, force_rate)
    body_z_rate = vscale(
        vsub(force_rate, vscale(body_z, force_along_body)),
        divide(scalar(1.), magnitude))

    cosine = (np.cos(yaw[0]), -np.sin(yaw[0])*yaw[1])
    sine = (np.sin(yaw[0]), np.cos(yaw[0])*yaw[1])
    heading = (np.array([cosine[0], sine[0], 0.]),
               np.stack((cosine[1], sine[1], np.zeros(dimensions))))
    cross_axis = cross(body_z, heading)
    cross_norm = sqrt(dot(cross_axis, cross_axis))
    cross_norm = (max(cross_norm[0], 1e-8), cross_norm[1])
    body_y = vscale(cross_axis, divide(scalar(1.), cross_norm))
    alignment = dot(body_z, heading)
    yaw_component = divide(
        vsub(scalar(0.), multiply(dot(body_y, body_z_rate), alignment)),
        cross_norm)
    heading_yaw_component = divide(
        multiply((float(body_z[0][2]), body_z[1][2]), yaw_rate),
        multiply(cross_norm, cross_norm))
    axial_rate = (yaw_component[0]+heading_yaw_component[0],
                  yaw_component[1]+heading_yaw_component[1])
    transverse_rate_squared = dot(body_z_rate, body_z_rate)
    axial_rate_squared = multiply(axial_rate, axial_rate)
    result = (transverse_rate_squared[0]+axial_rate_squared[0],
              transverse_rate_squared[1]+axial_rate_squared[1])
    return float(result[0]), result[1]


def _flatness_body_rate_squared(acceleration, jerk, yaw, yaw_rate, gravity):
    if _native_aerial is not None:
        return _native_aerial.flatness_body_rate_squared(
            np.asarray(acceleration, dtype=float), np.asarray(jerk, dtype=float),
            yaw, yaw_rate, gravity)
    return _flatness_body_rate_squared_python(acceleration, jerk, yaw, yaw_rate, gravity)


def _add_penalty(values, jacobian, weight, epsilon):
    costs, derivatives = smoothed_l1_array(np.asarray(values, dtype=float), epsilon)
    return (weight*float(np.sum(costs)),
            weight*derivatives[..., None]*np.asarray(jacobian, dtype=float))


class PhysicalConstraints:
    def _physical_cost_gradient_python(self, sigma, velocity, acceleration, jerk):
        """Reference arithmetic for physical constraints without collision queries."""
        cfg, quad = self.config, self.quad
        gradients = [np.zeros(8) for _ in range(4)]
        cost = 0.0
        max_violation = -np.inf

        def add(values, jac, derivative_order, weight=None):
            nonlocal cost, max_violation
            values = np.asarray(values, dtype=float)
            jac = np.asarray(jac, dtype=float)
            weight = cfg.constraint_weight if weight is None else weight
            part, weighted_jac = _add_penalty(values, jac, weight, cfg.smoothing_epsilon)
            cost += part
            gradients[derivative_order] += np.sum(weighted_jac, axis=tuple(range(values.ndim)))
            max_violation = max(max_violation, float(np.max(values, initial=-np.inf)))

        speed_violation = np.dot(velocity[:3], velocity[:3])-cfg.max_speed**2
        speed_jac = np.zeros((1, 8)); speed_jac[0, :3] = 2*velocity[:3]
        add([speed_violation], speed_jac, 1)
        accel_violation = np.dot(acceleration[:3], acceleration[:3])-cfg.max_acceleration**2
        accel_jac = np.zeros((1, 8)); accel_jac[0, :3] = 2*acceleration[:3]
        add([accel_violation], accel_jac, 2)
        add([velocity[3]**2-cfg.max_yaw_rate**2],
            [np.eye(8)[3]*2*velocity[3]], 1)
        add([acceleration[3]**2-cfg.max_yaw_acceleration**2],
            [np.eye(8)[3]*2*acceleration[3]], 2)

        lower, upper = self.robot.limits.joint_lower, self.robot.limits.joint_upper
        for sign, bound in ((1., upper), (-1., lower)):
            values = sign*sigma[4:8]-sign*bound
            jac = np.zeros((4, 8)); jac[np.arange(4), 4+np.arange(4)] = sign
            add(values, jac, 0)
        joint_v = velocity[4:8]**2-np.asarray(cfg.joint_velocity_limits)**2
        jac = np.zeros((4, 8)); jac[np.arange(4), 4+np.arange(4)] = 2*velocity[4:8]
        add(joint_v, jac, 1)
        joint_a = acceleration[4:8]**2-np.asarray(cfg.joint_acceleration_limits)**2
        jac = np.zeros((4, 8)); jac[np.arange(4), 4+np.arange(4)] = 2*acceleration[4:8]
        add(joint_a, jac, 2)

        gravity = float(quad.g)
        force_direction = np.array([-acceleration[0], -acceleration[1],
                                    gravity-acceleration[2]])
        rho = max(float(np.linalg.norm(force_direction)), 1e-8)
        body_z = force_direction/rho
        thrust = self.mass*rho
        thrust_min, thrust_max = 4*quad.min_thrust, 4*quad.max_thrust
        thrust_grad = -self.mass*body_z
        add([thrust_min-thrust], [np.r_[-thrust_grad, np.zeros(5)]], 2)
        add([thrust-thrust_max], [np.r_[thrust_grad, np.zeros(5)]], 2)
        cos_tilt = float(np.clip(body_z[2], -1., 1.))
        tilt = float(np.arccos(cos_tilt))
        sin_tilt = max(float(np.sqrt(max(1-cos_tilt*cos_tilt, 0.))), 1e-8)
        tilt_grad = (np.array([0., 0., 1.])-cos_tilt*body_z)/(rho*sin_tilt)
        add([tilt-quad.max_tilt_angle], [np.r_[tilt_grad, np.zeros(5)]], 2)
        body_rate2, body_rate_gradient = _flatness_body_rate_squared_python(
            acceleration[:3], jerk[:3], sigma[3], velocity[3], gravity)
        body_rate_violation = body_rate2-cfg.max_body_rate**2
        body_rate_cost, body_rate_derivative = smoothed_l1_array(
            np.array([body_rate_violation]), cfg.smoothing_epsilon)
        cost += cfg.constraint_weight*float(body_rate_cost[0])
        body_rate_gradient *= cfg.constraint_weight*float(body_rate_derivative[0])
        gradients[0][3] += body_rate_gradient[6]
        gradients[1][3] += body_rate_gradient[7]
        gradients[2][:3] += body_rate_gradient[:3]
        gradients[3][:3] += body_rate_gradient[3:6]
        max_violation = max(max_violation, float(body_rate_violation))

        outside_low = self.bounds[0]-sigma[:3]
        outside_high = sigma[:3]-self.bounds[1]
        workspace_values = np.r_[outside_low, outside_high]
        workspace_jac = np.zeros((6, 8))
        workspace_jac[np.arange(3), np.arange(3)] = -1.
        workspace_jac[3+np.arange(3), np.arange(3)] = 1.
        add(workspace_values, workspace_jac, 0)

        return cost, gradients, max_violation

    def _physical_cost_gradient(self, sigma, velocity, acceleration, jerk):
        if _native_aerial is None:
            return self._physical_cost_gradient_python(sigma, velocity, acceleration, jerk)
        cfg, quad = self.config, self.quad
        parameters = np.array([
            cfg.constraint_weight, cfg.smoothing_epsilon, cfg.max_speed,
            cfg.max_acceleration, cfg.max_yaw_rate, cfg.max_yaw_acceleration,
            cfg.max_body_rate, quad.g, self.mass, quad.min_thrust,
            quad.max_thrust, quad.max_tilt_angle], dtype=float)
        return _native_aerial.physical_cost_gradient(
            *(np.asarray(value, dtype=float) for value in (sigma, velocity, acceleration, jerk)),
            np.asarray(self.robot.limits.joint_lower, dtype=float),
            np.asarray(self.robot.limits.joint_upper, dtype=float), self.bounds,
            parameters, np.asarray(cfg.joint_velocity_limits, dtype=float),
            np.asarray(cfg.joint_acceleration_limits, dtype=float))

    def sample_cost_gradient(self, sigma, velocity, acceleration, jerk):
        """Return cost, gradients for orders 0..3, maximum raw violation and clearance."""
        self.objective_samples += 1
        cost, gradients, max_violation = self._physical_cost_gradient(
            sigma, velocity, acceleration, jerk)
        collision_cost, collision_gradient, collision_acceleration_gradient, \
            collision_violation, clearance, self_violation, payload_violation = \
            self.collision_cost_gradient(sigma, acceleration)
        cost += collision_cost
        gradients[0] += collision_gradient
        gradients[2] += collision_acceleration_gradient
        max_violation = max(max_violation, collision_violation)
        max_violation = max(max_violation, self_violation)
        max_violation = max(max_violation, payload_violation)
        return cost, gradients, max_violation, clearance

    def integrated_penalty(self, durations, coefficients):
        if _native_aerial is None:
            return self._integrated_penalty_python(durations, coefficients)
        floor = (self.config.integral_resolution_floor_loaded if self.carry_payload
                 else self.config.integral_resolution_floor_unloaded)
        resolution = max(self.config.integral_resolution, floor)
        self.last_violation = -np.inf
        self.minimum_clearance = np.inf

        def sample(*values):
            result = self.sample_cost_gradient(*values)
            self.last_violation = max(self.last_violation, result[2])
            self.minimum_clearance = min(self.minimum_clearance, result[3])
            return result

        result = _native_aerial.integrated_penalty(
            np.asarray(durations, dtype=float), np.asarray(coefficients, dtype=float),
            resolution, sample)
        self.last_violation, self.minimum_clearance = result[3:]
        return result[:3]

    def _integrated_penalty_python(self, durations, coefficients):
        pieces = len(durations)
        # Preserve the narrow-passage defaults while allowing the wider
        # workcell scene to use fewer optimization quadrature points. Dense
        # full-geometry validation remains independent of these floors.
        floor = (self.config.integral_resolution_floor_loaded if self.carry_payload
                 else self.config.integral_resolution_floor_unloaded)
        resolution = max(self.config.integral_resolution, floor)
        self.last_violation = -np.inf
        self.minimum_clearance = np.inf

        def sample(piece, alpha, values):
            cost, grads, violation, clearance = self.sample_cost_gradient(*values[:4])
            self.last_violation = max(self.last_violation, violation)
            self.minimum_clearance = min(self.minimum_clearance, clearance)
            return cost, grads

        return integrate_samples(durations, coefficients, resolution, sample)


class AerialManipulatorTrajectoryEvaluator(WholeBodyCollision, PhysicalConstraints, TrajectoryValidation):
    """Initialized collision/constraint/validation context for one payload/gripper mode."""
