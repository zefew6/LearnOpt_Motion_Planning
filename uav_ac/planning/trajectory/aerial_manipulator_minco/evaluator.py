"""Analytic sampled penalties for whole-body 8-D quintic trajectories."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..gcopter.mappings import polynomial_basis_matrix, smoothed_l1_array
from .types import _flatness_attitude


def _flatness_body_rate_squared(acceleration, jerk, yaw, yaw_rate, gravity):
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


def _add_penalty(values, jacobian, weight, epsilon):
    costs, derivatives = smoothed_l1_array(np.asarray(values, dtype=float), epsilon)
    return (weight*float(np.sum(costs)),
            weight*derivatives[..., None]*np.asarray(jacobian, dtype=float))


class AerialManipulatorTrajectoryEvaluator:
    def __init__(self, robot, esdf, quad, config, workspace_bounds,
                 gripper_opening, carry_payload, deadline=None):
        self.robot, self.esdf, self.quad, self.config = robot, esdf, quad, config
        self.deadline = deadline
        self.objective_samples = 0
        self.validation_samples = 0
        self.mass = robot.mass
        self.bounds = np.asarray(workspace_bounds, dtype=float)
        self.gripper_opening = float(gripper_opening)
        self.carry_payload = bool(carry_payload)
        self.geometry = []
        geometry_spheres = []
        geoms = robot.collision_geometries()
        for geom_index, geom in enumerate(geoms):
            start, end = geom["local_start"], geom["local_end"]
            half_length = .5*float(np.linalg.norm(end-start))
            count = 3 if half_length > 1e-9 else 1
            axial_half_spacing = (half_length/(count-1) if count > 1 else 0.)
            inflated_radius = float(np.hypot(geom["radius"], axial_half_spacing))
            for alpha in np.linspace(0., 1., count):
                center = start+alpha*(end-start)
                sphere_index = len(self.geometry)
                self.geometry.append((geom["body"], center, inflated_radius, geom_index))
                geometry_spheres.append(sphere_index)
        self.self_pairs = []
        for first in range(len(geoms)):
            for second in range(first+1, len(geoms)):
                a, b = geoms[first], geoms[second]
                if (a["body"] == b["body"]
                        or a["body"] in b["ancestors"][:2]
                        or b["body"] in a["ancestors"][:2]):
                    continue
                if not ((a["contype"] & b["conaffinity"])
                        or (b["contype"] & a["conaffinity"])):
                    continue
                self.self_pairs.extend((i, j) for i in geometry_spheres
                                       if self.geometry[i][3] == first
                                       for j in geometry_spheres
                                       if self.geometry[j][3] == second)
        self.sphere_names = tuple(sphere[0] for sphere in self.geometry)
        self.sphere_points = np.asarray([sphere[1] for sphere in self.geometry], dtype=float)
        self.sphere_radii = np.asarray([sphere[2] for sphere in self.geometry], dtype=float)
        self.self_pairs = np.asarray(self.self_pairs, dtype=int).reshape(-1, 2)
        self.payload_body, self.payload_local = robot.frame_point("grasp")
        gripper_bodies = {"gripper_palm_body", "gripper_left_body", "gripper_right_body"}
        payload_pairs = ([] if not self.carry_payload else [
            (sphere, len(self.geometry)) for sphere in geometry_spheres
            if self.geometry[sphere][0] not in gripper_bodies
        ])
        self.payload_pairs = np.asarray(payload_pairs, dtype=int).reshape(-1, 2)
        if self.carry_payload:
            self.sphere_names += (self.payload_body,)
            self.sphere_points = np.vstack((self.sphere_points, self.payload_local))
            self.sphere_radii = np.r_[self.sphere_radii, config.payload_radius]

    def _check_deadline(self):
        if self.deadline is not None:
            import time
            if time.perf_counter() >= self.deadline:
                raise TimeoutError("aerial_manipulator_minco planning budget exceeded")

    def _configuration(self, sigma, acceleration=None):
        from .task_targets import yaw_quaternion
        quaternion = yaw_quaternion(sigma[3])
        if acceleration is not None:
            rotation, _ = _flatness_attitude(
                acceleration[:3], np.zeros(3), sigma[3], 0., self.quad.g)
            xyzw = Rotation.from_matrix(rotation).as_quat()
            quaternion = np.r_[xyzw[3], xyzw[:3]]
        return np.r_[sigma[:3], quaternion, sigma[4:8], self.gripper_opening]

    def _state_geometry(self, sigma):
        positions, jacobians = self.robot.point_positions_and_jacobians(
            self._configuration(sigma), self.sphere_names, self.sphere_points,
            check_limits=False)
        return positions, jacobians, self.sphere_radii

    def collision_feasible(self, sigma):
        """Fast RRT collision proxy query; no point Jacobians or penalties."""
        positions = self.robot.point_positions(
            self._configuration(sigma), self.sphere_names, self.sphere_points,
            check_limits=False)
        lower, upper = self.esdf.origin, self.esdf.upper
        clipped = np.clip(positions, lower, upper)
        distances = self.esdf.distance(clipped)
        outside = np.linalg.norm(positions-clipped, axis=1)
        signed = distances-outside
        lever_arms = np.linalg.norm(positions-np.asarray(sigma)[:3], axis=1)
        acceleration_tilt = np.arctan2(
            self.config.max_acceleration,
            max(self.quad.g-self.config.max_acceleration, 1e-6))
        yaw_only_tilt_bound = min(float(self.quad.max_tilt_angle), acceleration_tilt)
        tilt_margin = 2*lever_arms*np.sin(.5*yaw_only_tilt_bound)
        violations = (self.sphere_radii+self.config.obstacle_clearance
                      +self.config.esdf_interpolation_margin
                      +tilt_margin-signed)
        outside_mask = outside > 0.
        violations[outside_mask] = np.maximum(
            violations[outside_mask], self.config.obstacle_clearance
            +self.config.esdf_interpolation_margin+outside[outside_mask])
        if len(self.payload_pairs):
            first, second = self.payload_pairs.T
            pair_distance = np.linalg.norm(positions[first]-positions[second], axis=1)
            payload_violation = (self.sphere_radii[first]+self.sphere_radii[second]
                                 +self.config.self_clearance-pair_distance)
        else:
            payload_violation = np.array([-np.inf])
        return (float(np.max(violations, initial=-np.inf)),
                float(np.max(payload_violation, initial=-np.inf)))

    def collision_cost_gradient(self, sigma, acceleration=None):
        positions, jacobians, radii = self._state_geometry(sigma)
        acceleration = (np.zeros(8) if acceleration is None
                       else np.asarray(acceleration, dtype=float))
        force = np.array([-acceleration[0], -acceleration[1],
                          self.quad.g-acceleration[2]])
        force_norm = max(float(np.linalg.norm(force)), 1e-8)
        body_z = force/force_norm
        cos_tilt = float(np.clip(body_z[2], -1., 1.))
        tilt = float(np.arccos(cos_tilt))
        sin_tilt = max(float(np.sqrt(max(1-cos_tilt*cos_tilt, 0.))), 1e-8)
        tilt_gradient = (np.array([0., 0., 1.])-cos_tilt*body_z)/(force_norm*sin_tilt)
        relative_jacobians = jacobians.copy()
        relative_jacobians[:, :, :3] -= np.eye(3)[None, :, :]
        relative_positions = positions-sigma[:3]
        lever_arms = np.linalg.norm(relative_positions, axis=1)
        lever_directions = np.zeros_like(relative_positions)
        nonzero_levers = lever_arms > 1e-10
        lever_directions[nonzero_levers] = (
            relative_positions[nonzero_levers]/lever_arms[nonzero_levers, None])
        lever_state_gradients = np.einsum(
            "ni,nij->nj", lever_directions, relative_jacobians)
        tilt_uncertainty = 2*lever_arms*np.sin(.5*tilt)
        tilt_state_gradient = 2*np.sin(.5*tilt)*lever_state_gradients
        tilt_acceleration_gradient = np.zeros((len(positions), 8))
        tilt_acceleration_gradient[:, :3] = (
            lever_arms[:, None]*np.cos(.5*tilt)*tilt_gradient[None, :])
        lower, upper = self.esdf.origin, self.esdf.upper
        clipped = np.clip(positions, lower, upper)
        distances, gradients = self.esdf.distance_and_gradient(clipped)
        outside_delta = positions-clipped
        outside_norm = np.linalg.norm(outside_delta, axis=1)
        outside = outside_norm > 0.0
        signed_gradient = gradients.copy()
        signed_gradient[outside] -= outside_delta[outside]/outside_norm[outside, None]
        signed = distances-outside_norm
        violations = (radii+self.config.obstacle_clearance
                      +self.config.esdf_interpolation_margin
                      +tilt_uncertainty-signed)
        boundary_violation = (self.config.obstacle_clearance
                              +self.config.esdf_interpolation_margin+outside_norm)
        boundary_active = outside & (boundary_violation > violations)
        violations[outside] = np.maximum(violations[outside], boundary_violation[outside])
        costs, derivatives = smoothed_l1_array(violations, self.config.smoothing_epsilon)
        grad_violation = -np.einsum("nci,nc->ni", jacobians, signed_gradient)
        grad_violation += tilt_state_gradient
        grad_violation[boundary_active] = np.einsum(
            "nci,nc->ni", jacobians[boundary_active],
            outside_delta[boundary_active]/outside_norm[boundary_active, None])
        grad = self.config.obstacle_weight*np.sum(
            derivatives[:, None]*grad_violation, axis=0)
        grad_acceleration = self.config.obstacle_weight*np.sum(
            derivatives[:, None]*tilt_acceleration_gradient, axis=0)
        cost = self.config.obstacle_weight*float(np.sum(costs))
        obstacle_violation = float(np.max(violations, initial=-np.inf))
        minimum_clearance = float(np.min(signed-radii-tilt_uncertainty))
        self_violation = -np.inf
        payload_violation = -np.inf
        for pairs, kind in ((self.self_pairs, "self"), (self.payload_pairs, "payload")):
            if not len(pairs):
                continue
            first, second = pairs.T
            delta = positions[first]-positions[second]
            distances = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
            pair_violations = (radii[first]+radii[second]+self.config.self_clearance
                               -distances)
            pair_cost, pair_derivative = smoothed_l1_array(
                pair_violations, self.config.smoothing_epsilon)
            directions = delta/distances[:, None]
            pair_jacobians = jacobians[first]-jacobians[second]
            pair_gradients = -np.einsum("nci,nc->ni", pair_jacobians, directions)
            cost += self.config.self_collision_weight*float(np.sum(pair_cost))
            grad += self.config.self_collision_weight*np.einsum(
                "n,ni->i", pair_derivative, pair_gradients)
            if kind == "self":
                self_violation = float(np.max(pair_violations, initial=-np.inf))
            else:
                payload_violation = float(np.max(pair_violations, initial=-np.inf))
                minimum_clearance = min(
                    minimum_clearance,
                    float(np.min(distances-radii[first]-radii[second])))
        return (cost, grad, grad_acceleration, obstacle_violation,
                minimum_clearance, self_violation, payload_violation)

    def sample_cost_gradient(self, sigma, velocity, acceleration, jerk):
        """Return cost, gradients for orders 0..3, maximum raw violation and clearance."""
        self._check_deadline()
        self.objective_samples += 1
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
        body_rate2, body_rate_gradient = _flatness_body_rate_squared(
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

        collision_cost, collision_gradient, collision_acceleration_gradient, \
            collision_violation, clearance, _, payload_violation = \
            self.collision_cost_gradient(sigma, acceleration)
        cost += collision_cost
        gradients[0] += collision_gradient
        gradients[2] += collision_acceleration_gradient
        max_violation = max(max_violation, collision_violation)
        max_violation = max(max_violation, payload_violation)
        return cost, gradients, max_violation, clearance

    def integrated_penalty(self, durations, coefficients):
        pieces = len(durations)
        resolution = self.config.integral_resolution
        alpha = np.linspace(0., 1., resolution+1)
        quadrature = np.ones(resolution+1); quadrature[[0, -1]] = .5
        grad_coefficients = np.zeros_like(coefficients)
        grad_times = np.zeros(pieces)
        total_cost = 0.0
        self.last_violation = -np.inf
        self.minimum_clearance = np.inf
        for piece, duration in enumerate(durations):
            local_times = alpha*duration
            bases = [polynomial_basis_matrix(local_times, derivative)
                     for derivative in range(5)]
            values = [basis@coefficients[piece] for basis in bases]
            scale = duration/resolution*quadrature
            for sample in range(resolution+1):
                cost, grads, violation, clearance = self.sample_cost_gradient(
                    *(value[sample] for value in values[:4]))
                total_cost += scale[sample]*cost
                for order in range(4):
                    grad_coefficients[piece] += (
                        scale[sample]*np.outer(bases[order][sample], grads[order]))
                derivative_cost_time = (np.dot(grads[0], values[1][sample])
                                        + np.dot(grads[1], values[2][sample])
                                        + np.dot(grads[2], values[3][sample])
                                        + np.dot(grads[3], values[4][sample]))
                grad_times[piece] += (
                    scale[sample]*alpha[sample]*derivative_cost_time
                    + quadrature[sample]/resolution*cost)
                self.last_violation = max(self.last_violation, violation)
                self.minimum_clearance = min(self.minimum_clearance, clearance)
        return total_cost, grad_coefficients, grad_times

    def dense_validate(self, trajectory):
        max_violation = -np.inf
        minimum_clearance = np.inf
        collision = False
        map_inside = True
        maximum_dt = 0.0
        maximum_kind = "none"
        minimum_world_distance = np.inf
        minimum_self_distance = np.inf
        for piece, duration in enumerate(trajectory.durations):
            coefficient = trajectory.coefficients[piece]

            def evaluate(local):
                basis = [polynomial_basis_matrix(np.array([local]), order)[0]
                         for order in range(4)]
                return [row@coefficient for row in basis]

            count = max(2, int(np.ceil(duration/self.config.validation_dt)))
            coarse = np.linspace(0., duration, count+1)
            states = [evaluate(local) for local in coarse]
            accepted = [states[0]]
            intervals = [(coarse[i], coarse[i+1], states[i], states[i+1], 0)
                         for i in range(count)]
            refined = []
            while intervals:
                left, right, state_left, state_right, depth = intervals.pop()
                middle = .5*(left+right)
                state_middle = evaluate(middle)
                halves = ((state_left[0], state_middle[0]),
                          (state_middle[0], state_right[0]))
                position_change = max(np.linalg.norm(b[:3]-a[:3]) for a, b in halves)
                yaw_change = max(abs(float(b[3]-a[3])) for a, b in halves)
                joint_change = max(float(np.max(np.abs(b[4:8]-a[4:8])))
                                   for a, b in halves)
                if (depth < 8 and (position_change > .04 or yaw_change > .08
                                   or joint_change > .08)):
                    intervals.append((middle, right, state_middle, state_right, depth+1))
                    intervals.append((left, middle, state_left, state_middle, depth+1))
                else:
                    refined.append((right, state_right, right-left))
            refined.sort(key=lambda item: item[0])
            accepted.extend(state for _, state, _ in refined)
            maximum_dt = max(maximum_dt, max((dt for _, _, dt in refined), default=0.))
            for sigma, velocity, acceleration, jerk in accepted:
                self._check_deadline()
                self.validation_samples += 1
                gravity = float(self.quad.g)
                force = np.array([-acceleration[0], -acceleration[1],
                                  gravity-acceleration[2]])
                rho = max(float(np.linalg.norm(force)), 1e-8)
                body_rate = _flatness_attitude(
                    acceleration[:3], jerk[:3], sigma[3], velocity[3], gravity)[1]
                lower, upper = self.robot.limits.joint_lower, self.robot.limits.joint_upper
                constraint_names = [
                    "linear_speed", "linear_acceleration", "yaw_rate",
                    "yaw_acceleration", "joint_upper", "joint_lower",
                    "joint_velocity", "joint_acceleration", "maximum_thrust",
                    "minimum_thrust", "tilt", "body_rate", "workspace_lower",
                    "workspace_upper",
                ]
                constraints = [
                    np.dot(velocity[:3], velocity[:3])-self.config.max_speed**2,
                    np.dot(acceleration[:3], acceleration[:3])-self.config.max_acceleration**2,
                    velocity[3]**2-self.config.max_yaw_rate**2,
                    acceleration[3]**2-self.config.max_yaw_acceleration**2,
                    float(np.max(sigma[4:8]-upper)),
                    float(np.max(lower-sigma[4:8])),
                    float(np.max(velocity[4:8]**2-
                                 np.asarray(self.config.joint_velocity_limits)**2)),
                    float(np.max(acceleration[4:8]**2-
                                 np.asarray(self.config.joint_acceleration_limits)**2)),
                    self.mass*rho-4*self.quad.max_thrust,
                    4*self.quad.min_thrust-self.mass*rho,
                    float(np.arccos(np.clip(force[2]/rho, -1., 1.))
                          -self.quad.max_tilt_angle),
                    float(np.dot(body_rate, body_rate)-self.config.max_body_rate**2),
                    float(np.max(self.bounds[0]-sigma[:3])),
                    float(np.max(sigma[:3]-self.bounds[1])),
                ]
                configuration = self._configuration(sigma, acceleration)
                try:
                    positions = self.robot.point_positions(
                        configuration, self.sphere_names, self.sphere_points)
                    outside = np.maximum(self.esdf.origin-positions, 0.)+np.maximum(
                        positions-self.esdf.upper, 0.)
                    outside_distance = float(np.max(np.linalg.norm(outside, axis=1), initial=0.))
                    map_inside &= outside_distance <= 1e-12
                    constraint_names.append("esdf_bounds")
                    constraints.append(outside_distance)
                    exact = self.robot.check_collision(
                        configuration, clearance=self.config.obstacle_clearance,
                        self_clearance=self.config.self_clearance,
                        payload_attached=self.carry_payload)
                    world_clearance = exact["minimum_world_distance"]
                    self_clearance = exact["minimum_self_distance"]
                    minimum_world_distance = min(minimum_world_distance, world_clearance)
                    minimum_self_distance = min(minimum_self_distance, self_clearance)
                    if np.isfinite(world_clearance):
                        constraint_names.append("world_clearance")
                        constraints.append(self.config.obstacle_clearance-world_clearance)
                    if np.isfinite(self_clearance):
                        constraint_names.append("self_clearance")
                        constraints.append(self.config.self_clearance-self_clearance)
                    collision |= bool(exact["collision"] or any(value > 0 for value in constraints[-2:]))
                    clearances = [value for value in (world_clearance, self_clearance)
                                  if np.isfinite(value)]
                    if clearances:
                        minimum_clearance = min(minimum_clearance, min(clearances))
                except ValueError:
                    collision = True
                    constraint_names.append("exact_collision_query")
                    constraints.append(np.inf)
                local_max = max(constraints)
                if local_max > max_violation:
                    max_violation = local_max
                    maximum_kind = constraint_names[int(np.argmax(constraints))]
        passed = (not collision and map_inside and np.isfinite(max_violation)
                  and max_violation <= 2e-3 and np.isfinite(minimum_clearance))
        self.last_validation_metrics = {
            "maximum_violation_kind": maximum_kind,
            "minimum_world_distance": float(minimum_world_distance),
            "minimum_self_distance": float(minimum_self_distance),
            "collision_detected": bool(collision),
            "map_inside": bool(map_inside),
        }
        return passed, float(minimum_clearance), float(max_violation), float(maximum_dt)


__all__ = ["AerialManipulatorTrajectoryEvaluator"]
