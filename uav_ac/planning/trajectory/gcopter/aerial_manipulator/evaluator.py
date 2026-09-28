"""Analytic sampled penalties for whole-body 8-D quintic trajectories."""

import numpy as np

from ..mappings import polynomial_basis_matrix, smoothed_l1_array


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
        return (np.cross(first[0], second[0]),
                np.stack([np.cross(first[1][:, i], second[0])
                          +np.cross(first[0], second[1][:, i])
                          for i in range(dimensions)], axis=1))

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
                 gripper_opening, carry_payload):
        self.robot, self.esdf, self.quad, self.config = robot, esdf, quad, config
        self.bounds = np.asarray(workspace_bounds, dtype=float)
        self.gripper_opening = float(gripper_opening)
        self.carry_payload = bool(carry_payload)
        self.geometry = []
        geometry_spheres = []
        for geom_index, geom in enumerate(robot.collision_geometries()):
            start, end = geom["local_start"], geom["local_end"]
            half_length = .5*float(np.linalg.norm(end-start))
            count = 3 if half_length > 1e-9 else 1
            inflated_radius = float(geom["radius"])+(half_length/(count-1) if count > 1 else 0.)*.5
            for alpha in np.linspace(0., 1., count):
                center = start+alpha*(end-start)
                sphere_index = len(self.geometry)
                self.geometry.append((geom["body"], center, inflated_radius, geom_index))
                geometry_spheres.append(sphere_index)
        geoms = robot.collision_geometries()
        self.self_pairs = []
        for first in range(len(geoms)):
            for second in range(first+1, len(geoms)):
                a, b = geoms[first], geoms[second]
                if (a["body"] == b["body"] or a["parent"] == b["body"]
                        or b["parent"] == a["body"]):
                    continue
                if not ((a["contype"] & b["conaffinity"])
                        or (b["contype"] & a["conaffinity"])):
                    continue
                self.self_pairs.extend((i, j) for i in geometry_spheres
                                       if self.geometry[i][3] == first
                                       for j in geometry_spheres
                                       if self.geometry[j][3] == second)
        self.payload_body, self.payload_local = robot.frame_point("grasp")
        gripper_bodies = {"gripper_palm_body", "gripper_left_body", "gripper_right_body"}
        self.payload_pairs = ([] if not self.carry_payload else [
            (sphere, len(self.geometry)) for sphere in geometry_spheres
            if self.geometry[sphere][0] not in gripper_bodies
        ])

    def _configuration(self, sigma):
        from .task_targets import yaw_quaternion
        return np.r_[sigma[:3], yaw_quaternion(sigma[3]), sigma[4:8], self.gripper_opening]

    def _state_geometry(self, sigma):
        names = [sphere[0] for sphere in self.geometry]
        points = np.asarray([sphere[1] for sphere in self.geometry], dtype=float)
        radii = np.asarray([sphere[2] for sphere in self.geometry], dtype=float)
        if self.carry_payload:
            names.append(self.payload_body)
            points = np.vstack((points, self.payload_local))
            radii = np.r_[radii, self.config.payload_radius]
        positions, jacobians = self.robot.point_positions_and_jacobians(
            self._configuration(sigma), names, points, check_limits=False)
        return positions, jacobians, radii

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
        gradients[outside] = outside_delta[outside]/outside_norm[outside, None]
        signed = distances-outside_norm
        violations = radii+self.config.obstacle_clearance+tilt_uncertainty-signed
        costs, derivatives = smoothed_l1_array(violations, self.config.smoothing_epsilon)
        grad_violation = -np.einsum("nci,nc->ni", jacobians, gradients)
        grad_violation += tilt_state_gradient
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
            for i, j in pairs:
                delta = positions[i]-positions[j]
                distance = max(float(np.linalg.norm(delta)), 1e-10)
                violation = radii[i]+radii[j]+self.config.self_clearance-distance
                pair_cost, pair_derivative = smoothed_l1_array(
                    np.array([violation]), self.config.smoothing_epsilon)
                direction = delta/distance
                grad_violation = -(jacobians[i]-jacobians[j]).T@direction
                cost += self.config.self_collision_weight*float(pair_cost[0])
                grad += (self.config.self_collision_weight*pair_derivative[0]*grad_violation)
                if kind == "self":
                    self_violation = max(self_violation, float(violation))
                else:
                    payload_violation = max(payload_violation, float(violation))
                if kind == "payload":
                    minimum_clearance = min(minimum_clearance,
                                            distance-radii[i]-radii[j])
        return (cost, grad, grad_acceleration, obstacle_violation,
                minimum_clearance, self_violation, payload_violation)

    def sample_cost_gradient(self, sigma, velocity, acceleration, jerk):
        """Return cost, gradients for orders 0..3, maximum raw violation and clearance."""
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
        thrust = self.robot.mass*rho
        thrust_min, thrust_max = 4*quad.min_thrust, 4*quad.max_thrust
        thrust_grad = -self.robot.mass*body_z
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
        for piece, duration in enumerate(trajectory.durations):
            count = max(2, int(np.ceil(duration/self.config.validation_dt)))
            for local in np.linspace(0., duration, count+1):
                basis = [polynomial_basis_matrix(np.array([local]), d)[0]
                         for d in range(4)]
                state = [b@trajectory.coefficients[piece] for b in basis]
                _, _, violation, clearance = self.sample_cost_gradient(*state)
                max_violation = max(max_violation, violation)
                minimum_clearance = min(minimum_clearance, clearance)
                try:
                    exact = self.robot.check_collision(
                        self._configuration(state[0]), clearance=0.)
                    collision |= bool(exact["collision"])
                    minimum_clearance = min(minimum_clearance,
                                            float(exact["minimum_distance"]))
                except ValueError:
                    collision = True
                    max_violation = np.inf
        passed = (not collision and np.isfinite(max_violation)
                  and max_violation <= 2e-3 and np.isfinite(minimum_clearance))
        return passed, float(minimum_clearance), float(max_violation)


__all__ = ["AerialManipulatorTrajectoryEvaluator"]
