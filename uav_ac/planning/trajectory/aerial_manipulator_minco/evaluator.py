"""Analytic sampled penalties for whole-body 8-D quintic trajectories."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from ...geometry.grid_map import GridMap
from ...geometry.collision_broadphase import RRTBroadphase
from ..gcopter.mappings import polynomial_basis_matrix, smoothed_l1_array
from .types import _flatness_attitude, _flatness_attitude_tangent_jacobian as _flatness_attitude_tangent_jacobian_python

try:
    from ...native import _aerial_constraints as _native_aerial
except ImportError:
    _native_aerial = None


def _flatness_attitude_tangent_jacobian(acceleration, yaw, gravity=9.81):
    if _native_aerial is not None:
        return _native_aerial.flatness_attitude_tangent_jacobian(
            np.asarray(acceleration, dtype=float), yaw, gravity)
    return _flatness_attitude_tangent_jacobian_python(acceleration, yaw, gravity)


_FIXED_GRIPPER_GAP_PAIRS = frozenset({
    frozenset(("gripper_pad_left", "gripper_pad_right")),
    frozenset(("gripper_finger_left", "gripper_pad_right")),
    frozenset(("gripper_pad_left", "gripper_finger_right")),
})


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


def _exact_collision_penalty_python(
        cost, state_gradient, acceleration_gradient, distances, jacobians, kinds,
        attitude_jacobian, world_margin, self_margin, world_weight, self_weight,
        epsilon, obstacle_violation, self_violation, payload_violation, minimum_clearance):
    """Reference exact-witness reduction; inputs remain owned by the caller."""
    grad = np.asarray(state_gradient, dtype=float).copy()
    grad_acceleration = np.asarray(acceleration_gradient, dtype=float).copy()
    for kind, distance, pair_jacobian in zip(kinds, distances, jacobians, strict=True):
        margin = world_margin if kind == 1 else self_margin
        violation = margin-float(distance)
        exact_cost, exact_derivative = smoothed_l1_array(np.asarray([violation]), epsilon)
        weight = world_weight if kind == 1 else self_weight
        derivative = weight*float(exact_derivative[0])
        cost += weight*float(exact_cost[0])
        state_gradient = np.r_[
            pair_jacobian[:3], pair_jacobian[3:6]@attitude_jacobian[:, 3],
            pair_jacobian[6:10]]
        acceleration_gradient = np.zeros(8)
        acceleration_gradient[:3] = pair_jacobian[3:6]@attitude_jacobian[:, :3]
        grad -= derivative*state_gradient
        grad_acceleration -= derivative*acceleration_gradient
        if kind == 1:
            obstacle_violation = max(obstacle_violation, violation)
        elif kind == 2:
            self_violation = max(self_violation, violation)
        else:
            payload_violation = max(payload_violation, violation)
        minimum_clearance = min(minimum_clearance, float(distance-margin))
    return (cost, grad, grad_acceleration, obstacle_violation,
            minimum_clearance, self_violation, payload_violation)


class AerialManipulatorTrajectoryEvaluator:
    def __init__(self, robot, esdf, quad, config, workspace_bounds,
                 gripper_opening, carry_payload, occupancy=None):
        self.robot, self.esdf, self.quad, self.config = robot, esdf, quad, config
        self.objective_samples = 0
        self.validation_samples = 0
        self.mass = robot.mass
        self.bounds = np.asarray(workspace_bounds, dtype=float)
        self.gripper_opening = float(gripper_opening)
        self.carry_payload = bool(carry_payload)
        # RRT only needs a conservative binary feasibility map.  Keep the
        # signed field for optimization and validation, but avoid trilinear
        # ESDF interpolation on every search sample and edge state.
        self.occupancy = (GridMap.from_esdf(esdf)
                          if occupancy is None else occupancy)
        self.geometry = []
        self.esdf_discretization_margin = max(
            float(config.esdf_discretization_margin),
            float(np.sqrt(3.)*esdf.resolution+1e-9))
        self.rrt_exact_candidate_states = 0
        self.rrt_exact_candidate_seconds = 0.0
        self.rrt_exact_pair_queries = 0
        self.rrt_batch_fk_seconds = 0.0
        self.rrt_occupancy_check_seconds = 0.0
        self.rrt_sphere_pair_check_seconds = 0.0
        self.rrt_world_pair_filter_seconds = 0.0
        self.rrt_exact_geometry_seconds = 0.0
        self.rrt_environment_candidates = 0
        self.rrt_self_candidates = 0
        self.rrt_payload_candidates = 0
        self.rrt_exact_rejected_states = 0
        geometry_spheres = []
        geoms = robot.collision_geometries()
        for geom_index, geom in enumerate(geoms):
            start, end = geom["local_start"], geom["local_end"]
            half_length = .5*float(np.linalg.norm(end-start))
            radius = float(geom["radius"])
            # Bound every capsule segment with overlapping spheres whose
            # center spacing is at most one diameter. Three large spheres
            # made the broad phase needlessly close valid paths to obstacles.
            count = (max(2, int(np.ceil(half_length/radius))+1)
                     if half_length > 1e-9 and radius > 1e-9 else 1)
            axial_half_spacing = (half_length/(count-1) if count > 1 else 0.)
            inflated_radius = float(np.hypot(radius, axial_half_spacing))
            for alpha in np.linspace(0., 1., count):
                center = start+alpha*(end-start)
                sphere_index = len(self.geometry)
                self.geometry.append((geom["body"], center, inflated_radius, geom_index))
                geometry_spheres.append(sphere_index)
        self.sphere_geom_indices = np.asarray(
            [sphere[3] for sphere in self.geometry], dtype=int)
        self.geom_names = tuple(geom["name"] for geom in geoms)
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
        # These three opposing gripper contacts have a rigidly determined
        # separation of at least the commanded opening. They cannot violate
        # self_clearance when the opening exceeds that margin, although their
        # conservative link spheres overlap in every state and would trigger
        # exact MuJoCo queries for every RRT sample.
        self.rrt_fixed_clearance_self_pairs_skipped = 0
        if self.gripper_opening > self.config.self_clearance+1e-9:
            keep = np.ones(len(self.self_pairs), dtype=bool)
            skipped = set()
            for index, (first, second) in enumerate(self.self_pairs):
                names = frozenset((
                    self.geom_names[self.sphere_geom_indices[first]],
                    self.geom_names[self.sphere_geom_indices[second]],
                ))
                if names in _FIXED_GRIPPER_GAP_PAIRS:
                    keep[index] = False
                    skipped.add(names)
            self.self_pairs = self.self_pairs[keep]
            self.rrt_fixed_clearance_self_pairs_skipped = len(skipped)
        self.self_pair_names = tuple(
            (self.geom_names[self.sphere_geom_indices[first]],
             self.geom_names[self.sphere_geom_indices[second]])
            for first, second in self.self_pairs)
        self.self_geom_pairs = tuple(sorted({
            tuple(self.geom_names[self.sphere_geom_indices[index]] for index in pair)
            for pair in self.self_pairs
        }))
        self.world_geom_pairs = tuple(robot.collision_pairs("world"))
        self.world_pair_set = set(self.world_geom_pairs)
        self.world_aabbs = robot.collision_environment_aabbs()
        self.world_aabb_names = tuple(self.world_aabbs)
        self.world_aabb_lower = np.asarray(
            [self.world_aabbs[name][0] for name in self.world_aabb_names])
        self.world_aabb_upper = np.asarray(
            [self.world_aabbs[name][1] for name in self.world_aabb_names])
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
            self.sphere_radii = np.r_[self.sphere_radii, robot.payload_radius]
            self.sphere_geom_indices = np.r_[self.sphere_geom_indices, -1]

        # Compile model identities and broad-phase-to-exact indices once.
        query_names, query_kinds, query_lookup = [], [], {}

        def register(pair, kind):
            key = frozenset(pair)
            if key not in query_lookup:
                query_lookup[key] = len(query_names)
                query_names.append(pair)
                query_kinds.append(kind)
            return query_lookup[key]

        self._world_query_indices = np.asarray(
            [register(pair, 1) for pair in self.world_geom_pairs], dtype=int)
        geom_lookup = {name: index for index, name in enumerate(self.geom_names)}
        self._world_query_geometry = np.asarray(
            [geom_lookup[pair[0]] for pair in self.world_geom_pairs], dtype=int)
        self._self_query_indices = np.asarray(
            [register(pair, 2) for pair in self.self_pair_names], dtype=int)
        self._payload_query_indices = np.asarray([
            register(("payload_marker_geom", self.geom_names[self.sphere_geom_indices[first]]), 3)
            for first, _ in self.payload_pairs], dtype=int)
        self._query_pairs = robot.compile_collision_pairs(
            query_names, payload_attached=self.carry_payload)
        self._query_base_kinds = np.asarray(query_kinds, dtype=np.int8)
        self._query_is_payload = np.asarray([
            "payload_marker_geom" in pair for pair in query_names], dtype=bool)
        self._query_is_world = np.zeros(len(query_names), dtype=bool)
        self._query_is_world[self._world_query_indices] = True
        self._query_is_world &= ~self._query_is_payload
        self._world_query_table = np.full(
            (len(self.geom_names), len(self.world_aabb_names)), -1, dtype=int)
        for world_column, world_name in enumerate(self.world_aabb_names):
            for geometry_index, geometry_name in enumerate(self.geom_names):
                pair = (geometry_name, world_name)
                if pair in self.world_pair_set:
                    self._world_query_table[geometry_index, world_column] = query_lookup[frozenset(pair)]

        self._broadphase = RRTBroadphase(
            self.occupancy, radii=self.sphere_radii,
            self_pairs=self.self_pairs, payload_pairs=self.payload_pairs,
            self_query_indices=self._self_query_indices,
            payload_query_indices=self._payload_query_indices,
            sphere_geom_indices=self.sphere_geom_indices,
            world_lower=self.world_aabb_lower, world_upper=self.world_aabb_upper,
            world_query_table=self._world_query_table,
            query_count=len(self._query_pairs.ids), self_clearance=self.config.self_clearance)
        self.rrt_broadphase_seconds = 0.0

    def _configuration(self, sigma, acceleration=None):
        from .task_targets import yaw_quaternion
        quaternion = yaw_quaternion(sigma[3])
        if acceleration is not None:
            rotation, _ = _flatness_attitude(
                acceleration[:3], np.zeros(3), sigma[3], 0., self.quad.g)
            xyzw = Rotation.from_matrix(rotation).as_quat()
            quaternion = np.r_[xyzw[3], xyzw[:3]]
        return np.r_[sigma[:3], quaternion, sigma[4:8], self.gripper_opening]

    def collision_feasible(self, sigma, *, narrow_phase=True,
                           occupancy_margin=None, exact_world_clearance=None,
                           sphere_positions=None):
        """Return environment and self/payload feasibility violations.

        RRT candidates use the complete sphere envelope and may request exact
        geometry confirmation with the RRT clearance. MINCO uses the same
        envelope and its larger optimization clearance before exact checks.
        """
        configuration = self._configuration(sigma)
        prepared = None
        if sphere_positions is None:
            prepared = self.robot.prepare_kinematics(configuration, check_limits=False)
            positions = self.robot.point_positions(
                None, self.sphere_names, self.sphere_points,
                check_limits=False, prepared=prepared)
        else:
            positions = np.asarray(sphere_positions, dtype=float)
            if positions.shape != (len(self.sphere_names), 3) or not np.all(
                    np.isfinite(positions)):
                raise ValueError("sphere_positions must be a finite (S, 3) array")
        occupancy_margin = (float(occupancy_margin) if occupancy_margin is not None else
                            self.config.obstacle_clearance+self.esdf_discretization_margin
                            if narrow_phase else self.config.rrt_obstacle_margin)
        exact_world_clearance = (float(exact_world_clearance)
                                 if exact_world_clearance is not None else
                                 self.config.obstacle_clearance)
        phase_started = time.perf_counter()
        broad = self._broadphase.query(
            positions, occupancy_margin, exact_world_clearance, narrow_phase=narrow_phase)
        self.rrt_broadphase_seconds += time.perf_counter()-phase_started
        environment, self_collision, payload = (float(values[0]) for values in broad[:3])
        if narrow_phase and max(environment, self_collision, payload) > 0.:
            return self._refine_collision(
                configuration, environment, self_collision, payload,
                broad[3][0], int(broad[4][0]), bool(broad[5][0]),
                exact_world_clearance, prepared=prepared)
        return environment, max(self_collision, payload)

    def _refine_collision(self, configuration, environment_violation,
                          self_violation, payload_violation, selected, world_count,
                          outside, exact_world_clearance, *, prepared=None):
        """Apply unchanged MuJoCo narrow-phase rules to a prepared broad phase."""
        indices = np.flatnonzero(selected)
        self.rrt_exact_pair_queries += len(indices)
        phase_started = time.perf_counter()
        exact = (self.robot.exact_collision_distances(
            None if prepared is not None else configuration,
            pairs=self._query_pairs, pair_indices=indices,
            payload_attached=self.carry_payload, prepared=prepared,
            with_jacobians=False, check_limits=False, with_points=False)
                 if len(indices) else {"distances": np.empty(0)})
        self.rrt_exact_geometry_seconds += time.perf_counter()-phase_started
        if len(indices):
            distances = exact["distances"]
            world = self._query_is_world[indices]
            payload = self._query_is_payload[indices]
            self_collision = ~(world | payload)
            world_distances = distances[world]
            self_distances = distances[self_collision]
            payload_distances = distances[payload]
            if len(world_distances):
                environment_violation = exact_world_clearance-float(np.min(world_distances))
            if len(self_distances):
                self_violation = self.config.self_clearance-float(np.min(self_distances))
            if len(payload_distances):
                payload_violation = self.config.self_clearance-float(np.min(payload_distances))
        if environment_violation > 0. and not world_count and not outside:
            # This was a conservative occupancy hit with no nearby exact
            # world geometry; reject only pairs that survive the AABB test.
            environment_violation = -1.0
        # An out-of-grid query is a hard map-boundary violation.  Exact
        # narrow-phase distances can clear a conservative occupancy hit,
        # but they must never turn an unknown/outside cell into free space.
        if outside:
            environment_violation = max(environment_violation, 1.0)
        return environment_violation, max(self_violation, payload_violation)

    def collision_feasible_batch(self, states, *, exact_candidates=True):
        """Batch coarse checks once, then refine only their candidate states."""
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 8 or not np.all(np.isfinite(states)):
            raise ValueError("states must be a finite (B, 8) array")
        if not len(states):
            return np.empty(0, dtype=bool)
        phase_started = time.perf_counter()
        configurations = np.asarray([self._configuration(state) for state in states])
        points = self.robot.point_positions_batch(
            configurations, self.sphere_names, self.sphere_points, check_limits=False)
        self.rrt_batch_fk_seconds += time.perf_counter()-phase_started
        phase_started = time.perf_counter()
        broad = self._broadphase.query(
            points, self.config.rrt_obstacle_margin, self.config.rrt_obstacle_margin,
            narrow_phase=exact_candidates)
        self.rrt_broadphase_seconds += time.perf_counter()-phase_started
        environment_candidate, self_candidate, payload_candidate = (
            values > 0. for values in broad[:3])
        candidates = environment_candidate | self_candidate | payload_candidate
        valid = ~candidates
        self.rrt_environment_candidates += int(np.sum(environment_candidate))
        self.rrt_self_candidates += int(np.sum(self_candidate))
        self.rrt_payload_candidates += int(np.sum(payload_candidate))
        if exact_candidates:
            exact_started = time.perf_counter()
            for index in np.flatnonzero(candidates):
                environment, self_collision = self._refine_collision(
                    configurations[index], float(broad[0][index]), float(broad[1][index]),
                    float(broad[2][index]), broad[3][index], int(broad[4][index]),
                    bool(broad[5][index]), self.config.rrt_obstacle_margin)
                valid[index] = environment <= 0. and self_collision <= 0.
            self.rrt_exact_rejected_states += int(np.sum(candidates & ~valid))
            self.rrt_exact_candidate_states += int(np.sum(candidates))
            self.rrt_exact_candidate_seconds += time.perf_counter()-exact_started
        return valid

    def _full_pose_geometry(self, sigma, acceleration):
        acceleration = np.asarray(acceleration, dtype=float)
        rotation, attitude_jacobian = _flatness_attitude_tangent_jacobian(
            acceleration[:3], sigma[3], self.quad.g)
        configuration = self._configuration(sigma, acceleration)
        prepared = self.robot.prepare_kinematics(configuration, check_limits=False)
        positions, pose_jacobians = self.robot.point_positions_and_pose_jacobians(
            None, self.sphere_names, self.sphere_points,
            check_limits=False, prepared=prepared)
        state_jacobians = np.zeros((len(positions), 3, 8))
        acceleration_jacobians = np.zeros_like(state_jacobians)
        state_jacobians[:, :, :3] = pose_jacobians[:, :, :3]
        state_jacobians[:, :, 3] = np.einsum(
            "sij,j->si", pose_jacobians[:, :, 3:6], attitude_jacobian[:, 3])
        state_jacobians[:, :, 4:8] = pose_jacobians[:, :, 6:10]
        acceleration_jacobians[:, :, :3] = np.einsum(
            "sij,jk->sik", pose_jacobians[:, :, 3:6], attitude_jacobian[:, :3])
        return (positions, state_jacobians, acceleration_jacobians,
                attitude_jacobian, configuration, prepared)

    def collision_cost_gradient(self, sigma, acceleration=None):
        acceleration = (np.zeros(8) if acceleration is None
                        else np.asarray(acceleration, dtype=float))
        if acceleration.shape != (8,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("acceleration must be a finite 8-vector")
        positions, jacobians, acceleration_jacobians, attitude_jacobian, configuration, prepared = \
            self._full_pose_geometry(sigma, acceleration)
        radii = self.sphere_radii
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
                      +self.esdf_discretization_margin-signed)
        boundary_violation = (self.config.obstacle_clearance
                              +self.esdf_discretization_margin+outside_norm)
        boundary_active = outside & (boundary_violation > violations)
        violations[outside] = np.maximum(violations[outside], boundary_violation[outside])
        # The sphere cover is a broad phase. Exact geometry decides candidates
        # so the conservative capsule cover cannot close a valid shelf opening.
        world_candidate_geom = np.zeros(len(self.geom_names), dtype=bool)
        robot_sphere = self.sphere_geom_indices >= 0
        world_candidate_geom[self.sphere_geom_indices[
            robot_sphere & (violations > 0.)]] = True
        world_suppressed = np.zeros(len(positions), dtype=bool)
        world_suppressed[robot_sphere] = (
            world_candidate_geom[self.sphere_geom_indices[robot_sphere]]
            & ~outside[robot_sphere])
        costs, derivatives = smoothed_l1_array(violations, self.config.smoothing_epsilon)
        costs[world_suppressed] = 0.
        derivatives[world_suppressed] = 0.
        grad_violation = -np.einsum("nci,nc->ni", jacobians, signed_gradient)
        grad_acceleration_violation = -np.einsum(
            "nci,nc->ni", acceleration_jacobians, signed_gradient)
        grad_violation[boundary_active] = np.einsum(
            "nci,nc->ni", jacobians[boundary_active],
            outside_delta[boundary_active]/outside_norm[boundary_active, None])
        grad_acceleration_violation[boundary_active] = np.einsum(
            "nci,nc->ni", acceleration_jacobians[boundary_active],
            outside_delta[boundary_active]/outside_norm[boundary_active, None])
        grad = self.config.obstacle_weight*np.sum(
            derivatives[:, None]*grad_violation, axis=0)
        grad_acceleration = self.config.obstacle_weight*np.sum(
            derivatives[:, None]*grad_acceleration_violation, axis=0)
        cost = self.config.obstacle_weight*float(np.sum(costs))
        broad_obstacle_violation = float(np.max(
            violations[~world_suppressed], initial=-np.inf))
        obstacle_violation = broad_obstacle_violation
        broad_clearance = signed-radii
        minimum_clearance = float(np.min(
            broad_clearance[~world_suppressed], initial=np.inf))

        selected = np.zeros(len(self._query_pairs.ids), dtype=bool)
        selected[self._world_query_indices[
            world_candidate_geom[self._world_query_geometry]]] = True
        query_kinds = self._query_base_kinds.copy()
        self_violation = -np.inf
        payload_violation = -np.inf
        for pairs, kind in ((self.self_pairs, "self"), (self.payload_pairs, "payload")):
            if not len(pairs):
                continue
            first, second = pairs.T
            delta = positions[first]-positions[second]
            pair_distances = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
            pair_violations = (radii[first]+radii[second]+self.config.self_clearance
                               -pair_distances)
            pair_cost, pair_derivative = smoothed_l1_array(
                pair_violations, self.config.smoothing_epsilon)
            candidate = pair_violations > 0.
            query_indices = (self._payload_query_indices if kind == "payload"
                             else self._self_query_indices)[candidate]
            selected[query_indices] = True
            query_kinds[query_indices] = 3 if kind == "payload" else 2
            pair_cost[candidate] = 0.
            pair_derivative[candidate] = 0.
            directions = delta/pair_distances[:, None]
            pair_jacobians = jacobians[first]-jacobians[second]
            pair_acceleration_jacobians = (acceleration_jacobians[first]
                                           -acceleration_jacobians[second])
            pair_gradients = -np.einsum("nci,nc->ni", pair_jacobians, directions)
            pair_acceleration_gradients = -np.einsum(
                "nci,nc->ni", pair_acceleration_jacobians, directions)
            weight = self.config.self_collision_weight
            cost += weight*float(np.sum(pair_cost))
            grad += weight*np.einsum("n,ni->i", pair_derivative, pair_gradients)
            grad_acceleration += weight*np.einsum(
                "n,ni->i", pair_derivative, pair_acceleration_gradients)
            noncandidate = pair_violations.copy()
            noncandidate[candidate] = -np.inf
            if kind == "self":
                self_violation = float(np.max(noncandidate, initial=-np.inf))
            else:
                payload_violation = float(np.max(noncandidate, initial=-np.inf))
                pair_clearance = pair_distances-radii[first]-radii[second]
                pair_clearance[candidate] = np.inf
                minimum_clearance = min(minimum_clearance,
                                        float(np.min(pair_clearance, initial=np.inf)))
        indices = np.flatnonzero(selected)
        if len(indices):
            jacobian_thresholds = {
                self._query_pairs.keys[index]: (self.config.obstacle_clearance
                    if query_kinds[index] == 1 else self.config.self_clearance)
                for index in indices
            }
            exact = self.robot.exact_collision_distances(
                None, pairs=self._query_pairs, pair_indices=indices, prepared=prepared,
                payload_attached=self.carry_payload,
                with_jacobians=True, with_pose_jacobians=True,
                jacobian_distance_thresholds=jacobian_thresholds,
                check_limits=False)
            accumulate_exact = (_exact_collision_penalty_python if _native_aerial is None
                                else _native_aerial.exact_collision_penalty)
            return accumulate_exact(
                cost, grad, grad_acceleration,
                np.asarray(exact["distances"], dtype=float),
                np.asarray(exact["jacobians"], dtype=float),
                np.asarray(query_kinds[indices], dtype=np.int8), attitude_jacobian,
                self.config.obstacle_clearance, self.config.self_clearance,
                self.config.obstacle_weight, self.config.self_collision_weight,
                self.config.smoothing_epsilon, obstacle_violation, self_violation,
                payload_violation, minimum_clearance)
        return (cost, grad, grad_acceleration, obstacle_violation,
                minimum_clearance, self_violation, payload_violation)

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
        maximum_piece = -1
        minimum_world_distance = np.inf
        minimum_self_distance = np.inf
        for piece, duration in enumerate(trajectory.durations):
            coefficient = trajectory.coefficients[piece]

            # Bound internal extrema explicitly.  A diffeomorphic waypoint map
            # constrains only the knots; a quintic can still overshoot between
            # them even when every sampled knot is legal.
            lower, upper = self.robot.limits.joint_lower, self.robot.limits.joint_upper
            for joint in range(4):
                derivative = coefficient[1:, 4+joint]*np.arange(1, 6)
                roots = np.polynomial.polynomial.polyroots(derivative)
                local = np.r_[0., duration, roots.real[
                    (np.abs(roots.imag) <= 1e-9)
                    & (roots.real > 0.) & (roots.real < duration)]]
                values = polynomial_basis_matrix(local, 0)@coefficient[:, 4+joint]
                candidates = ((float(np.max(values-upper[joint])), "joint_upper_extremum"),
                              (float(np.max(lower[joint]-values)), "joint_lower_extremum"))
                for violation, kind in candidates:
                    if violation > max_violation:
                        max_violation, maximum_kind = violation, kind

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
                    # ``exact['collision']`` is physical contact. Clearance-margin
                    # violations stay in ``max_violation`` and are judged against
                    # the dense validator's small numeric tolerance below.
                    collision |= bool(exact["collision"])
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
                    maximum_piece = int(piece)
        passed = (not collision and map_inside and np.isfinite(max_violation)
                  and max_violation <= 3e-3 and np.isfinite(minimum_clearance))
        self.last_validation_metrics = {
            "maximum_violation_kind": maximum_kind,
            "maximum_violation_piece": maximum_piece,
            "minimum_world_distance": float(minimum_world_distance),
            "minimum_self_distance": float(minimum_self_distance),
            "collision_detected": bool(collision),
            "map_inside": bool(map_inside),
        }
        return passed, float(minimum_clearance), float(max_violation), float(maximum_dt)


__all__ = ["AerialManipulatorTrajectoryEvaluator"]
