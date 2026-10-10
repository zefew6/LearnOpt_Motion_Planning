"""Whole-body collision contexts shared by search and optimization."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from ...geometry.grid_map import GridMap
from ...geometry.collision_broadphase import RRTBroadphase
from ..gcopter.mappings import polynomial_basis_matrix, smoothed_l1_array
from .flatness import _flatness_attitude, _flatness_attitude_tangent_jacobian as _flatness_attitude_tangent_jacobian_python

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


class WholeBodyCollision:
    def __init__(self, robot, esdf, quad, config, workspace_bounds,
                 gripper_opening, carry_payload, occupancy=None):
        self.robot, self.esdf, self.quad, self.config = robot, esdf, quad, config
        self.use_exact_world_penalty = True
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
        spheres = getattr(robot, 'planning_collision_spheres', lambda: ())()
        self.manual_spheres = bool(spheres)
        if self.manual_spheres:
            self._initialize_manual_spheres(spheres)
            return
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

    def _initialize_manual_spheres(self, spheres):
        """Immutable metadata for the authored robot; pair omissions are explicit."""
        expected = tuple(f'planning_sphere_{i:02d}' for i in range(1, 22))
        if tuple(s['name'] for s in spheres) != expected:
            raise ValueError('manual aerial collision model requires planning_sphere_01..21')
        self.geometry = [(s['body'], s['center'], s['radius'], i)
                         for i, s in enumerate(spheres)]
        self.sphere_names = tuple(s['body'] for s in spheres)
        self.sphere_points = np.array([s['center'] for s in spheres])
        self.sphere_radii = np.array([s['radius'] for s in spheres])
        self.geom_names = expected
        # One-based site identities from the reviewed mechanical layout.
        mount = {(1, 6), (1, 7), (1, 8)}
        connection = {(6, 8), (7, 8), (10, 11), (12, 14), (13, 14),
                      (17, 18), (17, 20), (10, 12)}
        pairs, omissions = [], {}
        welded = {'arm_link_3_body', 'gripper_palm_body'}
        self.rrt_fixed_clearance_self_pairs_skipped = 0
        for i, a in enumerate(spheres):
            for j in range(i+1, len(spheres)):
                b = spheres[j]
                reason = None
                if a['body'] == b['body'] or {a['body'], b['body']} == welded:
                    reason = 'rigid_assembly'
                elif (i+1, j+1) in mount:
                    reason = 'fixed_mount'
                elif (i+1, j+1) in connection:
                    reason = 'joint_connection'
                elif ({a['body'], b['body']} == {'gripper_left_body', 'gripper_right_body'}
                      and self.config.self_clearance <= .02):
                    reason = 'opposing_fingers_minimum_gap'
                    self.rrt_fixed_clearance_self_pairs_skipped += 1
                if reason:
                    omissions[(i, j)] = reason
                else:
                    pairs.append((i, j))
        self.self_pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        self.excluded_self_pairs = omissions
        self.self_pair_names = tuple((expected[i], expected[j]) for i, j in pairs)
        self.self_geom_pairs = self.self_pair_names
        gripper = {'gripper_palm_body', 'gripper_left_body', 'gripper_right_body'}
        self.payload_body, self.payload_local = self.robot.frame_point('grasp')
        self.payload_pairs = np.array([(i, len(spheres)) for i, s in enumerate(spheres)
            if self.carry_payload and s['body'] not in gripper], dtype=int).reshape(-1, 2)
        if self.carry_payload:
            self.sphere_names += (self.payload_body,)
            self.sphere_points = np.vstack((self.sphere_points, self.payload_local))
            self.sphere_radii = np.r_[self.sphere_radii, self.robot.payload_radius]
        self.sphere_geom_indices = np.full(len(self.sphere_radii), -1, dtype=int)
        for array in (self.sphere_points, self.sphere_radii, self.self_pairs, self.payload_pairs):
            array.setflags(write=False)
        self.rrt_broadphase_seconds = 0.

    def _manual_feasibility_batch(self, positions, margin):
        """Sphere/ESDF feasibility, avoiding cube occupancy false positives."""
        positions = np.asarray(positions, dtype=float)
        clipped = np.clip(positions, self.esdf.origin, self.esdf.upper)
        distances = self.esdf.distance(clipped.reshape(-1, 3)).reshape(positions.shape[:2])
        outside = np.linalg.norm(positions-clipped, axis=2)
        environment = np.max(self.sphere_radii+margin+self.esdf_discretization_margin
                             -distances+outside, axis=1)
        # No extrapolation into unknown space, including sphere extents.
        boundary = np.max(np.maximum(self.esdf.origin+self.sphere_radii[None,:,None]-positions,
                         positions-(self.esdf.upper-self.sphere_radii[None,:,None])), axis=(1,2))
        environment = np.maximum(environment, boundary)
        values = []
        for pairs in (self.self_pairs, self.payload_pairs):
            if not len(pairs):
                values.append(np.full(len(positions), -np.inf))
                continue
            first, second = pairs.T
            length = np.linalg.norm(positions[:,first]-positions[:,second], axis=2)
            values.append(np.max(self.sphere_radii[first]+self.sphere_radii[second]
                                 +self.config.self_clearance-length, axis=1))
        return environment, values[0], values[1]

    def _manual_collision_cost_gradient(self, sigma, acceleration):
        positions, jacobians, acceleration_jacobians, _, _, _ = self._full_pose_geometry(sigma, acceleration)
        clipped = np.clip(positions, self.esdf.origin, self.esdf.upper)
        distances, gradients = self.esdf.distance_and_gradient(clipped)
        outside_delta = positions-clipped
        outside_norm = np.linalg.norm(outside_delta, axis=1)
        outside = outside_norm > 0.
        signed_gradient = gradients.copy()
        signed_gradient[(positions < self.esdf.origin) | (positions > self.esdf.upper)] = 0.
        signed_gradient[outside] -= outside_delta[outside]/outside_norm[outside, None]
        signed = distances-outside_norm
        violations = self.sphere_radii+self.config.obstacle_clearance+self.esdf_discretization_margin-signed
        boundary = self.config.obstacle_clearance+self.esdf_discretization_margin+outside_norm
        active = outside & (boundary > violations)
        violations[outside] = np.maximum(violations[outside], boundary[outside])
        normals = -signed_gradient
        normals[active] = outside_delta[active]/outside_norm[active, None]
        lower_face = self.esdf.origin+self.sphere_radii[:,None]-positions
        upper_face = positions-(self.esdf.upper-self.sphere_radii[:,None])
        faces = np.maximum(lower_face, upper_face)
        axis = np.argmax(faces, axis=1)
        face_violation = faces[np.arange(len(positions)), axis]
        face_active = face_violation > violations
        violations = np.maximum(violations, face_violation)
        normals[face_active] = 0.
        ids = np.flatnonzero(face_active)
        normals[ids, axis[ids]] = np.where(
            lower_face[ids,axis[ids]] > upper_face[ids,axis[ids]], -1., 1.)
        costs, derivatives = smoothed_l1_array(violations, self.config.smoothing_epsilon)
        weights = self.config.obstacle_weight*derivatives
        cost = self.config.obstacle_weight*float(costs.sum())
        gradient = np.einsum('n,nci,nc->i', weights, jacobians, normals)
        gradient_acceleration = np.einsum('n,nci,nc->i', weights, acceleration_jacobians, normals)
        clearance = min(float(np.min(signed-self.sphere_radii, initial=np.inf)),
                        -float(np.max(face_violation)))
        pair_violations = []
        for pairs in (self.self_pairs, self.payload_pairs):
            if not len(pairs):
                pair_violations.append(-np.inf)
                continue
            first, second = pairs.T
            delta = positions[first]-positions[second]
            length = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
            violation = self.sphere_radii[first]+self.sphere_radii[second]+self.config.self_clearance-length
            values, derivative = smoothed_l1_array(violation, self.config.smoothing_epsilon)
            cost += self.config.self_collision_weight*float(values.sum())
            weight = -self.config.self_collision_weight*derivative
            direction = delta/length[:, None]
            gradient += np.einsum('n,nci,nc->i', weight, jacobians[first]-jacobians[second], direction)
            gradient_acceleration += np.einsum('n,nci,nc->i', weight,
                acceleration_jacobians[first]-acceleration_jacobians[second], direction)
            pair_violations.append(float(np.max(violation, initial=-np.inf)))
            clearance = min(clearance, float(np.min(length-self.sphere_radii[first]-self.sphere_radii[second])))
        return (cost, gradient, gradient_acceleration,
                float(np.max(violations, initial=-np.inf)), clearance, *pair_violations)

    def _configuration(self, sigma, acceleration=None):
        from .flatness import yaw_quaternion
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
        if self.manual_spheres:
            broad = self._manual_feasibility_batch(positions[None,:], self.config.rrt_obstacle_margin)
            return float(broad[0][0]), max(float(broad[1][0]), float(broad[2][0]))
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
        if self.manual_spheres:
            phase_started = time.perf_counter()
            broad = self._manual_feasibility_batch(points, self.config.rrt_obstacle_margin)
            self.rrt_broadphase_seconds += time.perf_counter()-phase_started
            return (broad[0] <= 1e-10) & (broad[1] <= 1e-10) & (broad[2] <= 1e-10)
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
        xyzw = Rotation.from_matrix(rotation).as_quat()
        configuration = np.r_[sigma[:3], xyzw[3], xyzw[:3], sigma[4:8], self.gripper_opening]
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
        if self.manual_spheres:
            acceleration = np.zeros(8) if acceleration is None else np.asarray(acceleration, dtype=float)
            if acceleration.shape != (8,) or not np.all(np.isfinite(acceleration)):
                raise ValueError('acceleration must be a finite 8-vector')
            return self._manual_collision_cost_gradient(sigma, acceleration)
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
        signed_gradient[(positions < lower) | (positions > upper)] = 0.
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
        penetrating_query_indices = np.empty(0, dtype=int)
        if self.use_exact_world_penalty:
            candidates = np.flatnonzero(world_candidate_geom[self._world_query_geometry])
            if len(candidates):
                # In deep capsule/box overlap MuJoCo may return a flat -radius
                # distance while its witness normal remains nonzero. The ESDF
                # envelope provides an actual descent loss for those samples.
                queried = self._world_query_indices[candidates]
                distances_exact = self.robot.exact_collision_distances(
                    None, pairs=self._query_pairs, pair_indices=queried, prepared=prepared,
                    payload_attached=False, check_limits=False, with_points=False)['distances']
                penetrating = distances_exact < 0.
                penetrating_geoms = self._world_query_geometry[candidates[penetrating]]
                penetrating_query_indices = queried[penetrating]
                unsuppress = np.isin(self.sphere_geom_indices, penetrating_geoms)
                world_suppressed[unsuppress] = False
        if not self.use_exact_world_penalty:
            world_suppressed[:] = False
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
        selected[penetrating_query_indices] = False
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
        if not self.use_exact_world_penalty:
            selected[query_kinds == 1] = False
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
