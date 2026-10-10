"""Dense whole-body validation independent of optimization quadrature."""

import numpy as np
from ..gcopter.trajectory import polynomial_basis_matrix
from .flatness import _flatness_attitude
from .trajectory import AerialManipulatorTrajectory


def validate_gripper_dwell(evaluator, trajectory, start_gap, end_gap, duration):
    from .trajectory import gripper_gap_motion, AerialManipulatorTrajectory
    old_gap = evaluator.gripper_opening
    coefficients = np.zeros((1, 6, 8))
    coefficients[0, 0] = trajectory.evaluate(trajectory.total_time)
    stationary = AerialManipulatorTrajectory(np.array([evaluator.config.validation_dt]),
        coefficients, np.repeat(coefficients[:, 0], 2, axis=0), 0., 0, True, 'dwell')
    checks = []
    count = max(2, int(np.ceil(duration/evaluator.config.validation_dt)))
    try:
        for elapsed in np.linspace(0., duration, count+1):
            evaluator.gripper_opening = gripper_gap_motion(start_gap, end_gap, elapsed, duration)[0]
            checks.append(evaluator.dense_validate(stationary))
    finally:
        evaluator.gripper_opening = old_gap
    return (all(c[0] for c in checks), min(c[1] for c in checks),
            max(c[2] for c in checks), duration/count)


def validate_task_waypoint(robot, target, derivatives, gap, gravity=9.81):
    from .constraints import task_residual
    residual = task_residual(robot, target, derivatives, gap, gravity)
    errors, offset, valid = {}, 0, True
    for name, unit, tolerance in (('position', 'm', 1e-4), ('orientation', 'rad', 1e-3),
                                 ('linear_velocity', 'mps', 1e-3), ('angular_velocity', 'radps', 1e-3)):
        if getattr(target, name) is None:
            continue
        size = int(np.count_nonzero(getattr(target, name+'_mask')))
        error = float(np.linalg.norm(residual[offset:offset+size]))
        errors[name+'_'+unit] = error
        valid = valid and np.isfinite(error) and error <= tolerance
        offset += size
    return bool(valid), residual, errors

class TrajectoryValidation:
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


def _is_retimeable_validation(metrics):
    kind = metrics.get("maximum_violation_kind")
    dynamic = {
        "linear_speed", "linear_acceleration", "yaw_rate", "yaw_acceleration",
        "joint_velocity", "joint_acceleration", "maximum_thrust", "minimum_thrust",
        "tilt", "body_rate",
    }
    return (kind in dynamic and not metrics.get("collision_detected", True)
            and metrics.get("map_inside", False))


def _retiming_scale(kind, violation, config):
    limits = {
        "linear_speed": config.max_speed,
        "linear_acceleration": config.max_acceleration,
        "yaw_rate": config.max_yaw_rate,
        "yaw_acceleration": config.max_yaw_acceleration,
        "joint_velocity": min(config.joint_velocity_limits),
        "joint_acceleration": min(config.joint_acceleration_limits),
        "body_rate": config.max_body_rate,
    }
    if kind in limits:
        # The constraints are expressed as squared magnitudes. Acceleration
        # scales with time^-2, while velocity/body rate scale with time^-1.
        power = 0.25 if kind.endswith("acceleration") else 0.5
        needed = (1.0+max(0.0, float(violation))/limits[kind]**2)**power
        return max(1.05, 1.05*needed)
    # Thrust and tilt depend nonlinearly on acceleration; use a conservative
    # bounded retry and let full-pose validation decide whether it is safe.
    return 1.1


def _time_stretch(trajectory, scale):
    if not np.isfinite(scale) or scale <= 1.0:
        raise ValueError("time stretch must be finite and greater than one")
    coefficients = trajectory.coefficients/scale**np.arange(6)[None, :, None]
    return AerialManipulatorTrajectory(
        trajectory.durations*scale, coefficients, trajectory.rrt_path,
        trajectory.cost, trajectory.iterations, trajectory.optimizer_converged,
        trajectory.optimizer_message, trajectory.validation_passed,
        trajectory.minimum_clearance, trajectory.maximum_violation,
        trajectory.validation_sample_dt)
