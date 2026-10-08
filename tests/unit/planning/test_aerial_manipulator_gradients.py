from math import factorial

import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import (
    AerialManipulatorMINCOConfig,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.evaluator import (
    AerialManipulatorTrajectoryEvaluator,
    _flatness_body_rate_squared,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.planner import (
    AerialManipulatorMINCO,
    _combined_metrics,
    _initial_duration,
    _time_stretch,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.types import (
    AerialManipulatorTrajectory,
)
from uav_ac.planning.trajectory.gcopter.mappings import inverse_time
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.planning.trajectory.gcopter.types import GCOPTERTrajectory
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_piecewise_quintic_evaluation_is_shared_and_preserves_boundaries():
    rng = np.random.default_rng(17)
    durations = np.array([.35, .8])
    coefficients = rng.normal(size=(2, 6, 8))
    path = np.zeros((2, 8))
    aerial = AerialManipulatorTrajectory(
        durations, coefficients, path, 0.0, 0, True, "test")
    gcopter = GCOPTERTrajectory(
        durations, coefficients, np.zeros(2, dtype=int), 0.0, 0, True, "test")
    times = np.array([-.1, 0., .35, .35+1.e-10, 1.15, 1.4])

    def baseline(query, derivative):
        query = np.asarray(query, dtype=float)
        scalar = query.ndim == 0
        flat = np.clip(query.reshape(-1), 0., np.sum(durations))
        boundaries = np.cumsum(durations)
        pieces = np.minimum(np.searchsorted(boundaries, flat, side="right"), 1)
        starts = np.r_[0., boundaries[:-1]]
        values = []
        for local, piece in zip(flat-starts[pieces], pieces, strict=True):
            basis = np.array([
                0. if power < derivative else
                factorial(power)/factorial(power-derivative)*local**(power-derivative)
                for power in range(6)])
            values.append(basis@coefficients[piece])
        values = np.asarray(values)
        return values[0] if scalar else values

    for derivative in range(6):
        expected = baseline(times, derivative)
        assert gcopter.evaluate(times, derivative) == pytest.approx(expected)
        assert aerial.evaluate(times, derivative) == pytest.approx(expected)
        assert aerial.evaluate(.35, derivative) == pytest.approx(
            gcopter.evaluate(.35, derivative))


def test_local_refinement_metrics_sum_attempt_costs_once():
    base = {"rrt_nodes": 12, "rrt_seconds": .4, "astar_expansions": 8}
    attempts = [
        {"optimizer_seconds": .7, "objective_calls": 20,
         "objective_samples": 80, "optimizer_iterations": 12,
         "validation_seconds": .1, "validation_samples": 50,
         "minco_pieces": 5},
        {"optimizer_seconds": .3, "objective_calls": 10,
         "objective_samples": 40, "optimizer_iterations": 7,
         "validation_seconds": .2, "validation_samples": 60,
         "minco_pieces": 6},
    ]

    metrics = _combined_metrics(base, attempts, 2.0)

    assert metrics["rrt_nodes"] == 12
    assert metrics["rrt_seconds"] == .4
    assert metrics["astar_expansions"] == 8
    assert metrics["optimizer_seconds"] == pytest.approx(1.0)
    assert metrics["objective_calls"] == 30
    assert metrics["objective_samples"] == 120
    assert metrics["optimizer_iterations"] == 19
    assert metrics["validation_seconds"] == pytest.approx(.3)
    assert metrics["validation_samples"] == 110
    assert metrics["minco_pieces"] == 6
    assert metrics["total_seconds"] == 2.0


def test_flatness_body_rate_gradient_matches_finite_difference():
    acceleration = np.array([.7, -.3, .2])
    jerk = np.array([.4, .1, -.2])
    yaw, yaw_rate, gravity = .6, -.25, 9.81
    value, gradient = _flatness_body_rate_squared(
        acceleration, jerk, yaw, yaw_rate, gravity)
    variables = np.r_[acceleration, jerk, yaw, yaw_rate]
    epsilon = 1e-6
    numerical = np.empty(8)
    for index in range(8):
        plus, minus = variables.copy(), variables.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            _flatness_body_rate_squared(
                plus[:3], plus[3:6], plus[6], plus[7], gravity)[0]
            -_flatness_body_rate_squared(
                minus[:3], minus[3:6], minus[6], minus[7], gravity)[0]
        )/(2*epsilon)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)


def test_analytic_whole_body_sample_gradient_matches_finite_difference():
    simulation = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorMINCOConfig(
        max_speed=.2, max_acceleration=.5, max_body_rate=.3,
        max_yaw_rate=.8, max_yaw_acceleration=.5,
        joint_velocity_limits=(.1,)*4, joint_acceleration_limits=(.2,)*4,
        obstacle_clearance=.35, integral_resolution=2)
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -2., -3.], [2., 2., 1.], .2,
        ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits,
        .06, carry_payload=True)
    assert len(evaluator.self_pairs)
    assert len(evaluator.payload_pairs)
    assert evaluator.rrt_fixed_clearance_self_pairs_skipped == 3
    skipped_gripper_pairs = {
        frozenset(("gripper_pad_left", "gripper_pad_right")),
        frozenset(("gripper_finger_left", "gripper_pad_right")),
        frozenset(("gripper_pad_left", "gripper_finger_right")),
    }
    assert not any(frozenset(pair) in skipped_gripper_pairs
                   for pair in evaluator.self_geom_pairs)
    assert all(evaluator.geometry[sphere][0] not in {
        "gripper_palm_body", "gripper_left_body", "gripper_right_body"
    } for sphere, _ in evaluator.payload_pairs)
    qpos_before, qvel_before, time_before = (
        simulation.data.qpos.copy(), simulation.data.qvel.copy(), simulation.time)
    sigma = np.r_[0., 0., -1., .15, 0., 0., 0., 0.]
    sigma[4] = robot.limits.joint_upper[0]+.03
    velocity = np.array([.35, -.03, 0., .4, .25, 0., 0., 0.])
    acceleration = np.array([.7, .1, 0., .8, .3, 0., 0., 0.])
    jerk = np.array([.5, -.2, 0., .1, .1, 0., 0., 0.])
    arguments = [sigma, velocity, acceleration, jerk]
    cost, gradients, _, _ = evaluator.sample_cost_gradient(*arguments)
    epsilon = 1e-6
    for order, index in ((0, 0), (0, 3), (0, 4), (1, 0), (1, 3),
                         (1, 4), (2, 0), (3, 0)):
        plus = [value.copy() for value in arguments]
        minus = [value.copy() for value in arguments]
        plus[order][index] += epsilon
        minus[order][index] -= epsilon
        cost_plus = evaluator.sample_cost_gradient(*plus)[0]
        cost_minus = evaluator.sample_cost_gradient(*minus)[0]
        numerical = (cost_plus-cost_minus)/(2*epsilon)
        np.testing.assert_allclose(gradients[order][index], numerical,
                                   rtol=2e-4, atol=2e-4)
    assert np.isfinite(cost)
    np.testing.assert_array_equal(simulation.data.qpos, qpos_before)
    np.testing.assert_array_equal(simulation.data.qvel, qvel_before)
    assert simulation.time == time_before


def test_complete_shared_total_time_and_minco_adjoint_gradient():
    simulation = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorMINCOConfig(
        integral_resolution=2, max_iterations=2,
        obstacle_clearance=.02, constraint_weight=10.,
        joint_waypoint_parameterization="direct")
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -2., -3.], [2., 2., 1.], .2,
        ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits,
        .06, carry_payload=False)
    start_q = robot.configuration
    from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import quaternion_yaw
    start = np.r_[start_q[:3], quaternion_yaw(start_q[3:7]), start_q[7:11]]
    goal = start.copy(); goal[0] += .12; goal[3] += .03
    boundary = np.zeros((3, 8)); boundary[0] = start
    tail = np.zeros((3, 8)); tail[0] = goal
    minco = MINCOQuintic(boundary, tail, 2)
    midpoint = .5*(start+goal)
    variables = np.r_[midpoint, inverse_time(np.array([1.0]))]
    planner = AerialManipulatorMINCO(config)
    proportions = np.array([.3, .7])
    cost, gradient = planner._objective(
        variables, minco, 2, evaluator, time_proportions=proportions)
    epsilon = 2e-6
    for index in (0, 2, 3, 4, 8):
        plus, minus = variables.copy(), variables.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        numerical = (planner._objective(
                         plus, minco, 2, evaluator, time_proportions=proportions)[0]
                     -planner._objective(
                         minus, minco, 2, evaluator, time_proportions=proportions)[0])/(2*epsilon)
        np.testing.assert_allclose(gradient[index], numerical, rtol=4e-3, atol=3e-2)
    assert np.isfinite(cost)

    durations = proportions*1.4
    coefficients, _ = minco.solve(np.array([midpoint]), durations)
    coefficients = coefficients.reshape(2, 6, 8)
    for derivative in range(4):
        left = sum(coefficients[0, order]*
                   (durations[0]**(order-derivative))*
                   factorial(order)/factorial(order-derivative)
                   for order in range(derivative, 6))
        right = coefficients[1, derivative]*factorial(derivative)
        np.testing.assert_allclose(left, right, atol=2e-8, rtol=2e-8)


def test_flatness_references_keep_quaternion_sign_continuous_across_pi():
    class Robot:
        configuration = np.r_[np.zeros(3), [0., 0., 0., 1.], np.zeros(4), .06]

    robot = Robot()
    yaw_start = np.pi-.04
    robot.configuration[3:7] = [np.cos(yaw_start/2), 0., 0., np.sin(yaw_start/2)]
    coefficients = np.zeros((1, 6, 8))
    coefficients[0, 0, 2] = -1.
    coefficients[0, 0, 3] = yaw_start
    coefficients[0, 1, 3] = .08
    trajectory = AerialManipulatorTrajectory(
        np.array([1.]), coefficients, np.array([[0., 0., -1., yaw_start, 0., 0., 0., 0.],
                                                [0., 0., -1., yaw_start+.08, 0., 0., 0., 0.]]),
        0., 0, True, "test", True, .1, 0., .025)
    first = trajectory.reference(.1, robot, gripper_opening=.06).configuration[3:7]
    second = trajectory.reference(.9, robot, gripper_opening=.06).configuration[3:7]
    assert np.dot(first, second) > 0.
    assert np.dot(second, robot.configuration[3:7]) > 0.


def test_uniform_time_stretch_preserves_path_and_scales_derivatives():
    coefficients = np.zeros((1, 6, 8))
    coefficients[0, 0] = np.array([0., 0., -1., .2, 0., 0., 0., 0.])
    coefficients[0, 1] = np.array([.3, -.1, .2, .4, .1, 0., 0., -.1])
    coefficients[0, 2] = np.array([.1, .05, 0., -.1, .02, 0., .03, 0.])
    path = np.array([[0., 0., -1., .2, 0., 0., 0., 0.],
                     [.4, -.05, -.8, .5, .12, 0., .03, -.1]])
    trajectory = AerialManipulatorTrajectory(
        np.array([1.2]), coefficients, path, 0., 0, True, "test")
    scale = 1.3
    stretched = _time_stretch(trajectory, scale)
    for time in (.0, .2, .7, 1.2):
        np.testing.assert_allclose(stretched.evaluate(scale*time),
                                   trajectory.evaluate(time), atol=1e-12)
        np.testing.assert_allclose(stretched.evaluate(scale*time, 1),
                                   trajectory.evaluate(time, 1)/scale, atol=1e-12)
        np.testing.assert_allclose(stretched.evaluate(scale*time, 2),
                                   trajectory.evaluate(time, 2)/scale**2, atol=1e-12)


def test_initial_duration_accounts_for_acceleration_demand():
    config = AerialManipulatorMINCOConfig(
        max_speed=3., max_acceleration=3., initial_duration_scale=1.2)
    path = np.zeros((2, 8))
    path[1, 0] = 4.
    expected = max(
        np.sqrt(6.0*4./config.max_acceleration),
        config.minimum_total_time+.25)
    assert _initial_duration(path, config) == expected


def test_esdf_out_of_bounds_penalty_gradient_points_back_into_the_map():
    simulation = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorMINCOConfig(obstacle_clearance=.05)
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -2., -3.], [2., 2., 1.], .2, ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits, .06, False)
    sigma = np.r_[2.05, 0., -1., 0., np.zeros(4)]
    value, gradient = evaluator.collision_cost_gradient(sigma)[:2]
    epsilon = 1e-6
    plus, minus = sigma.copy(), sigma.copy()
    plus[0] += epsilon
    minus[0] -= epsilon
    numerical = (evaluator.collision_cost_gradient(plus)[0]
                 -evaluator.collision_cost_gradient(minus)[0])/(2*epsilon)
    assert value > 0.
    np.testing.assert_allclose(gradient[0], numerical, rtol=2e-4, atol=2e-3)


def test_dense_validation_checks_collision_with_flatness_recovered_attitude():
    simulation = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorMINCOConfig(validation_dt=.05)
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -2., -3.], [2., 2., 1.], .2, ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits, .06, False)
    coefficients = np.zeros((1, 6, 8))
    coefficients[0, 0, :3] = [0., 0., -1.]
    coefficients[0, 2, 0] = .1
    trajectory = AerialManipulatorTrajectory(
        np.array([1.]), coefficients,
        np.array([[0., 0., -1., 0., 0., 0., 0., 0.],
                  [.1, 0., -1., 0., 0., 0., 0., 0.]]),
        0., 0, True, "test")
    seen = []
    original = robot.check_collision

    def record(configuration=None, *args, **kwargs):
        seen.append(np.asarray(configuration)[3:7].copy())
        return original(configuration, *args, **kwargs)

    robot.check_collision = record
    evaluator.dense_validate(trajectory)
    robot.check_collision = original
    assert any(np.linalg.norm(quaternion[1:3]) > 1e-3 for quaternion in seen)


def test_collision_gradient_chains_full_pose_through_acceleration_and_yaw():
    simulation = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    origin = np.array([-2., -2., -3.])
    resolution = .2
    axes = [origin[i]+np.arange(36)*resolution for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    field = .1*xx+.2*yy+.3*zz-1.
    esdf = ESDF(field, origin, resolution)
    config = AerialManipulatorMINCOConfig(
        esdf_resolution=resolution, esdf_discretization_margin=np.sqrt(3)*resolution,
        obstacle_clearance=.05)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits, .06, False)
    # Isolate the smooth shared-distance-field term; exact MuJoCo checks are
    # covered by a separate full-pose witness-Jacobian test.
    evaluator.sphere_geom_indices[:] = -1
    evaluator.self_pairs = np.empty((0, 2), dtype=int)
    evaluator.payload_pairs = np.empty((0, 2), dtype=int)
    evaluator.world_geom_pairs = ()
    sigma = np.r_[.2, -.1, -1., .25, .2, -.3, .1, -.15]
    acceleration = np.array([.7, -.4, .5, 0., 0., 0., 0., 0.])
    value, gradient, grad_acceleration = evaluator.collision_cost_gradient(
        sigma, acceleration)[:3]
    assert value > 0.
    assert np.linalg.norm(grad_acceleration[:3]) > 0.
    epsilon = 1e-6
    numerical_state = np.empty(8)
    numerical_acceleration = np.empty(8)
    for index in range(8):
        plus, minus = sigma.copy(), sigma.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        numerical_state[index] = (
            evaluator.collision_cost_gradient(plus, acceleration)[0]
            -evaluator.collision_cost_gradient(minus, acceleration)[0])/(2*epsilon)
        plus_acc, minus_acc = acceleration.copy(), acceleration.copy()
        plus_acc[index] += epsilon; minus_acc[index] -= epsilon
        numerical_acceleration[index] = (
            evaluator.collision_cost_gradient(sigma, plus_acc)[0]
            -evaluator.collision_cost_gradient(sigma, minus_acc)[0])/(2*epsilon)
    np.testing.assert_allclose(gradient, numerical_state, rtol=2e-4, atol=3e-3)
    np.testing.assert_allclose(grad_acceleration, numerical_acceleration,
                               rtol=3e-4, atol=5e-3)


def test_repeated_collision_queries_use_precompiled_geometry_pairs(monkeypatch):
    simulation = MujocoSimulation('tests/fixtures/aerial_manipulator_gradient.xml',
                                  record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorMINCOConfig(esdf_resolution=.2, esdf_discretization_margin=.35)
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -3., -3.], [6., 3., 1.], .2, ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits, .06, False)
    state = np.r_[.8, -.1, -.1, 0., [.2, -.3, .1, -.15]]
    baseline = evaluator.collision_feasible(state)
    assert max(baseline) > 0.

    def unexpected(*args, **kwargs):
        pytest.fail('collision query resolved geometry names again')

    monkeypatch.setattr(robot._model, 'compile_collision_pairs', unexpected)
    assert evaluator.collision_feasible(state) == pytest.approx(baseline)
    cost, *_ = evaluator.collision_cost_gradient(state, np.zeros(8))
    assert np.isfinite(cost)
