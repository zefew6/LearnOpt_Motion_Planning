import numpy as np

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.trajectory.gcopter.aerial_manipulator.config import (
    AerialManipulatorGCOPTERConfig,
)
from uav_ac.planning.trajectory.gcopter.aerial_manipulator.evaluator import (
    AerialManipulatorTrajectoryEvaluator,
)
from uav_ac.planning.trajectory.gcopter.aerial_manipulator.planner import (
    AerialManipulatorGCOPTER,
)
from uav_ac.planning.trajectory.gcopter.aerial_manipulator.types import (
    AerialManipulatorTrajectory,
)
from uav_ac.planning.trajectory.gcopter.mappings import inverse_time
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_analytic_whole_body_sample_gradient_matches_finite_difference():
    simulation = MujocoSimulation(
        "uav_ac/simulation/models/aerial_manipulator_pick_place.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorGCOPTERConfig(
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
    assert evaluator.self_pairs
    assert evaluator.payload_pairs
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
        "uav_ac/simulation/models/aerial_manipulator_pick_place.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    config = AerialManipulatorGCOPTERConfig(
        pieces=2, integral_resolution=2, max_iterations=2,
        obstacle_clearance=.02, constraint_weight=10.)
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -2., -3.], [2., 2., 1.], .2,
        ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, config, simulation.space_limits,
        .06, carry_payload=False)
    start_q = robot.configuration
    from uav_ac.planning.trajectory.gcopter.aerial_manipulator.task_targets import quaternion_yaw
    start = np.r_[start_q[:3], quaternion_yaw(start_q[3:7]), start_q[7:11]]
    goal = start.copy(); goal[0] += .12; goal[3] += .03
    boundary = np.zeros((3, 8)); boundary[0] = start
    tail = np.zeros((3, 8)); tail[0] = goal
    minco = MINCOQuintic(boundary, tail, 2)
    midpoint = .5*(start+goal)
    variables = np.r_[midpoint, inverse_time(np.array([1.0]))]
    planner = AerialManipulatorGCOPTER(config)
    cost, gradient = planner._objective(variables, minco, 2, evaluator)
    epsilon = 2e-6
    for index in (0, 2, 3, 4, 8):
        plus, minus = variables.copy(), variables.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        numerical = (planner._objective(plus, minco, 2, evaluator)[0]
                     -planner._objective(minus, minco, 2, evaluator)[0])/(2*epsilon)
        np.testing.assert_allclose(gradient[index], numerical, rtol=4e-3, atol=3e-2)
    assert np.isfinite(cost)


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
