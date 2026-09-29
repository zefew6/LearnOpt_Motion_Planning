"""Regression coverage for MINCO's internal joint-waypoint coordinates."""

import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import (
    AerialManipulatorMINCOConfig,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.evaluator import (
    AerialManipulatorTrajectoryEvaluator,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.planner import (
    AerialManipulatorMINCO,
    _decode_internal_joint_waypoints,
    _decode_tanh_joint_waypoints,
    _encode_internal_joint_waypoints,
    _tanh_joint_waypoint_derivative,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import quaternion_yaw
from uav_ac.planning.trajectory.gcopter.mappings import inverse_time
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_tanh_joint_waypoint_mapping_keeps_internal_knots_strictly_inside_limits():
    lower = np.array([-2.8, -2.2, -2.4, -2.4])
    upper = np.array([2.8, 2.2, 2.4, 2.4])
    physical = np.array([
        [.1, -.2, -1.1, .3, *lower],
        [.2, .3, -1.2, -.4, *upper],
        [-.3, .1, -.9, .2, .4, -.3, .2, -.1],
    ])

    raw = _encode_internal_joint_waypoints(physical, "tanh", lower, upper)
    decoded = _decode_internal_joint_waypoints(raw, "tanh", lower, upper)

    np.testing.assert_array_equal(decoded[:, :4], physical[:, :4])
    assert np.all(decoded[:, 4:8] > lower)
    assert np.all(decoded[:, 4:8] < upper)
    np.testing.assert_allclose(decoded[0, 4:8], lower+1.0e-6, atol=2e-15)
    np.testing.assert_allclose(decoded[1, 4:8], upper-1.0e-6, atol=2e-15)
    np.testing.assert_allclose(decoded[2, 4:8], physical[2, 4:8], atol=1e-12)

    epsilon = 1.0e-6
    numerical = np.empty(4)
    for joint in range(4):
        plus, minus = raw[2, 4:8].copy(), raw[2, 4:8].copy()
        plus[joint] += epsilon
        minus[joint] -= epsilon
        numerical[joint] = (
            _decode_tanh_joint_waypoints(plus, lower, upper)[joint]
            - _decode_tanh_joint_waypoints(minus, lower, upper)[joint])/(2*epsilon)
    np.testing.assert_allclose(
        _tanh_joint_waypoint_derivative(raw[2, 4:8], lower, upper),
        numerical, rtol=2e-7, atol=2e-7)

    outside = physical.copy()
    outside[2, 4] = lower[0]-2.0e-6
    with pytest.raises(ValueError, match="exceeds"):
        _encode_internal_joint_waypoints(outside, "tanh", lower, upper)


def test_joint_waypoint_parameterization_and_esdf_grid_configuration_are_validated():
    config = AerialManipulatorMINCOConfig()
    assert config.joint_waypoint_parameterization == "tanh"
    assert config.esdf_resolution == pytest.approx(.02)
    assert config.esdf_discretization_margin >= np.sqrt(3.0)*config.esdf_resolution
    with pytest.raises(ValueError, match="joint_waypoint_parameterization"):
        AerialManipulatorMINCOConfig(joint_waypoint_parameterization="clip")
    with pytest.raises(ValueError, match="one voxel diagonal"):
        AerialManipulatorMINCOConfig(
            esdf_resolution=.02, esdf_discretization_margin=.034)


def test_tanh_objective_gradient_is_the_minco_adjoint_gradient_times_mapping_jacobian():
    simulation = MujocoSimulation(
        "uav_ac/simulation/models/aerial_manipulator_pick_place.xml",
        record_actual_trajectory=False)
    robot = simulation.robot
    direct_config = AerialManipulatorMINCOConfig(
        integral_resolution=2, max_iterations=2,
        obstacle_clearance=.02, constraint_weight=10.,
        joint_waypoint_parameterization="direct")
    tanh_config = AerialManipulatorMINCOConfig(
        integral_resolution=2, max_iterations=2,
        obstacle_clearance=.02, constraint_weight=10.)
    esdf = ESDF.from_axis_aligned_boxes(
        np.empty((0, 6)), [-2., -2., -3.], [2., 2., 1.], .2,
        ground_height=0.)
    evaluator = AerialManipulatorTrajectoryEvaluator(
        robot, esdf, simulation.quad, direct_config, simulation.space_limits,
        .06, carry_payload=False)

    configuration = robot.configuration
    start = np.r_[configuration[:3], quaternion_yaw(configuration[3:7]),
                  configuration[7:11]]
    goal = start.copy()
    goal[0] += .12
    goal[3] += .03
    head, tail = np.zeros((3, 8)), np.zeros((3, 8))
    head[0], tail[0] = start, goal
    minco = MINCOQuintic(head, tail, 2)
    physical_midpoint = .5*(start+goal)
    physical_midpoint[4:8] = [.25, -.2, .35, -.3]
    tau = inverse_time(np.array([1.0]))
    direct_variables = np.r_[physical_midpoint, tau]
    lower, upper = robot.limits.joint_lower, robot.limits.joint_upper
    tanh_variables = np.r_[
        _encode_internal_joint_waypoints(
            physical_midpoint[None], "tanh", lower, upper).reshape(-1), tau]

    direct_planner = AerialManipulatorMINCO(direct_config)
    tanh_planner = AerialManipulatorMINCO(tanh_config)
    direct_cost, direct_gradient = direct_planner._objective(
        direct_variables, minco, 2, evaluator)
    tanh_cost, tanh_gradient = tanh_planner._objective(
        tanh_variables, minco, 2, evaluator)

    np.testing.assert_allclose(tanh_cost, direct_cost, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(tanh_gradient[:4], direct_gradient[:4], rtol=2e-10,
                               atol=2e-10)
    np.testing.assert_allclose(
        tanh_gradient[4:8],
        direct_gradient[4:8]*_tanh_joint_waypoint_derivative(
            tanh_variables[4:8], lower, upper),
        rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(tanh_gradient[-1], direct_gradient[-1], rtol=2e-10,
                               atol=2e-10)

    epsilon = 2.0e-6
    for index in (4, 7):
        plus, minus = tanh_variables.copy(), tanh_variables.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (
            tanh_planner._objective(plus, minco, 2, evaluator)[0]
            - tanh_planner._objective(minus, minco, 2, evaluator)[0])/(2*epsilon)
        np.testing.assert_allclose(tanh_gradient[index], numerical,
                                   rtol=4e-3, atol=3e-2)
