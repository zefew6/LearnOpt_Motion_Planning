import numpy as np

from uav_ac.control.cascaded_controller import CascadedController
from uav_ac.control.aerial_manipulator_controller import (
    AerialManipulatorController,
)
from uav_ac.robot.aerial_manipulator import AerialManipulatorReference
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def _arm_reference(time, amplitudes, frequency):
    phase = frequency * max(time - 4.0, 0.0)
    cosine, sine = np.cos(phase), np.sin(phase)
    position = amplitudes * (1.0 - cosine) ** 2 / 4.0
    velocity = amplitudes * frequency * (1.0 - cosine) * sine / 2.0
    acceleration = amplitudes * frequency**2 * (
        sine**2 + (1.0 - cosine) * cosine) / 2.0
    return position, velocity, acceleration


def test_whole_body_controller_tracks_arm_while_holding_hover_in_fixture():
    simulation = MujocoSimulation(
        "tests/fixtures/aerial_manipulator.xml", record_actual_trajectory=False)
    simulation.set_allowed_contact_pairs((
        ("target_table", "graspable_object"),))
    robot = simulation.robot
    initial = robot.configuration.copy()
    initial[11] = 0.045
    robot.reset(initial)
    flight_controller = CascadedController(simulation.quad.g, simulation.quad.dt)
    controller = AerialManipulatorController(flight_controller, robot, simulation.quad)
    amplitudes = np.array([0.15, 0.4, -0.35, 0.25])
    frequency = 1.2
    peak_position_error = 0.0
    peak_tilt = 0.0
    peak_joint_error = 0.0

    for _ in range(20_000):
        desired, desired_velocity, desired_acceleration = _arm_reference(
            simulation.time, amplitudes, frequency)
        configuration = initial.copy()
        configuration[7:11] = desired
        reference = AerialManipulatorReference(
            configuration,
            np.r_[np.zeros(6), desired_velocity, 0.0],
            np.r_[np.zeros(6), desired_acceleration, 0.0],
        )
        robot.apply(controller.step(reference))
        simulation.step()
        peak_position_error = max(
            peak_position_error,
            float(np.linalg.norm(robot.configuration[:3] - initial[:3])))
        peak_tilt = max(peak_tilt, float(np.max(np.abs(simulation.quad.euler_angles[:2]))))
        if simulation.time >= 5.0:
            peak_joint_error = max(
                peak_joint_error,
                float(np.max(np.abs(desired - robot.state.joint_positions))))
        if simulation.collision_detected:
            break

    assert not simulation.collision_detected
    assert peak_position_error < 0.05
    assert peak_tilt < 0.174533
    assert peak_joint_error < 0.03
    assert controller.saturation_count == 0
