import numpy as np
import pytest

from uav_ac.control.rl_controller import (
    OBSERVATION_SIZE,
    RLController,
    action_to_command,
    encode_observation,
)
from uav_ac.control.trajectory_controller import TrajectoryReference
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def _yaw_quaternion(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _reference(trajectory, index=0, *, is_control_tick=True):
    return TrajectoryReference(
        trajectory, index, 0.01,
        is_terminal=index == len(trajectory) - 1,
        is_control_tick=is_control_tick,
    )


def test_rl_observation_should_have_fixed_shape_dtype_and_bounds():
    simulation = MujocoSimulation(record_actual_trajectory=False)
    trajectory = np.zeros((30, 10))
    trajectory[:, :3] = simulation.start_position
    trajectory[:, 6:9] = 100.0

    observation = encode_observation(
        simulation.quad, _reference(trajectory), np.ones(4))

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -5.0)
    assert np.all(observation <= 5.0)


def test_rl_observation_should_be_invariant_to_common_ned_yaw_rotation():
    first = MujocoSimulation(record_actual_trajectory=False).quad
    second = MujocoSimulation(record_actual_trajectory=False).quad
    first.X[:3] = np.array([2.0, 5.0, -1.0])
    first.X[3:7] = _yaw_quaternion(0.3)
    first.X[7:10] = np.array([0.4, -0.2, 0.1])
    first.X[10:13] = np.array([0.1, 0.2, -0.1])
    first.omega[:] = 0.8
    trajectory = np.zeros((30, 10))
    trajectory[:, :3] = np.array([3.0, 4.0, -1.2])
    trajectory[:, 3:6] = np.array([0.7, 0.1, -0.2])
    trajectory[:, 6:9] = np.array([0.2, -0.3, 0.1])
    trajectory[:, 9] = 0.2
    trajectory[5:, :2] += np.array([1.0, -0.5])

    angle = 0.7
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    second.X[:] = first.X
    second.X[:2] = rotation @ first.X[:2]
    second.X[3:7] = _yaw_quaternion(0.3 + angle)
    second.X[7:9] = rotation @ first.X[7:9]
    second.omega[:] = first.omega
    rotated_trajectory = trajectory.copy()
    rotated_trajectory[:, :2] = trajectory[:, :2] @ rotation.T
    rotated_trajectory[:, 3:5] = trajectory[:, 3:5] @ rotation.T
    rotated_trajectory[:, 6:8] = trajectory[:, 6:8] @ rotation.T
    rotated_trajectory[:, 9] += angle

    first_observation = encode_observation(first, _reference(trajectory), np.zeros(4))
    second_observation = encode_observation(
        second, _reference(rotated_trajectory), np.zeros(4))

    assert second_observation == pytest.approx(first_observation, abs=1e-6)


def test_rl_action_zero_should_mean_hover_and_extremes_should_reach_thrust_limits():
    quad = MujocoSimulation(record_actual_trajectory=False).quad

    hover = action_to_command(np.zeros(4), quad)
    minimum = action_to_command(np.array([-1.0, 0.0, 0.0, 0.0]), quad)
    maximum = action_to_command(np.array([1.0, 0.0, 0.0, 0.0]), quad)

    assert hover.thrust == pytest.approx(quad.m * quad.g)
    assert hover.moment == pytest.approx(np.zeros(3))
    assert minimum.thrust == pytest.approx(4.0 * quad.min_thrust)
    assert maximum.thrust == pytest.approx(4.0 * quad.max_thrust)


class _CountingPolicy:
    def __init__(self):
        self.call_count = 0

    def predict(self, observation, **kwargs):
        self.call_count += 1
        return np.array([0.1, 0.2, -0.3, 0.4]), None


def test_rl_controller_should_predict_only_on_control_ticks_and_hold_command():
    simulation = MujocoSimulation(record_actual_trajectory=False)
    trajectory = np.zeros((2, 10))
    trajectory[:, :3] = simulation.start_position
    policy = _CountingPolicy()
    controller = RLController(
        policy, simulation.quad, control_dt=0.01, steps_per_action=10)

    first = controller.step(simulation.quad, _reference(trajectory, is_control_tick=True))
    held = controller.step(simulation.quad, _reference(trajectory, is_control_tick=False))

    assert policy.call_count == 1
    assert held.thrust == pytest.approx(first.thrust)
    assert held.moment == pytest.approx(first.moment)
    controller.reset()
    assert controller._previous_action == pytest.approx(np.zeros(4))
    assert controller._policy_state is None
