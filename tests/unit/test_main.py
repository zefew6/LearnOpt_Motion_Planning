import numpy as np
import pytest

from uav_ac.main import (
    _generate_mission_trajectory,
    _trajectory_after_takeoff,
)
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_trajectory_after_takeoff_should_hide_vertical_departure_segment():
    # Arrange
    trajectory = np.zeros((5, 10))
    trajectory[:, :3] = np.array([
        [1.0, 7.0, -0.021],
        [1.0, 7.0, -0.7],
        [1.0, 7.0, -1.3],
        [2.0, 7.0, -1.3],
        [3.0, 6.0, -1.5],
    ])
    takeoff_waypoint = np.array([1.0, 7.0, -1.3])

    # Act
    visible_trajectory = _trajectory_after_takeoff(trajectory, takeoff_waypoint)

    # Assert
    assert visible_trajectory[:, :3] == pytest.approx(trajectory[2:, :3])


def test_generate_mission_trajectory_should_keep_takeoff_vertical_and_above_ground():
    # Arrange
    simulation = MujocoSimulation()

    # Act
    trajectory = _generate_mission_trajectory(
        simulation.mission_waypoints,
        simulation.obstacles,
        velocity=2.0,
        dt=simulation.quad.dt * 10,
    )
    takeoff_index = np.argmin(np.linalg.norm(
        trajectory[:, :3] - simulation.mission_waypoints[1], axis=1))
    takeoff_positions = trajectory[:takeoff_index + 1, :3]

    # Assert
    assert takeoff_positions[:, :2] == pytest.approx(
        np.repeat(simulation.start_position[np.newaxis, :2], len(takeoff_positions), axis=0))
    assert np.all(trajectory[:, 2] <= 0.0)
