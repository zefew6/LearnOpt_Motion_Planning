import numpy as np
import pytest

from uav_ac.main import (
    _generate_mission_trajectory,
    _plan_trajectory,
    _sample_open_field_mission,
    _trajectory_after_takeoff,
    _wind_control_callbacks,
)
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH, MujocoSimulation


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


def test_default_planning_waypoints_should_start_at_current_vehicle_position(monkeypatch):
    simulation = MujocoSimulation(record_actual_trajectory=False)
    captured = {}

    def capture_waypoints(waypoints, obstacles, velocity, dt):
        captured["waypoints"] = waypoints.copy()
        return np.zeros((2, 10))

    monkeypatch.setattr("uav_ac.main._generate_mission_trajectory", capture_waypoints)
    simulation.mission_waypoints[0] += np.array([0.2, -0.1, 0.0])

    _plan_trajectory("mini_snap", simulation, 2.0, 0.01)

    assert captured["waypoints"][0] == pytest.approx(simulation.start_position)


def test_open_field_main_mission_should_be_seeded_and_not_use_scene_waypoints():
    simulation = MujocoSimulation(OPEN_FIELD_SCENE_PATH, record_actual_trajectory=False)

    first = _sample_open_field_mission(simulation, 42)
    second = _sample_open_field_mission(simulation, 42)

    assert first == pytest.approx(second)
    assert 5 <= len(first) - 2 <= 10
    assert not np.allclose(first[-1], simulation.goal_position)


def test_main_wind_wrapper_should_apply_and_clear_external_force():
    simulation = MujocoSimulation(record_actual_trajectory=False)

    class Controller:
        def step(self):
            pass

        def reset(self):
            pass

    step, reset = _wind_control_callbacks(simulation, Controller(), True)

    step()
    assert np.linalg.norm(simulation._external_force_world) > 0.0
    reset()
    assert simulation._external_force_world == pytest.approx(np.zeros(3))
