from unittest.mock import Mock

import numpy as np

from uav_ac.planning.pipeline import mission_planner


def test_minimum_snap_mission_solves_the_full_waypoint_sequence(monkeypatch):
    waypoints = np.array([[0., 0., -1.], [2., 0., -1.], [3., 2., -1.]])
    obstacles = np.empty((0, 6))
    trajectory = np.zeros((3, 11))
    solver = Mock()
    solver.get_trajectory.return_value = trajectory
    constructor = Mock(return_value=solver)
    monkeypatch.setattr(mission_planner, "MinimumSnap", constructor)

    result = mission_planner.generate_minimum_snap_mission(
        waypoints, obstacles, velocity=3.0, dt=0.01)

    assert result is trajectory
    constructor.assert_called_once_with(waypoints, obstacles, 3.0, 0.01)
    solver.get_trajectory.assert_called_once_with()
