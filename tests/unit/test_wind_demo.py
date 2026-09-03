import numpy as np
import pytest

from uav_ac.wind import _trajectory_after_takeoff


def test_wind_demo_trajectory_should_hide_vertical_takeoff():
    # Arrange
    trajectory = np.zeros((4, 10))
    trajectory[:, :3] = np.array([
        [1.0, 7.0, -0.02],
        [1.0, 7.0, -0.8],
        [1.0, 7.0, -1.3],
        [2.0, 7.0, -1.3],
    ])

    # Act
    visible = _trajectory_after_takeoff(
        trajectory, np.array([1.0, 7.0, -1.3]))

    # Assert
    assert visible == pytest.approx(trajectory[2:])
