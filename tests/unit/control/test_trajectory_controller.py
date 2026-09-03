import numpy as np
import pytest

from uav_ac.control import CascadedController, TrajectoryController
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_trajectory_controller_reset_should_clear_tracking_state():
    # Arrange
    simulation = MujocoSimulation()
    controller = CascadedController(simulation.quad.g, simulation.quad.dt * 10)
    tracker = TrajectoryController(
        controller, simulation.quad, np.zeros((2, 10)),
        steps_per_reference=10)
    tracker.trajectory_index = 1
    tracker.inner_step = 12
    controller._thrust_cmd = 4.0
    controller._pqr_cmd[:] = 1.0
    controller.integral_error = 2.0

    # Act
    tracker.reset()

    # Assert
    assert tracker.trajectory_index == 0
    assert tracker.inner_step == 0
    assert controller._thrust_cmd == pytest.approx(0.0)
    assert controller._pqr_cmd == pytest.approx(np.zeros(3))
    assert controller.integral_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("trajectory", "frequency", "message"),
    [
        (np.zeros((2, 9)), 10, "trajectory"),
        (np.full((2, 10), np.nan), 10, "finite"),
        (np.zeros((2, 10)), 0, "steps_per_reference"),
    ],
)
def test_trajectory_controller_should_reject_invalid_configuration(
        trajectory, frequency, message):
    # Arrange
    simulation = MujocoSimulation()
    controller = CascadedController(simulation.quad.g, simulation.quad.dt * 10)

    # Act
    def create_invalid_tracker():
        TrajectoryController(
            controller, simulation.quad, trajectory,
            steps_per_reference=frequency)

    # Assert
    with pytest.raises(ValueError, match=message):
        create_invalid_tracker()
