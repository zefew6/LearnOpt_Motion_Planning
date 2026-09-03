import numpy as np
import pytest

from uav_ac.simulation.wind_disturb import GustingCrosswind


def test_gusting_crosswind_should_be_repeatable_and_bounded():
    # Arrange
    wind = GustingCrosswind()
    times = np.linspace(0.0, 30.0, 301)

    # Act
    forces = np.array([wind.force_ned(time) for time in times])

    # Assert
    assert wind.force_ned(3.2) == pytest.approx(wind.force_ned(3.2))
    steady = np.asarray(wind.steady_force)
    amplitude = np.asarray(wind.gust_force)
    assert np.all(forces >= steady - amplitude - 1.0e-12)
    assert np.all(forces <= steady + amplitude + 1.0e-12)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"steady_force": (0.0, 0.0)}, "steady_force"),
        ({"gust_force": (0.1, -0.1, 0.1)}, "gust_force"),
        ({"angular_frequency": (1.0, np.nan, 1.0)}, "angular_frequency"),
    ],
)
def test_gusting_crosswind_should_reject_invalid_parameters(arguments, message):
    # Arrange
    # Act
    def create_invalid_wind():
        GustingCrosswind(**arguments)

    # Assert
    with pytest.raises(ValueError, match=message):
        create_invalid_wind()


@pytest.mark.parametrize("time", [-0.1, np.nan])
def test_gusting_crosswind_should_reject_invalid_time(time):
    # Arrange
    wind = GustingCrosswind()

    # Act
    def evaluate_invalid_time():
        wind.force_ned(time)

    # Assert
    with pytest.raises(ValueError, match="time"):
        evaluate_invalid_time()
