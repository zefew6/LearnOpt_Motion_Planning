import numpy as np
import pytest

from uav_ac.rl.assets import ideal_initialization_library
from uav_ac.rl.training import DEFAULT_TRAINING_CONFIG
from uav_ac.rl.trajectory_bank import (
    TrajectoryBank,
    _retime_trajectory,
    sample_open_field_waypoints,
)
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH, MujocoSimulation


def _trajectory(simulation, length, x_offset=0.0):
    trajectory = np.zeros((length, 10))
    trajectory[:, :3] = simulation.start_position
    trajectory[:, 0] += np.linspace(0.0, x_offset, length)
    return trajectory


def test_trajectory_bank_should_round_trip_variable_length_assets(tmp_path):
    simulation = MujocoSimulation(OPEN_FIELD_SCENE_PATH, record_actual_trajectory=False)
    trajectories = [_trajectory(simulation, 4), _trajectory(simulation, 7, 0.3)]
    libraries = [
        ideal_initialization_library(trajectory, simulation.quad)
        for trajectory in trajectories
    ]
    bank = TrajectoryBank.from_sequences(
        trajectories,
        libraries,
        [{"split": "train"}, {"split": "test"}],
    )

    loaded = TrajectoryBank.load(bank.save(tmp_path / "bank"))

    assert len(loaded) == 2
    assert loaded.trajectory(0) == pytest.approx(trajectories[0])
    assert loaded.trajectory(1) == pytest.approx(trajectories[1])
    assert loaded.initialization_library(1).states == pytest.approx(libraries[1].states)
    assert loaded.indices("test") == pytest.approx([1])


@pytest.mark.parametrize(
    "pattern", ("s_curve", "arc", "zigzag", "random_turns", "climb_dive", "sweep"))
def test_open_field_waypoint_sampler_should_be_repeatable_and_bounded(pattern):
    simulation = MujocoSimulation(OPEN_FIELD_SCENE_PATH, record_actual_trajectory=False)
    config = DEFAULT_TRAINING_CONFIG["trajectory_bank"]

    first = sample_open_field_waypoints(
        simulation.start_position, 7, np.random.default_rng(42), config, pattern=pattern)
    second = sample_open_field_waypoints(
        simulation.start_position, 7, np.random.default_rng(42), config, pattern=pattern)

    assert first == pytest.approx(second)
    assert first.shape == (9, 3)
    assert np.all(first[2:, :2] >= np.asarray(config["horizontal_bounds"])[0])
    assert np.all(first[2:, :2] <= np.asarray(config["horizontal_bounds"])[1])
    assert np.all(-first[2:, 2] >= config["altitude"][0])
    assert np.all(-first[2:, 2] <= config["altitude"][1])


def test_retime_should_set_average_speed_and_scale_derivatives():
    trajectory = np.zeros((101, 10))
    trajectory[:, 0] = np.linspace(0.0, 10.0, len(trajectory))
    trajectory[:, 3] = 10.0
    trajectory[:, 6] = 2.0

    retimed = _retime_trajectory(trajectory, 0.01, target_average_speed=5.0)

    duration = (len(retimed) - 1) * 0.01
    assert 10.0 / duration == pytest.approx(5.0)
    assert retimed[:, 3] == pytest.approx(trajectory[0, 3] * 0.5)
    assert retimed[:, 6] == pytest.approx(trajectory[0, 6] * 0.25)
