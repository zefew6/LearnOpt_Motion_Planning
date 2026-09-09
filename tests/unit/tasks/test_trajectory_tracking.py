import numpy as np
import pytest

from uav_ac.control import CascadedController, TrajectoryController
from uav_ac.rl.common.environment import MujocoTrajectoryTrackingEnv as LegacyEnv
from uav_ac.simulation.mujoco_sim import ENU_TO_NED, MujocoSimulation
from uav_ac.simulation.wind_disturb import GustingCrosswind, RandomWindConfig
from uav_ac.tasks.trajectory_tracking import MujocoTrajectoryTrackingEnv


def make_env(**kwargs):
    trajectory = np.zeros((40, 10))
    trajectory[:, :3] = [1.0, 7.0, -1.3]
    return MujocoTrajectoryTrackingEnv(
        trajectory, random_start=False, perturb_initial_state=False, **kwargs)


def test_legacy_import_is_the_task_owned_class():
    assert LegacyEnv is MujocoTrajectoryTrackingEnv
    assert LegacyEnv.__module__ == "uav_ac.tasks.trajectory_tracking"


def test_step_controller_matches_cascaded_physics_substep_scheduler():
    environment = make_env()
    environment.reset(seed=3)
    simulation = MujocoSimulation(record_actual_trajectory=False)
    simulation.reset(environment.quad.X, environment.quad.omega)
    controller = CascadedController(environment.quad.g, environment.control_dt)
    scheduler = TrajectoryController(
        CascadedController(simulation.quad.g, environment.control_dt), simulation.quad,
        environment.trajectory, environment.steps_per_action)
    calls = []

    class RecordingController:
        def step(self, quad, reference):
            calls.append((reference.index, reference.is_control_tick))
            return controller.step(quad, reference)

    for interval in range(3):
        result = environment.step_controller(RecordingController())
        for _ in range(environment.steps_per_action):
            scheduler.step()
            simulation.step()
        np.testing.assert_allclose(environment.quad.X, simulation.quad.X, atol=1e-12)
        assert len(result) == 5
        assert environment.reference.index == interval + 1
    assert calls == [(i, j == 0) for i in range(3) for j in range(10)]


def test_fixed_wind_is_applied_in_world_frame_and_resets():
    wind = GustingCrosswind()
    environment = make_env(fixed_wind=wind)
    environment.reset(seed=2)
    _, _, _, _, info = environment.step(np.zeros(4))
    time = environment.simulation.data.time - environment.quad.dt
    np.testing.assert_allclose(info["wind_force_ned"], wind.force_ned(time))
    np.testing.assert_allclose(environment.simulation._external_force_world,
                               ENU_TO_NED @ wind.force_ned(time))
    assert info["wind_enabled"]
    environment.reset(options={"wind_enabled": False})
    assert not environment.step(np.zeros(4))[4]["wind_enabled"]


def test_wind_rng_is_independent_of_episode_rng_and_seed():
    config = RandomWindConfig(probability=1.0)
    first = make_env(wind_config=config, wind_seed=23, curriculum_progress=1.0)
    second = make_env(wind_config=config, wind_seed=23, curriculum_progress=1.0)
    calm = make_env()
    first.reset(seed=4)
    second.reset(seed=9)
    calm.reset(seed=4)
    assert first._wind == second._wind
    np.testing.assert_array_equal(first.np_random.random(10), calm.np_random.random(10))


def test_success_position_tolerance_is_configurable():
    strict = make_env(success_position_error=0.1)
    relaxed = make_env(success_position_error=0.3)
    metrics = {"position_error": 0.2, "velocity_error": 0.0, "tilt": 0.0, "yaw_error": 0.0}
    for environment in (strict, relaxed):
        environment._reference_index = len(environment.trajectory) - 1
    assert not strict._is_success(metrics)
    assert relaxed._is_success(metrics)


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf])
def test_invalid_success_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="success_position_error"):
        make_env(success_position_error=tolerance)


def test_fixed_and_random_wind_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_env(fixed_wind=GustingCrosswind(), wind_config=RandomWindConfig())
