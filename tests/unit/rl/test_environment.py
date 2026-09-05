import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from uav_ac.rl.environment import MujocoTrajectoryTrackingEnv
from uav_ac.rl.assets import ideal_initialization_library
from uav_ac.rl.trajectory_bank import TrajectoryBank
from uav_ac.simulation.mujoco_sim import ENU_TO_NED, OPEN_FIELD_SCENE_PATH, MujocoSimulation
from uav_ac.simulation.wind_disturb import RandomWindConfig


def _hover_trajectory(length=40):
    simulation = MujocoSimulation(record_actual_trajectory=False)
    trajectory = np.zeros((length, 10))
    trajectory[:, :3] = simulation.start_position
    return trajectory


def test_rl_environment_should_pass_gymnasium_checker():
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(), random_start=False, perturb_initial_state=False)

    check_env(environment)


def test_rl_environment_reward_should_report_its_complete_decomposition():
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)

    _, reward, _, _, info = environment.step(np.array([0.1, -0.2, 0.3, -0.4]))

    expected = info["tracking_reward"] - info["action_cost"] - info["action_delta_cost"]
    assert reward == pytest.approx(expected)
    assert info["action_cost"] == pytest.approx(0.005 * 0.3)
    assert info["action_delta_cost"] == pytest.approx(0.01 * 0.3)


def test_rl_environment_should_terminate_with_success_bonus_at_final_reference():
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(1), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)

    _, reward, terminated, truncated, info = environment.step(np.zeros(4))

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert reward > 20.0


@pytest.mark.parametrize(
    ("state_update", "failure_reason"),
    [
        (lambda state: state.__setitem__(0, 4.0), "position_error"),
    ],
)
def test_rl_environment_should_terminate_for_position_failures(state_update, failure_reason):
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)
    state = environment.quad.X.copy()
    state_update(state)
    environment.simulation.reset(state, environment.quad.omega)

    _, reward, terminated, truncated, info = environment.step(np.zeros(4))

    assert terminated is True
    assert truncated is False
    assert info["failure_reason"] == failure_reason
    assert reward < -19.0


def test_rl_environment_should_terminate_out_of_bounds(monkeypatch):
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)
    monkeypatch.setattr(environment, "_is_in_bounds", lambda: False)

    _, _, terminated, truncated, info = environment.step(np.zeros(4))

    assert terminated is True
    assert truncated is False
    assert info["failure_reason"] == "out_of_bounds"


def test_rl_environment_should_terminate_for_excessive_tilt():
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)
    state = environment.quad.X.copy()
    angle = np.deg2rad(80.0)
    state[3:7] = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
    environment.simulation.reset(state, environment.quad.omega)

    _, _, terminated, _, info = environment.step(np.zeros(4))

    assert terminated is True
    assert info["failure_reason"] == "tilt"


def test_rl_environment_should_return_valid_observation_for_nonfinite_failure(monkeypatch):
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)

    def make_state_nonfinite():
        environment.quad.X[0] = np.nan
        return environment.quad.X.copy()

    monkeypatch.setattr(environment.simulation, "step", make_state_nonfinite)
    observation, _, terminated, _, info = environment.step(np.zeros(4))

    assert terminated is True
    assert info["failure_reason"] == "non_finite_state"
    assert environment.observation_space.contains(observation)


def test_rl_environment_should_truncate_after_one_second_at_unsatisfied_final_reference():
    environment = MujocoTrajectoryTrackingEnv(
        _hover_trajectory(1), random_start=False, perturb_initial_state=False)
    environment.reset(seed=42)
    state = environment.quad.X.copy()
    yaw = np.deg2rad(30.0)
    state[3:7] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    environment.simulation.reset(state, environment.quad.omega)

    for _ in range(100):
        _, _, terminated, truncated, info = environment.step(np.zeros(4))
        if terminated or truncated:
            break

    assert terminated is False
    assert truncated is True
    assert info["success"] is False


def _small_trajectory_bank():
    simulation = MujocoSimulation(OPEN_FIELD_SCENE_PATH, record_actual_trajectory=False)
    trajectories = []
    for offset in (0.0, 0.2, 0.4):
        trajectory = np.zeros((20, 10))
        trajectory[:, :3] = simulation.start_position
        trajectory[:, 0] += np.linspace(0.0, offset, len(trajectory))
        trajectories.append(trajectory)
    libraries = [
        ideal_initialization_library(trajectory, simulation.quad)
        for trajectory in trajectories
    ]
    return TrajectoryBank.from_sequences(
        trajectories,
        libraries,
        [
            {"split": "train", "average_speed": 2.5},
            {"split": "train", "average_speed": 3.5},
            {"split": "test", "average_speed": 4.5},
        ],
    )


def test_rl_environment_should_select_only_requested_bank_split():
    environment = MujocoTrajectoryTrackingEnv(
        _small_trajectory_bank(),
        model_path=OPEN_FIELD_SCENE_PATH,
        split="train",
        random_start=False,
        perturb_initial_state=False,
    )

    selected = {
        environment.reset(seed=seed)[1]["trajectory_id"] for seed in range(20)
    }

    assert selected == {0, 1}
    _, info = environment.reset(seed=1, options={"trajectory_id": 1})
    assert info["trajectory_id"] == 1
    assert info["trajectory_split"] == "train"
    assert info["trajectory_average_speed"] == pytest.approx(3.5)


def test_rl_environment_should_apply_seeded_episode_wind_in_ned():
    environment = MujocoTrajectoryTrackingEnv(
        _small_trajectory_bank(),
        model_path=OPEN_FIELD_SCENE_PATH,
        split="train",
        random_start=False,
        perturb_initial_state=False,
        curriculum_progress=1.0,
        wind_config=RandomWindConfig(probability=1.0),
    )
    environment.reset(seed=42, options={
        "trajectory_id": 0, "wind_enabled": True, "wind_scale": 1.0})

    _, _, _, _, info = environment.step(np.zeros(4))

    assert info["wind_enabled"] is True
    assert info["maximum_wind_force"] > 0.0
    assert environment.simulation._external_force_world == pytest.approx(
        ENU_TO_NED @ info["wind_force_ned"])
