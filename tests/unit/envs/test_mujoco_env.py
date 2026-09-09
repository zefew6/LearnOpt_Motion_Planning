import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from uav_ac.envs import MujocoEnv
from uav_ac.scenes import load_scene
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH, MujocoSimulation
from uav_ac.tasks.trajectory_tracking import MujocoTrajectoryTrackingEnv


@pytest.fixture
def vehicle_scene(tmp_path):
    """Keep only the vehicle, ground and physical numerics from a real scene."""
    root = ET.parse(OPEN_FIELD_SCENE_PATH).getroot()
    world = root.find("worldbody")
    for element in list(world):
        if element.get("name") not in {"ground", "quadrotor"}:
            world.remove(element)
    custom = root.find("custom")
    for element in list(custom):
        if element.get("name") == "planning_bounds":
            custom.remove(element)
    path = tmp_path / "vehicle.xml"
    ET.ElementTree(root).write(path)
    return path


class AltitudeTask:
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)

    def reset(self, simulation, rng, options):
        state = simulation.quad.X.copy()
        state[2] = -float(options.get("altitude", rng.uniform(1.0, 2.0)))
        simulation.reset(state)
        return {"altitude": -state[2]}

    def observation(self, simulation):
        return np.array([-simulation.quad.z], dtype=np.float32)

    def reward(self, simulation, action):
        return -abs(-simulation.quad.z - 1.5)

    def terminated(self, simulation):
        return simulation.data.time >= 0.02


def test_nontrajectory_task_passes_gym_checker(vehicle_scene):
    environment = MujocoEnv(AltitudeTask(), model_path=vehicle_scene)
    check_env(environment)
    observation, info = environment.reset(seed=4, options={"altitude": 1.5})
    assert observation == pytest.approx([1.5])
    assert info == {"altitude": 1.5}
    _, reward, terminated, truncated, _ = environment.step(np.zeros(4))
    assert reward == pytest.approx(-abs(-environment.quad.z - 1.5))
    assert not terminated and not truncated
    assert environment.simulation.data.time == pytest.approx(environment.control_dt)
    assert environment.step(np.zeros(4))[2] is True


def test_optional_metadata_and_visualization_allow_vehicle_only_scene(vehicle_scene):
    scene = load_scene(vehicle_scene)
    assert scene.model_path == vehicle_scene.resolve()
    assert scene.goal_position is None
    assert scene.space_limits is None
    assert scene.mission_waypoints.shape == (0, 3)
    assert scene.obstacles.shape == (0, 6)
    simulation = MujocoSimulation(vehicle_scene)
    simulation.set_trajectory_visualization(np.array([[0, 0, -1], [1, 0, -1]]))
    for _ in range(60):
        simulation.step()
    simulation.reset()
    with pytest.raises(ValueError, match="planning_bounds"):
        simulation.sample_free_space()
    with pytest.raises(ValueError, match="goal"):
        MujocoTrajectoryTrackingEnv(np.zeros((1, 10)), model_path=vehicle_scene)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_generic_environment_rejects_invalid_stride(vehicle_scene, stride):
    with pytest.raises(ValueError, match="positive integer"):
        MujocoEnv(AltitudeTask(), model_path=vehicle_scene, steps_per_action=stride)


@pytest.mark.parametrize("action", [np.zeros(3), [np.nan, 0, 0, 0]])
def test_generic_environment_rejects_invalid_action(vehicle_scene, action):
    environment = MujocoEnv(AltitudeTask(), model_path=vehicle_scene)
    environment.reset()
    with pytest.raises(ValueError, match="four finite"):
        environment.step(action)
