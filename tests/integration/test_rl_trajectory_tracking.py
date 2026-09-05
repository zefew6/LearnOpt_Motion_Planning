import json

import numpy as np
from stable_baselines3 import PPO

from uav_ac.control import RLController, TrajectoryController
from uav_ac.control.rl_controller import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    RL_CONFIG_VERSION,
    quad_parameters,
)
from uav_ac.rl.common.environment import MujocoTrajectoryTrackingEnv
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_short_cpu_ppo_training_should_save_load_and_match_deployment_timing(tmp_path):
    simulation = MujocoSimulation(record_actual_trajectory=False)
    trajectory = np.zeros((40, 10))
    trajectory[:, :3] = simulation.start_position
    environment = MujocoTrajectoryTrackingEnv(
        trajectory, random_start=False, perturb_initial_state=False)
    model = PPO(
        "MlpPolicy",
        environment,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}},
        device="cpu",
        seed=42,
    )

    model.learn(total_timesteps=32)
    model.save(tmp_path / "best_model")
    config = {
        "schema_version": RL_CONFIG_VERSION,
        "observation_dim": OBSERVATION_SIZE,
        "action_shape": [ACTION_SIZE],
        "action_low": [-1.0] * ACTION_SIZE,
        "action_high": [1.0] * ACTION_SIZE,
        "steps_per_action": 10,
        "control_dt": 0.01,
        "quad_parameters": quad_parameters(simulation.quad),
    }
    (tmp_path / "rl_config.json").write_text(json.dumps(config), encoding="utf-8")

    controller = RLController.from_run(tmp_path, simulation.quad, device="cpu")
    tracker = TrajectoryController(controller, simulation.quad, trajectory, 10)
    for _ in range(20):
        tracker.step()
        simulation.step()

    assert tracker.trajectory_index == 1
    assert np.all(np.isfinite(simulation.quad.X))
    assert np.all(np.isfinite(controller._previous_action))
