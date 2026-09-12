"""Gym environment composition for gate-racing experiments."""

import gymnasium as gym
import numpy as np

from uav_ac.envs.mujoco_env import MujocoEnv
from uav_ac.tasks.gate_racing import GateRacingTask
from .config import scene_path


def make_environment(settings, *, perturb=True, record_actual_trajectory=False):
    env = MujocoEnv(GateRacingTask(acmpc=settings["policy_type"] == "acmpc",
                                 perturb_initial_state=perturb),
                    model_path=scene_path(settings), steps_per_action=settings["steps_per_action"],
                    record_actual_trajectory=record_actual_trajectory)
    if settings["policy_type"] == "acmpc" and not np.isclose(env.control_dt, settings["mpc"]["dt"], rtol=0, atol=1e-12):
        raise ValueError("MPC dt must equal environment control_dt")
    steps = settings["episode_seconds"] / env.control_dt
    if not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError("episode_seconds must be an integer number of control intervals")
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(round(steps)))
