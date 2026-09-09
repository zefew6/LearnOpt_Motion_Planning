"""Gym lifecycle and normalized wrench execution without a trajectory dependency."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from uav_ac.control.rl_controller import ACTION_SIZE, action_to_command
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.tasks.base import Task


class MujocoEnv(gym.Env):
    """Execute a task using four bounded normalized thrust/moment actions."""

    metadata = {"render_modes": []}

    def __init__(self, task: Task, *, model_path: str | Path,
                 steps_per_action: int = 10):
        super().__init__()
        if (isinstance(steps_per_action, bool)
                or not isinstance(steps_per_action, (int, np.integer))
                or steps_per_action < 1):
            raise ValueError("steps_per_action must be a positive integer")
        self.task = task
        self.simulation = MujocoSimulation(model_path, record_actual_trajectory=False)
        self.quad = self.simulation.quad
        self.steps_per_action = int(steps_per_action)
        self.control_dt = self.quad.dt * self.steps_per_action
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_SIZE,), np.float32)
        self.observation_space = task.observation_space

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.simulation.reset()
        info = self.task.reset(
            self.simulation, self.np_random, {} if options is None else dict(options))
        return self.task.observation(self.simulation), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain four finite values")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        command = action_to_command(action, self.quad)
        for _ in range(self.steps_per_action):
            self.quad.set_propeller_speed(command.thrust, command.moment)
            self.simulation.step()
        observation = self.task.observation(self.simulation)
        reward = float(self.task.reward(self.simulation, action))
        terminated = bool(self.task.terminated(self.simulation))
        return observation, reward, terminated, False, {}
