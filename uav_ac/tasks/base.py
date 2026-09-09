"""Small task contract for the generic MuJoCo Gym environment."""

from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from uav_ac.simulation.mujoco_sim import MujocoSimulation


class Task(Protocol):
    """Define episode semantics; the environment owns physics and wrench actions.

    reset runs after the simulation reset and may choose a new physical state.
    observation/reward/terminated are evaluated after all physics substeps.
    Use Gymnasium TimeLimit for task-independent episode time limits.
    """

    observation_space: gym.Space

    def reset(self, simulation: MujocoSimulation, rng: np.random.Generator,
              options: dict[str, Any]) -> dict[str, Any]: ...

    def observation(self, simulation: MujocoSimulation) -> Any: ...

    def reward(self, simulation: MujocoSimulation, action: np.ndarray) -> float: ...

    def terminated(self, simulation: MujocoSimulation) -> bool: ...
