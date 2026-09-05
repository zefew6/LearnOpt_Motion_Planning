"""Deployment-identical physical inputs for the MPC actor."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from uav_ac.control.rl_controller import encode_observation, OBSERVATION_SIZE, OBSERVATION_CLIP


def observation_space(horizon_steps: int) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "features": gym.spaces.Box(-OBSERVATION_CLIP, OBSERVATION_CLIP, (OBSERVATION_SIZE,), np.float32),
        "mpc_state": gym.spaces.Box(-np.inf, np.inf, (13,), np.float64),
        "reference": gym.spaces.Box(-np.inf, np.inf, (horizon_steps + 1, 10), np.float64),
    })


def encode_mpc_observation(quad, reference, previous_action, horizon_steps):
    state = quad.X.copy()
    state[3:7] /= np.linalg.norm(state[3:7])
    # Canonicalize the double cover. The cost builder aligns the reference to it.
    pivot = np.argmax(np.abs(state[3:7]))
    if state[3 + pivot] < 0:
        state[3:7] *= -1
    return {
        "features": encode_observation(quad, reference, previous_action),
        "mpc_state": state.astype(np.float64),
        "reference": reference.horizon(horizon_steps + 1)[:, :10].astype(np.float64),
    }
