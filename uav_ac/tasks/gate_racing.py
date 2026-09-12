"""Ordered gate racing without a reference path or time trajectory."""
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from uav_ac.scenes.loader import ENU_TO_NED

SCENE_PATH = Path(__file__).resolve().parents[1] / "simulation/models/gate_racing.xml"
FEATURE_SCALES = np.array([1.] * 4 + [10.] * 3 + [10.] * 3 + [1.] * 4
                          + [10.] * 24 + [1.] * 2)
FEATURE_SIZE = len(FEATURE_SCALES)


def observation_space(acmpc=False):
    features = gym.spaces.Box(-10., 10., (FEATURE_SIZE,), np.float32)
    if not acmpc:
        return features
    return gym.spaces.Dict({
        "features": features,
        "mpc_state": gym.spaces.Box(-np.inf, np.inf, (13,), np.float64),
        "previous_action": gym.spaces.Box(-1., 1., (4,), np.float64),
    })


@dataclass(frozen=True)
class Gate:
    center: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray

    @property
    def corners(self):
        local = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]])
        return self.center + (local * np.r_[0., self.half_size]) @ self.rotation.T

    def crossing(self, previous, current):
        """Return segment fraction for a forward, strictly interior crossing."""
        a = self.rotation.T @ (previous - self.center)
        b = self.rotation.T @ (current - self.center)
        if not a[0] < 0 <= b[0]:
            return None
        fraction = -a[0] / (b[0] - a[0])
        intersection = a + fraction * (b - a)
        return float(fraction) if np.all(np.abs(intersection[1:]) < self.half_size - 1e-9) else None


def read_gates(simulation):
    model, data = simulation.model, simulation.data
    entries = sorted((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i), i)
                     for i in range(model.nsite)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "").startswith("gate_"))
    if not entries or [name for name, _ in entries] != [f"gate_{i:02d}" for i in range(len(entries))]:
        raise ValueError("gate sites must be consecutively numbered from gate_00")
    gates = []
    for _, i in entries:
        if model.site_type[i] != mujoco.mjtGeom.mjGEOM_BOX or np.any(model.site_size[i, 1:] <= 0):
            raise ValueError("gate sites must be boxes with positive aperture sizes")
        gates.append(Gate(ENU_TO_NED @ data.site_xpos[i],
                          ENU_TO_NED @ data.site_xmat[i].reshape(3, 3),
                          model.site_size[i, 1:].copy()))
    return gates


class GateRacingTask:
    def __init__(self, *, acmpc=False, perturb_initial_state=True):
        self.acmpc = acmpc
        self.perturb_initial_state = perturb_initial_state
        self.observation_space = observation_space(acmpc)

    def reset(self, simulation, rng, options):
        self.gates = read_gates(simulation)
        if simulation.space_limits is None:
            raise ValueError("gate racing requires scene planning_bounds")
        state = simulation.quad.X.copy()
        if options.get("perturb_initial_state", self.perturb_initial_state):
            state[:3] += rng.uniform(-.2, .2, 3)
            state[7:10] = rng.uniform(-.1, .1, 3)
            delta = rng.uniform(-.02, .02, 3)
            state[3:7] = np.r_[1., delta]
            state[3:7] /= np.linalg.norm(state[3:7])
        quad = simulation.quad
        simulation.reset(state, np.full(4, np.sqrt(quad.m * quad.g / (4 * quad.kf))))
        if simulation.collision_detected:
            raise ValueError("gate racing reset is in collision")
        self.gate_index = 0
        self.previous_action = np.zeros(4, dtype=np.float64)
        self.pending_reward = 0.
        self.reason = None
        self.speed_integral = self.peak_speed = self.elapsed = 0.
        return self.info(simulation)

    def after_substep(self, simulation, previous_state, action):
        if self.reason is not None:
            return
        self.previous_action = action.astype(np.float64).copy()
        state = simulation.quad.X
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("non-finite gate-racing physical state")
        dt = simulation.quad.dt
        speed = float(np.linalg.norm(state[7:10]))
        self.elapsed += dt
        self.speed_integral += speed * dt
        self.peak_speed = max(self.peak_speed, speed)
        low, high = simulation.space_limits
        if simulation.collision_detected or np.any(state[:3] < low) or np.any(state[:3] > high):
            self.reason = "collision" if simulation.collision_detected else "out_of_bounds"
            self.pending_reward -= 10.
            return
        self.pending_reward -= .01 * np.linalg.norm(state[10:13]) * dt / .01
        start, end = previous_state[:3], state[:3]
        while self.gate_index < len(self.gates):
            gate = self.gates[self.gate_index]
            fraction = gate.crossing(start, end)
            stop = end if fraction is None else start + fraction * (end - start)
            self.pending_reward += np.linalg.norm(start-gate.center) - np.linalg.norm(stop-gate.center)
            if fraction is None:
                break
            self.gate_index += 1
            self.pending_reward += 10.
            if self.gate_index == len(self.gates):
                self.pending_reward += 10.
                self.reason = "success"
                break
            start = stop

    def observation(self, simulation):
        quad = simulation.quad
        state = quad.X.copy()
        state[3:7] /= np.linalg.norm(state[3:7])
        if state[3 + np.argmax(np.abs(state[3:7]))] < 0:
            state[3:7] *= -1
        corners = np.zeros((2, 4, 3))
        valid = np.zeros(2)
        for offset, gate in enumerate(self.gates[self.gate_index:self.gate_index+2]):
            corners[offset] = (gate.corners - state[:3]) @ quad.R()
            valid[offset] = 1
        features = np.r_[state[3:7], state[7:10] @ quad.R(), state[10:13],
                         self.previous_action, corners.ravel(), valid]
        features = np.clip(features / FEATURE_SCALES, -10, 10).astype(np.float32)
        return {"features": features, "mpc_state": state.astype(np.float64),
                "previous_action": self.previous_action.copy()} if self.acmpc else features

    def reward(self, simulation, action):
        reward, self.pending_reward = self.pending_reward, 0.
        return float(reward)

    def terminated(self, simulation):
        return self.reason is not None

    def info(self, simulation):
        return {"success": self.reason == "success", "gates_passed": self.gate_index,
                "collision": self.reason == "collision", "termination_reason": self.reason or "running",
                "elapsed_seconds": self.elapsed, "mean_speed": self.speed_integral / max(self.elapsed, 1e-12),
                "peak_speed": self.peak_speed}
