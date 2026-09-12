from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from uav_ac.envs.mujoco_env import MujocoEnv
from uav_ac.tasks.gate_racing import Gate, GateRacingTask, SCENE_PATH, FEATURE_SIZE


@pytest.fixture
def env():
    environment = MujocoEnv(GateRacingTask(perturb_initial_state=False), model_path=SCENE_PATH)
    environment.reset(seed=3)
    yield environment
    environment.close()


@pytest.mark.parametrize("a,b,expected", [([-2,0,0], [2,0,0], .5),
    ([2,0,0], [-2,0,0], None), ([-2,1,0], [2,1,0], None),
    ([-2,1.2,0], [2,1.2,0], None), ([-20,.2,.3], [20,.2,.3], .5),
    ([0,0,0], [1,0,0], None), ([-1,0,0], [0,0,0], 1.)])
def test_direction_and_aperture(a, b, expected):
    gate = Gate(np.zeros(3), np.eye(3), np.ones(2))
    assert gate.crossing(np.array(a), np.array(b)) == expected


@pytest.mark.parametrize("acmpc", [False, True])
def test_observation_reset_and_coordinates(acmpc):
    env = MujocoEnv(GateRacingTask(acmpc=acmpc), model_path=SCENE_PATH)
    check_env(env)
    a, _ = env.reset(seed=4)
    b, _ = env.reset(seed=4)
    for x, y in zip(a.values(), b.values()) if acmpc else [(a, b)]:
        np.testing.assert_array_equal(x, y)
    env.reset(options={"perturb_initial_state": False})
    task = env.task
    np.testing.assert_allclose(task.gates[0].center, [8, 0, -2])
    np.testing.assert_allclose(task.gates[3].rotation[:, 0], [0, -1, 0], atol=1e-12)
    task.gate_index = 5
    observation = task.observation(env.simulation)
    features = observation["features"] if acmpc else observation
    assert features.shape == (FEATURE_SIZE,)
    np.testing.assert_array_equal(features[-2:], [1, 0])
    np.testing.assert_array_equal(features[26:38], 0)
    task.gate_index = 6
    features = task.observation(env.simulation)
    features = features["features"] if acmpc else features
    np.testing.assert_array_equal(features[14:], 0)
    env.close()


def move(task, sim, previous, current, *, collision=False):
    old = sim.quad.X.copy()
    old[:3] = previous
    state = old.copy()
    state[:3] = current
    fake = SimpleNamespace(quad=SimpleNamespace(X=state, dt=.001),
                           collision_detected=collision, space_limits=sim.space_limits)
    task.after_substep(fake, old, np.zeros(4))
    return task.reward(fake, np.zeros(4))


def test_progress_switch_and_single_consumption(env):
    task, sim = env.task, env.simulation
    task.gates = [Gate(np.array([8.,0.,-2.]), np.eye(3), np.ones(2)),
                  Gate(np.array([18.,0.,-2.]), np.eye(3), np.ones(2))]
    reward = move(task, sim, [7,0,-2], [9,0,-2])
    assert reward == pytest.approx(12.)
    assert task.gate_index == 1
    assert task.reward(sim, np.zeros(4)) == 0
    assert not task.terminated(sim)
    assert move(task, sim, [17,0,-2], [19,0,-2]) == pytest.approx(21.)
    assert task.reason == "success"


def test_multiple_gates_and_skipped_gate(env):
    task, sim = env.task, env.simulation
    task.gates = [Gate(np.array([x,0.,-2.]), np.eye(3), np.ones(2)) for x in [8., 9.]]
    move(task, sim, [7,0,-2], [10,0,-2])
    assert task.gate_index == 2 and task.reason == "success"
    env.reset()
    move(task, sim, [17,0,-3], [19,0,-3])
    assert task.gate_index == 0


def test_collision_takes_priority(env):
    reward = move(env.task, env.simulation, [7,0,-2], [9,0,-2], collision=True)
    assert reward == -10
    assert env.task.gate_index == 0
    assert env.task.reason == "collision"


def test_real_fast_crossing_and_early_out_of_bounds(env):
    state = env.quad.X.copy()
    state[:3] = [7.995, 0, -2]
    state[7] = 20
    env.simulation.reset(state, env.quad.omega.copy())
    obs, _, done, _, info = env.step(np.zeros(4))
    assert not done and info["gates_passed"] == 1
    np.testing.assert_array_equal(obs[10:14], np.zeros(4))
    env.reset()
    state = env.quad.X.copy()
    state[0], state[7] = 37.999, 20
    env.simulation.reset(state)
    _, _, done, _, info = env.step(np.zeros(4))
    assert done and info["termination_reason"] == "out_of_bounds"
    assert env.simulation.data.time == pytest.approx(.001)


def test_rotor_contact_is_a_collision(env):
    state = env.quad.X.copy()
    state[:3] = [8, .86, -2]
    env.simulation.reset(state)
    assert env.simulation.collision_detected
    model = env.simulation.model
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i))
             for contact in env.simulation.data.contact for i in (contact.geom1, contact.geom2)}
    assert any((name or "").startswith("rotor_visual_") for name in names)
    assert "body" not in names


def test_time_limit(env):
    limited = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    limited.reset()
    _, _, terminated, truncated, _ = limited.step(np.zeros(4))
    assert truncated and not terminated


def test_rate_penalty_independent_of_substep_partition(env):
    def integrate(dt, count):
        env.reset()
        state = env.quad.X.copy()
        state[10] = 2.
        sim = SimpleNamespace(quad=SimpleNamespace(X=state, dt=dt), collision_detected=False,
                              space_limits=env.simulation.space_limits)
        for _ in range(count):
            env.task.after_substep(sim, state, np.zeros(4))
        return env.task.reward(sim, np.zeros(4))
    assert integrate(.001, 10) == pytest.approx(-.02)
    assert integrate(.002, 5) == pytest.approx(-.02)
