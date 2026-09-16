import mujoco
import numpy as np
import pytest

from uav_ac.envs.mujoco_env import MujocoEnv
from uav_ac.scenes.loader import ENU_TO_NED
from uav_ac.tasks.gate_course import (apply_course, course_settings, describe_course,
                                     frame_boxes, sample_course, valid_course)
from uav_ac.tasks.gate_racing import GateRacingTask, SCENE_PATH, read_gates


def test_thousand_seeded_courses():
    bounds = (np.array([-8., -32., -9.]), np.array([38., 8., 0.]))
    start = np.array([0., 0., -2.])
    settings = course_settings({"mode": "random"})
    for seed in range(1000):
        gates = sample_course(np.random.default_rng(seed), start, bounds, settings)
        assert len(gates) == 6 and valid_course(gates, start, bounds)
        points = np.array([start] + [g.center for g in gates])
        steps = np.diff(points, axis=0)
        assert np.all((np.linalg.norm(steps, axis=1) >= 5) & (np.linalg.norm(steps, axis=1) <= 8))
        assert np.all(np.abs(steps[:, 2]) <= 1)
        assert np.all((points[1:, 2] >= -6) & (points[1:, 2] <= -2))
        headings = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
        assert np.all(np.abs(np.diff(headings)) <= np.pi/3 + 1e-12)
        for gate in gates:
            assert np.all((gate.half_size >= .75) & (gate.half_size <= 1.25))
            np.testing.assert_allclose(gate.rotation.T @ gate.rotation, np.eye(3), atol=1e-12)
        if seed < 10:
            again = sample_course(np.random.default_rng(seed), start, bounds, settings)
            assert describe_course(gates) == describe_course(again)


@pytest.mark.parametrize("settings", [{"typo": 1}, {"mode": "moving"}, {"width": [2, 1]},
    {"spacing": [float("nan"), 8]}, {"tilt_degrees": 90}, {"height_step": 5},
    {"width": [.2, 2]}, {"yaw_degrees": True}, {"mode": []}, {"width": "large"}])
def test_invalid_course(settings):
    with pytest.raises(ValueError):
        course_settings(settings)


def test_impossible_course_has_bounded_failure():
    with pytest.raises(ValueError, match="100 attempts"):
        sample_course(np.random.default_rng(0), np.array([0., 0., -2.]),
                      (np.array([-1., -1., -3.]), np.array([1., 1., 0.])), course_settings())


@pytest.mark.parametrize("acmpc", [False, True])
def test_random_reset_reproducibility_and_geometry(acmpc):
    env = MujocoEnv(GateRacingTask(acmpc=acmpc, course={"mode": "random"}), model_path=SCENE_PATH)
    try:
        first, _ = env.reset(seed=17)
        original = env.task.course_description
        for seed in [19, 22, 17]:
            current, _ = env.reset(seed=seed)
            if seed == 17:
                assert env.task.course_description == original
                for a, b in zip(first.values(), current.values()) if acmpc else [(first, current)]:
                    np.testing.assert_array_equal(a, b)
            else:
                assert env.task.course_description != original
        env.reset(seed=17, options={"perturb_initial_state": False})
        assert env.task.course_description == original
        sim = env.simulation
        course_seed = np.random.default_rng(17).integers(0, 2**63, size=2, dtype=np.int64)[0]
        sampled = sample_course(np.random.default_rng(course_seed), sim.start_position,
                                sim.space_limits, env.task.course)
        for index, gate in enumerate(env.task.gates):
            np.testing.assert_allclose(gate.rotation, sampled[index].rotation, atol=1e-12)
            np.testing.assert_allclose(gate.center, sampled[index].center, atol=1e-12)
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"gate_{index:02d}")
            ids = np.flatnonzero(sim.model.geom_bodyid == bid)
            for gid, (position, size) in zip(ids, frame_boxes(gate)):
                np.testing.assert_allclose(ENU_TO_NED @ sim.data.geom_xpos[gid], gate.center+gate.rotation @ position)
                np.testing.assert_allclose(sim.model.geom_size[gid], size)
                np.testing.assert_allclose(ENU_TO_NED @ sim.data.geom_xmat[gid].reshape(3, 3), gate.rotation, atol=1e-12)
        features = env.task.observation(sim)
        features = features["features"] if acmpc else features
        expected = np.stack([(g.corners-sim.quad.X[:3]) @ sim.quad.R() for g in env.task.gates[:2]]) / 10
        np.testing.assert_allclose(features[14:38].reshape(2, 4, 3), expected, atol=1e-7)
        # A real contact at the new tilted frame, followed by a clear center crossing.
        gate = env.task.gates[0]
        state = env.quad.X.copy()
        state[:3] = gate.center + gate.rotation @ np.array([0., gate.half_size[0]+.08, 0.])
        sim.reset(state)
        assert sim.collision_detected
        state[:3] = gate.center - .005 * gate.rotation[:, 0]
        state[7:10] = 20 * gate.rotation[:, 0]
        sim.reset(state)
        assert not sim.collision_detected
        _, _, done, _, info = env.step(np.zeros(4))
        assert not done and info["gates_passed"] == 1
    finally:
        env.close()


def test_recompiled_collision_bounds_follow_smaller_and_larger_apertures():
    from uav_ac.tasks.gate_racing import Gate
    env = MujocoEnv(GateRacingTask(perturb_initial_state=False), model_path=SCENE_PATH)
    try:
        env.reset()
        gates = read_gates(env.simulation)
        state = env.quad.X.copy()
        state[:3] = gates[0].center + gates[0].rotation @ np.array([0., 1.35, 0.])
        for half_width, collision in [(1.3, True), (.75, False), (1.3, True)]:
            gates[0] = Gate(gates[0].center, gates[0].rotation, np.array([half_width, 1.]))
            apply_course(env.simulation, gates)
            env.simulation.reset(state)
            assert env.simulation.collision_detected == collision
    finally:
        env.close()
