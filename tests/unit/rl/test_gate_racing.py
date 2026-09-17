from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from uav_ac.control.rl_controller import quad_parameters
from uav_ac.rl.acmpc.racing import RacingMPC, RacingACMPCPolicy
from uav_ac.rl.gate_racing import settings_from, make_environment, contract, read_run, train, evaluate_metrics, evaluate_model, replay
from uav_ac.rl.training import load_training_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_cost_gradient_batch_and_gate_response():
    settings = settings_from({"policy_type": "acmpc", "mpc": {"horizon_steps": 5}})
    env = make_environment(settings, perturb=False)
    obs, _ = env.reset()
    layer = RacingMPC(quad_parameters(env.unwrapped.quad), settings["mpc"])
    state = torch.tensor(np.stack([obs["mpc_state"]]*2))
    features = torch.tensor(np.stack([obs["features"]]*2))
    outputs = torch.zeros(2, layer.cost_size, requires_grad=True)
    previous = torch.zeros(2,4, dtype=torch.float64)
    local, cost = layer.build_racing_cost(state, features, outputs)
    assert cost.C.shape == (6,2,17,17)
    assert (cost.C.diagonal(dim1=-2, dim2=-1) > 0).all()
    assert torch.count_nonzero(cost.c[-1,:,13:]) == 0
    action = layer(state, previous, features, outputs, strict=True)
    assert action.abs().max() > 1e-5
    action.sum().backward()
    assert torch.isfinite(outputs.grad).all() and outputs.grad.norm() > 0
    changed = outputs.detach().clone()
    changed[:, layer.cost_size//2:] = .2
    batch = layer(state, previous, features, changed, strict=True)
    one = layer(state[:1], previous[:1], features[:1], changed[:1], strict=True)
    torch.testing.assert_close(batch[:1], one)
    assert batch.abs().max() <= 1
    assert not torch.allclose(batch, action)
    translated = state.clone()
    translated[:,:3] += 1000
    torch.testing.assert_close(layer(translated, previous, features, changed, strict=True), batch)
    state[0,0] = float("nan")
    with pytest.raises(ValueError, match="finite physical state"):
        layer(state, previous, features, changed, strict=True)
    env.close()


def test_ppo_physical_buffer_and_save_load(tmp_path):
    settings = settings_from({"policy_type":"acmpc", "mpc":{"horizon_steps":3}})
    env = make_environment(settings)
    model = PPO(RacingACMPCPolicy, env, n_steps=4, batch_size=4, n_epochs=1,
                policy_kwargs={"quad_parameters":quad_parameters(env.unwrapped.quad),
                               "mpc_settings": settings["mpc"], "net_arch":{"pi":[16],"vf":[16]}}, seed=4)
    before = model.policy.cost_net[-1].weight.detach().clone()
    model.learn(4)
    assert model.rollout_buffer.observations["mpc_state"].dtype == np.float64
    assert not torch.equal(before, model.policy.cost_net[-1].weight)
    obs, _ = env.reset(seed=5)
    expected, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "model.zip")
    loaded = PPO.load(tmp_path / "model.zip", env=env)
    np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], expected)
    env.close()


def test_geometric_cost_turns_toward_gate_without_tracking_reference(monkeypatch):
    from uav_ac.rl.acmpc import solver
    from uav_ac.tasks.gate_racing import Gate
    monkeypatch.setattr(solver, "reference_targets", lambda *args: pytest.fail("tracking target used for racing"))
    settings = settings_from({"policy_type":"acmpc","mpc":{"horizon_steps":10}})
    env = make_environment(settings,perturb=False)
    try:
        env.reset()
        layer = RacingMPC(quad_parameters(env.unwrapped.quad),settings["mpc"])
        actions = []
        for sign in [1,-1]:
            rotation = np.diag([sign,sign,1.])
            env.unwrapped.task.gates = [Gate(np.array([8.*sign,0.,-2.]),rotation,np.ones(2))]
            obs = env.unwrapped.task.observation(env.unwrapped.simulation)
            with torch.no_grad():
                actions.append(layer(torch.tensor(obs["mpc_state"][None]),torch.zeros(1,4),
                                     torch.tensor(obs["features"][None]),torch.zeros(1,layer.cost_size),strict=True))
        assert actions[0][0,2] < 0 < actions[1][0,2]
    finally:
        env.close()


@pytest.mark.parametrize("change", [{"trajectory_bank_path":"unused"}, {"scene":"../other"},
    {"n_envs":0}, {"ppo":{"batch_size":1}}, {"mpc":{"dt":-1}}, {"typo":1},
    {"course":{"typo":1}}, {"course":None}, {"course":{"mode":"moving"}}])
def test_invalid_settings(change):
    with pytest.raises((ValueError, FileNotFoundError)):
        settings_from(change)


def test_config_contract_and_timing(tmp_path):
    settings = load_training_config("configs/acmpc_gate_racing.yaml")
    assert "trajectory_bank_path" not in settings
    env = make_environment(settings)
    metadata = {"task":"gate_racing", "settings":settings, "contract":contract(settings, env)}
    env.close()
    (tmp_path / "rl_config.json").write_text(json.dumps(metadata))
    read_run(tmp_path)
    metadata["contract"]["scene_sha256"] = "changed"
    (tmp_path / "rl_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract"):
        read_run(tmp_path)
    assert env.unwrapped.control_dt == .01 and settings["mpc"]["dt"] == .02
    settings["mpc"]["dt"] = .015
    with pytest.raises(ValueError, match="dt"):
        make_environment(settings)


@pytest.mark.parametrize("policy", ["mlp", "acmpc"])
def test_train_resume_evaluate(tmp_path, policy):
    settings = settings_from({"policy_type":policy, "course":{"mode":"random"}, "n_envs":1, "total_timesteps":8,
        "episode_seconds":.02, "evaluation_episodes":2, "evaluation_interval":8,
        "checkpoint_interval":8, "mpc":{"horizon_steps":3},
        "ppo":{"n_steps":4, "batch_size":4, "n_epochs":1, "net_arch":{"pi":[16],"vf":[16]}}})
    run = train(tmp_path / policy, settings)
    assert (run / "best_model.zip").exists()
    model = PPO.load(run / "final_model.zip")
    assert model.num_timesteps == 8
    resumed = train(tmp_path / (policy+"_resume"), settings, resume=run / "final_model.zip")
    assert PPO.load(resumed / "final_model.zip").num_timesteps == 16
    metrics = evaluate_metrics(resumed, seeds=(4,5))
    assert metrics["timeout_rate"] == 1
    assert metrics["solver_failure_rate"] == 0
    assert metrics["missed_gate_rate"] == 0
    assert len(metrics["episodes"][0]["course"]) == 6
    changed = deepcopy(settings)
    changed["episode_seconds"] = .03
    with pytest.raises(ValueError, match="incompatible"):
        train(tmp_path / "incompatible", changed, resume=run / "final_model.zip")


def test_batched_evaluation_matches_single_and_frame_path():
    settings = settings_from({"episode_seconds":.04, "course":{"mode":"random"}})
    class HoverPolicy:
        device = torch.device("cpu")
        policy = object()
        def predict(self, obs, deterministic=True):
            return np.zeros((len(obs),4)), None
    model = HoverPolicy()
    batch = evaluate_model(model, settings, (4,5))
    frames = []
    single = evaluate_model(model, settings, (4,), frame=lambda env, done: frames.append((env.unwrapped.simulation.data.time, done)))
    for key in ("elapsed_seconds", "mean_speed", "peak_speed", "return", "gates_passed"):
        assert batch["episodes"][0][key] == pytest.approx(single["episodes"][0][key])
    assert frames[0] == (0., False)
    assert frames[-1] == pytest.approx((.04, True))
    assert batch["episodes"][0]["termination_reason"] == "timeout"
    assert batch["episodes"][0]["course"] == single["episodes"][0]["course"]
    assert batch["single_environment_latency"] is None
    assert single["single_environment_latency"]["samples"] == 4
    assert single["single_environment_latency"]["p99_ms"] >= single["single_environment_latency"]["p50_ms"]


def test_old_contract_rejected_and_course_is_part_of_contract(tmp_path):
    settings = settings_from({})
    env = make_environment(settings)
    try:
        signature = contract(settings, env)
        changed = settings_from({"course":{"mode":"random"}})
        assert contract(changed, env) != signature
        signature["task_version"] = 1
        (tmp_path / "rl_config.json").write_text(json.dumps({"settings":settings, "contract":signature}))
        with pytest.raises(ValueError, match="train a new run"):
            read_run(tmp_path)
    finally:
        env.close()


def test_replay_existing_video_rejected_before_loading(tmp_path):
    video = tmp_path / "existing.mp4"
    video.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        replay(tmp_path, output=video)
    assert video.read_bytes() == b"original"
