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


def test_cost_gradient_batch_and_hover():
    settings = settings_from({"policy_type": "acmpc", "mpc": {"horizon_steps": 5}})
    env = make_environment(settings, perturb=False)
    obs, _ = env.reset()
    layer = RacingMPC(quad_parameters(env.unwrapped.quad), settings["mpc"])
    state = torch.tensor(np.stack([obs["mpc_state"]]*2))
    outputs = torch.zeros(2, layer.cost_size, requires_grad=True)
    previous = torch.zeros(2,4, dtype=torch.float64)
    local, cost = layer.build_racing_cost(state, outputs)
    assert cost.C.shape == (6,2,17,17)
    assert (cost.C.diagonal(dim1=-2, dim2=-1) > 0).all()
    assert torch.count_nonzero(cost.c[-1,:,13:]) == 0
    action = layer(state, previous, outputs, strict=True)
    torch.testing.assert_close(action, torch.zeros_like(action), atol=1e-5, rtol=0)
    action.sum().backward()
    assert torch.isfinite(outputs.grad).all() and outputs.grad.norm() > 0
    changed = outputs.detach().clone()
    changed[:, layer.cost_size//2:] = .2
    batch = layer(state, previous, changed, strict=True)
    one = layer(state[:1], previous[:1], changed[:1], strict=True)
    torch.testing.assert_close(batch[:1], one)
    assert batch.abs().max() <= 1
    assert not torch.allclose(batch, action)
    translated = state.clone()
    translated[:,:3] += 1000
    torch.testing.assert_close(layer(translated, previous, changed, strict=True), batch)
    state[0,0] = float("nan")
    with pytest.raises(ValueError, match="finite physical state"):
        layer(state, previous, changed, strict=True)
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


@pytest.mark.parametrize("change", [{"trajectory_bank_path":"unused"}, {"scene":"../other"},
    {"n_envs":0}, {"ppo":{"batch_size":1}}, {"mpc":{"dt":-1}}, {"typo":1}])
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
    settings["mpc"]["dt"] = .02
    with pytest.raises(ValueError, match="dt"):
        make_environment(settings)


@pytest.mark.parametrize("policy", ["mlp", "acmpc"])
def test_train_resume_evaluate(tmp_path, policy):
    settings = settings_from({"policy_type":policy, "n_envs":1, "total_timesteps":8,
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
    changed = deepcopy(settings)
    changed["episode_seconds"] = .03
    with pytest.raises(ValueError, match="incompatible"):
        train(tmp_path / "incompatible", changed, resume=run / "final_model.zip")


def test_batched_evaluation_matches_single_and_frame_path():
    settings = settings_from({"episode_seconds":.04})
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


def test_replay_existing_video_rejected_before_loading(tmp_path):
    video = tmp_path / "existing.mp4"
    video.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        replay(tmp_path, output=video)
    assert video.read_bytes() == b"original"
