"""Zero-network-output acceptance: real dynamics, gate frames and timing."""
import time

import numpy as np
import pytest
import torch

from uav_ac.control.rl_controller import quad_parameters
from uav_ac.rl.acmpc.racing import RacingMPC
from uav_ac.rl.tasks.gate_racing.config import settings_from
from uav_ac.rl.tasks.gate_racing.environment import make_environment
from uav_ac.tasks.gate_course import apply_course, vehicle_diameter
from uav_ac.tasks.gate_racing import Gate, read_gates


@pytest.mark.parametrize("ratio", [4., 1.3])
def test_zero_weights_accelerate_and_cross_real_gate(ratio):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    settings = settings_from({"policy_type":"acmpc"})
    env = make_environment(settings, perturb=False)
    try:
        env.reset(seed=0)
        sim = env.unwrapped.simulation
        diameter = vehicle_diameter(sim)
        gates = [Gate(np.array([x,0.,-2.]), np.eye(3), np.full(2,ratio*diameter/2))
                 for x in (10.,20.,30.,34.,36.,37.)]
        apply_course(sim,gates)
        state = env.unwrapped.quad.X.copy()
        sim.reset(state, np.full(4,np.sqrt(sim.quad.m*sim.quad.g/(4*sim.quad.kf))))
        env.unwrapped.task.gates = read_gates(sim)[:1]
        obs = env.unwrapped.task.observation(sim)
        layer = RacingMPC(quad_parameters(sim.quad),settings["mpc"],
                          {**settings["racing"],"vehicle_diameter":diameter})
        speeds, latency = [], []
        for step in range(650):
            start = time.perf_counter()
            with torch.no_grad():
                action = layer(torch.tensor(obs["mpc_state"][None]),torch.tensor(obs["previous_action"][None]),
                               torch.tensor(obs["features"][None]),torch.zeros(1,layer.cost_size),strict=True)[0].numpy()
            latency.append(time.perf_counter()-start)
            obs,_,done,truncated,info = env.step(action)
            speeds.append(np.linalg.norm(sim.quad.X[7:10]))
            assert np.isfinite(sim.quad.X).all()
            if done or truncated:
                break
        assert info["success"], info
        assert sim.data.time == pytest.approx(info["elapsed_seconds"])
        assert not info["collision"]
        if ratio == 4:
            assert 6 <= info["gate_speeds"][0] <= 8
        else:
            assert info["gate_speeds"][0] < 4
            assert max(speeds) > info["gate_speeds"][0]+.5
        print({"diameter_ratio":ratio,"gate_speed":info["gate_speeds"][0],
               "peak_speed":max(speeds),"seconds":sim.data.time,
               "latency_ms_p50_p95_p99":(np.percentile(latency,[50,95,99])*1000).tolist(),
               "over_10ms_rate":float(np.mean(np.array(latency)>.01))})
    finally:
        env.close()
        torch.set_num_threads(old_threads)
