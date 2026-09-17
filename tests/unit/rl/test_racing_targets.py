import numpy as np
import pytest
import torch

from uav_ac.rl.acmpc.racing_targets import RacingSettings, gate_targets
from uav_ac.rl.tasks.gate_racing.config import settings_from
from uav_ac.rl.tasks.gate_racing.environment import make_environment
from uav_ac.tasks.gate_racing import Gate


def target_inputs(width=1.8, turn=False):
    env = make_environment(settings_from({"policy_type":"acmpc"}), perturb=False)
    obs, _ = env.reset(seed=0)
    env.unwrapped.task.gates = [Gate(np.array([8.,0.,-2.]),np.eye(3),np.full(2,width/2)),
                                Gate(np.array([13.,5. if turn else 0.,-2.]),np.eye(3),np.full(2,width/2))]
    obs = env.unwrapped.task.observation(env.unwrapped.simulation)
    env.close()
    return torch.tensor(obs["mpc_state"][None]), torch.tensor(obs["features"][None])


def test_gate_aperture_and_turn_reduce_target_speed():
    def speed(width, turn=False):
        state, features = target_inputs(width, turn)
        return gate_targets(state, features, 50, .02, RacingSettings())[-1].item()
    assert speed(1.8) == pytest.approx(7.)
    assert speed(.585) == pytest.approx(2., abs=1e-5)
    assert speed(1.8, True) < speed(1.8)


def test_gate_previews_are_ordered_and_final_target_keeps_moving():
    state, features = target_inputs()
    state[:, 7] = 7
    # Bring the current gate to 2 m in front of the vehicle.
    corners = features[:,14:38].reshape(1,2,4,3).clone()
    corners[...,0] -= .6
    features[:,14:38] = corners.reshape(1,24)
    positions, velocity, _, _, _ = gate_targets(state, features, 50, .02, RacingSettings())
    assert positions[0,-1,0] > 2
    assert torch.all(torch.diff(positions[0,:,0]) >= -1e-10)
    features[:,-1] = 0
    positions, velocity, _, _, _ = gate_targets(state, features, 50, .02, RacingSettings())
    assert positions[0,-1,0] > 2 and velocity[0,-1,0] > 6


def test_no_active_gate_is_finite_hover_target():
    state, features = target_inputs()
    features[:,14:] = 0
    position, velocity, frame, _, _ = gate_targets(state, features, 50, .02, RacingSettings())
    assert torch.isfinite(frame).all()
    assert torch.count_nonzero(position) == torch.count_nonzero(velocity) == 0


@pytest.mark.parametrize("values", [{"cruise_speed":9}, {"minimum_speed":8},
    {"braking_acceleration":0}, {"lateral_acceleration":float("nan")}])
def test_invalid_task_targets(values):
    with pytest.raises(ValueError):
        settings_from({"racing":values})
