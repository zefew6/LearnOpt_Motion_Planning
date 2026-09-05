import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from uav_ac.control.rl_controller import action_to_command, quad_parameters
from uav_ac.rl.common.environment import MujocoTrajectoryTrackingEnv
from uav_ac.rl.acmpc.policy import ACMPCPolicy
from uav_ac.rl.acmpc.dynamics import QuadrotorDynamics
from uav_ac.rl.acmpc.solver import DifferentiableMPC, reference_targets
from uav_ac.rl.acmpc.vendor.mpc import MPC, QuadCost, LinDx
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH


@pytest.fixture
def env():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    trajectory = np.zeros((50, 10))
    trajectory[:, 2] = -1.5
    env = MujocoTrajectoryTrackingEnv(trajectory, model_path=OPEN_FIELD_SCENE_PATH,
        observation_mode="acmpc", mpc_horizon_steps=5, random_start=False, perturb_initial_state=False)
    yield env
    env.close()
    torch.set_num_threads(previous_threads)


def test_wrench_and_dynamics_jacobian(env):
    dx = QuadrotorDynamics(quad_parameters(env.quad), .01)
    actions = torch.tensor([[0., 0., 0., 0.], [-.3,.2,-.1,.05], [.4,-.1,.05,.2]], dtype=torch.float64)
    for action, actual in zip(actions, dx.wrench(actions)):
        command = action_to_command(action.numpy(), env.quad)
        np.testing.assert_allclose(actual.numpy(), np.r_[command.thrust, command.moment])
    state = torch.zeros((3, 13), dtype=torch.float64)
    state[:, 3] = 1
    torch.testing.assert_close(dx(state[:1], actions[:1]), state[:1])
    # Positive collective thrust accelerates upward (negative NED z).
    assert dx(state, actions)[2, 9] < 0
    a, b = dx.grad_input(state[1:], actions[1:])
    delta = 1e-6
    for i in range(4):
        perturb = torch.zeros_like(actions[1:]); perturb[:, i] = delta
        numeric = (dx(state[1:], actions[1:]+perturb)-dx(state[1:], actions[1:]-perturb))/(2*delta)
        torch.testing.assert_close(b[..., i], numeric, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("target", [.1, 5.])
def test_linear_quadratic_fixed_point_gradient(target):
    torch.set_num_threads(1)
    dtype = torch.float64
    def solve(parameter):
        c = torch.zeros((4,1,2), dtype=dtype)
        c[:, :, 0] = -parameter
        C = torch.diag_embed(torch.tensor([1., .2], dtype=dtype)).expand(4,1,2,2).clone()
        F = torch.tensor([1.,1.], dtype=dtype).reshape(1,1,1,2).expand(3,1,1,2)
        solver = MPC(1,1,4,u_lower=-.4,u_upper=.4,n_batch=1,lqr_iter=10,eps=1e-7,verbose=-1)
        _, actions, _ = solver(torch.zeros(1,1,dtype=dtype),QuadCost(C,c),LinDx(F,None))
        return actions[0,0,0]
    parameter = torch.tensor(target,dtype=dtype,requires_grad=True)
    grad, = torch.autograd.grad(solve(parameter), parameter)
    numeric = (solve(parameter.detach()+1e-5)-solve(parameter.detach()-1e-5))/2e-5
    torch.testing.assert_close(grad, numeric, atol=1e-5, rtol=1e-4)


def test_batch_order_and_quaternion_sign(env):
    observation, _ = env.reset()
    layer = DifferentiableMPC(quad_parameters(env.quad), {"horizon_steps":5})
    state = torch.tensor(observation["mpc_state"]).repeat(3,1)
    state[:,0] += torch.tensor([.05,-.08,.02])
    reference = torch.tensor(observation["reference"]).repeat(3,1,1)
    previous = torch.zeros(3,4)
    costs = torch.zeros(3,layer.cost_size)
    with torch.no_grad():
        batch = layer(state,reference,previous,costs,strict=True)
        individual = torch.cat([layer(state[i:i+1],reference[i:i+1],previous[i:i+1],costs[i:i+1],strict=True) for i in range(3)])
        reverse = layer(state.flip(0),reference.flip(0),previous,costs,strict=True).flip(0)
        negative = state.clone(); negative[:,3:7] *= -1
        opposite = layer(negative,reference,previous,costs,strict=True)
    torch.testing.assert_close(batch, individual, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(batch, reverse, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(batch, opposite, atol=2e-6, rtol=1e-5)


def test_ppo_gradient_probability_and_save_load(env, tmp_path):
    model = PPO(ACMPCPolicy, env, n_steps=8, batch_size=8,n_epochs=1, seed=42,device="cpu",
        policy_kwargs={"quad_parameters":quad_parameters(env.quad),"mpc_settings":{"horizon_steps":5},
            "net_arch":{"pi":[32],"vf":[32]}})
    obs, _ = env.reset()
    obs["mpc_state"][0] += .1
    tensor, _ = model.policy.obs_to_tensor(obs)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        action, _, old_log = model.policy(tensor)
    model.policy.set_training_mode(True)
    _, log, _ = model.policy.evaluate_actions(tensor, action)
    torch.testing.assert_close(log, old_log, atol=1e-6, rtol=1e-6)
    (-log.mean()).backward()
    assert model.policy.cost_net[-1].weight.grad.norm() > 0
    assert torch.isfinite(model.policy.cost_net[-1].weight.grad).all()
    before = model.policy.cost_net[-1].weight.detach().clone()
    model.learn(16)
    assert not torch.equal(before, model.policy.cost_net[-1].weight)
    model.save(tmp_path/"model")
    loaded = PPO.load(tmp_path/"model", env=env, device="cpu")
    np.testing.assert_allclose(model.predict(obs,deterministic=True)[0], loaded.predict(obs,deterministic=True)[0])
    loaded.learn(8,reset_num_timesteps=False)


def test_deployment_observation_and_timing(env):
    from uav_ac.control.rl_controller import RLController
    from uav_ac.control.trajectory_controller import TrajectoryReference
    observation, _ = env.reset()
    class RecordingPolicy:
        def predict(self, obs, **kwargs):
            self.observation = obs
            self.calls = getattr(self,"calls",0)+1
            return np.array([.1,.02,0,0]), None
    policy = RecordingPolicy()
    controller = RLController(policy,env.quad,control_dt=.01,steps_per_action=10,
        observation_mode="acmpc",mpc_horizon_steps=5)
    reference = TrajectoryReference(env.trajectory,0,.01)
    command = controller.step(env.quad,reference)
    for key in observation:
        np.testing.assert_array_equal(policy.observation[key],observation[key])
    held = controller.step(env.quad,TrajectoryReference(env.trajectory,0,.01,is_control_tick=False))
    assert held is command and policy.calls == 1
    controller.reset()
    controller.step(env.quad,reference)
    np.testing.assert_array_equal(policy.observation["features"][22:26],np.zeros(4))


def test_unconverged_solver_raises_or_holds_previous(env):
    obs,_ = env.reset()
    layer = DifferentiableMPC(quad_parameters(env.quad),{"horizon_steps":5,"iterations":2,"retry_iterations":2,"tolerance":1e-12})
    x = torch.tensor(obs["mpc_state"])[None]; x[:,0]+=.3
    ref = torch.tensor(obs["reference"])[None]
    prev = torch.tensor([[.01,.02,0.,0.]])
    cost = torch.zeros(1,layer.cost_size)
    with pytest.raises(RuntimeError,match="did not converge"):
        layer(x,ref,prev,cost,strict=True)
    with torch.no_grad():
        torch.testing.assert_close(layer(x,ref,prev,cost,strict=False),prev)
    assert layer.last_diagnostics["failures"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA not visible in this process")
def test_cuda_solver_backward(env):
    observation,_ = env.reset()
    layer = DifferentiableMPC(quad_parameters(env.quad),{"horizon_steps":5}).cuda()
    x = torch.tensor(observation["mpc_state"],device="cuda")[None]; x[:,0]+=.1
    ref = torch.tensor(observation["reference"],device="cuda")[None]
    cost = torch.zeros(1,layer.cost_size,device="cuda",requires_grad=True)
    action = layer(x,ref,torch.zeros(1,4,device="cuda"),cost,strict=True)
    grad, = torch.autograd.grad(action[0,2],cost)
    assert torch.isfinite(grad).all() and grad.norm() > 0
