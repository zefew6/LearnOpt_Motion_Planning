"""Adapt controllers and checkpoint policies to one episode execution interface."""

from uav_ac.control import CascadedController
from uav_ac.control.rl_controller import RLController


class ControllerAgent:
    def __init__(self, controller):
        self.controller = controller

    def reset(self):
        self.controller.reset()

    def step(self, env, observation):
        return env.step_controller(self.controller)


class PolicyAgent:
    def __init__(self, controller):
        # The loader validates spaces, physics and ACMPC versions. The policy
        # then consumes exactly the observation emitted by the training env.
        self.policy = controller.policy
        self.reset()

    def reset(self):
        self.state = None
        self.episode_start = True

    def step(self, env, observation):
        import numpy as np
        action, self.state = self.policy.predict(
            observation, state=self.state,
            episode_start=np.array([self.episode_start]), deterministic=True)
        self.episode_start = False
        return env.step(action)


def make_agent(config, env):
    agent = config["agent"]
    if agent["type"] == "rl":
        return PolicyAgent(RLController.from_checkpoint(
            agent["checkpoint"], env.quad, device=agent["device"]))
    if agent["name"] == "cascaded":
        return ControllerAgent(CascadedController(env.quad.g, env.control_dt))
    from uav_ac.control.mpc_controller import MPCConfig, MPCController
    return ControllerAgent(MPCController(
        env.quad, MPCConfig(dt=env.control_dt, **agent.get("options", {}))))
