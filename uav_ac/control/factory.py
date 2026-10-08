"""Build a flight controller from validated interactive-flight settings."""

from __future__ import annotations

import math

from uav_ac.control import CascadedConfig, CascadedController, RLController
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def build_controller(config: dict, simulation: MujocoSimulation):
    """Create the selected controller and return it with its control period."""
    name = config["controller"]
    if name == "rl":
        controller = RLController.from_checkpoint(
            config["rl"]["checkpoint"], simulation.quad, device=config["rl"]["device"])
        return controller, controller.control_dt

    dt = float(config["control_dt"])
    stride = dt / simulation.quad.dt
    if not math.isclose(stride, round(stride), abs_tol=1e-8) or round(stride) < 1:
        raise ValueError("control_dt must be an integer multiple of the XML timestep")
    if name == "cascaded":
        settings = config.get("cascaded", {})
        if settings:
            CascadedConfig(**settings).apply_to(simulation.quad)
        return CascadedController(simulation.quad.g, dt), dt

    from uav_ac.control.mpc_controller import MPCConfig, MPCController
    return MPCController(simulation.quad, MPCConfig(dt=dt, **config["mpc"])), dt


