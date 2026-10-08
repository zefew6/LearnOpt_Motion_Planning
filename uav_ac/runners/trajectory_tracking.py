"""Plan and track a trajectory in the native MuJoCo viewer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from uav_ac.control import TrajectoryController
from uav_ac.control.factory import build_controller
from uav_ac.planning.pipeline.flight import plan_trajectory
from uav_ac.simulation.mujoco_sim import ENU_TO_NED, MujocoSimulation
from uav_ac.simulation.wind_disturb import GustingCrosswind


def _print_summary(config, trajectory, dt):
    positions = trajectory[:, :3]
    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    peak_speed = float(np.linalg.norm(trajectory[:, 3:6], axis=1).max())
    print(f"Flight: scene={Path(config['scene']).stem} | planner={config['planner']} | "
          f"controller={config['controller']} | wind={config['wind']}")
    print(f"Trajectory: duration={(len(trajectory)-1)*dt:.2f}s | length={length:.2f}m | "
          f"peak_speed={peak_speed:.2f}m/s | samples={len(trajectory)}")


def run_trajectory_tracking(config: dict) -> np.ndarray:
    """Plan and fly once; ordinary viewer runs intentionally write no files."""
    planning_capacity = 2 if config["planner"] == "bmtp" else 0
    simulation = MujocoSimulation(
        config["scene"], planning_path_capacity=planning_capacity)
    if config["wind"] != "none" and np.any(simulation.model.opt.wind):
        raise ValueError("fixed_gust conflicts with nonzero wind in the XML scene")
    controller, dt = build_controller(config, simulation)
    stride = dt / simulation.quad.dt
    if not math.isclose(stride, round(stride), abs_tol=1e-8) or round(stride) < 1:
        raise ValueError("controller period must be an integer multiple of the XML timestep")
    trajectory = plan_trajectory(config, simulation, dt)
    simulation.set_trajectory_visualization(trajectory[:, :3])
    tracker = TrajectoryController(
        controller, simulation.quad, trajectory, int(round(stride)))
    wind = (None if config["wind"] == "none"
            else GustingCrosswind(**config["wind_options"]))

    def step():
        if wind is not None:
            simulation.set_external_force_world(
                ENU_TO_NED @ wind.force_ned(float(simulation.data.time)))
        tracker.step()

    def reset():
        tracker.reset()
        simulation.set_external_force_world(np.zeros(3))

    _print_summary(config, trajectory, dt)
    simulation.run_interactive(step, reset, chase_camera=config.get("follow_camera", False))
    distance = float(np.linalg.norm(
        simulation.quad.position - simulation.goal_position))
    print(f"Finished: goal_error={distance:.2f}m | "
          f"collision={'yes' if simulation.collision_detected else 'no'}")
    return trajectory

