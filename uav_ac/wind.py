"""GCOPTER-only laboratory-course wind demo."""

import os
from functools import partial
from typing import Literal

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from uav_ac import utils
from uav_ac.control import CascadedController, TrajectoryController
from uav_ac.planning.pipeline import build_mission_corridor, generate_gcopter_mission
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, ENU_TO_NED, MujocoSimulation
from uav_ac.simulation.wind_disturb import GustingCrosswind


ControllerName = Literal["cascaded", "mpc"]
CONTROLLER: ControllerName = "mpc"


def _build_controller(
        controller_name: ControllerName, quad, trajectory_dt: float,
):
    """Construct the same controller variants supported by ``main.py``."""
    if controller_name == "cascaded":
        return CascadedController(quad.g, trajectory_dt)

    if controller_name == "mpc":
        from uav_ac.control.mpc_controller import MPCConfig, MPCController

        return MPCController(
            quad,
            MPCConfig(
                dt=trajectory_dt,
                horizon_steps=10,
                nlp_solver_type="SQP_RTI",
            ),
        )

    raise ValueError(f"unsupported controller: {controller_name}")


def _plan_gcopter_trajectory(
        simulation: MujocoSimulation, velocity: float, trajectory_dt: float,
) -> np.ndarray:
    corridor = build_mission_corridor(
        simulation, simulation.mission_waypoints, visualize=False)
    return generate_gcopter_mission(
        simulation.mission_waypoints, corridor, simulation.quad,
        velocity, trajectory_dt)


def _trajectory_after_takeoff(
        trajectory: np.ndarray, takeoff_waypoint: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(trajectory[:, :3] - takeoff_waypoint, axis=1)
    return trajectory[np.argmin(distances):]


def _wind_control_step(
        simulation: MujocoSimulation,
        wind: GustingCrosswind,
        trajectory_controller: TrajectoryController,
) -> None:
    force_ned = wind.force_ned(float(simulation.data.time))
    simulation.set_external_force_world(ENU_TO_NED @ force_ned)
    trajectory_controller.step()


def _reset_wind_control(
        simulation: MujocoSimulation,
        trajectory_controller: TrajectoryController,
) -> None:
    trajectory_controller.reset()
    simulation.set_external_force_world(np.zeros(3))


def main(
        steady_force: tuple[float, float, float] = (1.8, 1.30, 1.9),
        gust_force: tuple[float, float, float] = (1.76, 1.92, 1.4),
        angular_frequency: tuple[float, float, float] = (0.73, 1.37, 0.91),
        phase: tuple[float, float, float] = (0.0, 0.4, 1.1),
) -> None:
    """Run the GCOPTER wind-disturbance demonstration."""
    cfg, flight_cfg = utils.get_config()
    frequency = cfg.getint("frequency")
    velocity = flight_cfg.getfloat("velocity")
    min_distance_target = flight_cfg.getfloat("min_dist_target")
    simulation = MujocoSimulation(DEFAULT_SCENE_PATH)
    quad = simulation.quad
    trajectory_dt = quad.dt * frequency
    trajectory = _plan_gcopter_trajectory(simulation, velocity, trajectory_dt)
    simulation.set_trajectory_visualization(
        _trajectory_after_takeoff(
            trajectory, simulation.mission_waypoints[1])[:, :3])
    wind = GustingCrosswind(
        steady_force=steady_force,
        gust_force=gust_force,
        angular_frequency=angular_frequency,
        phase=phase,
    )
    trajectory_controller = TrajectoryController(
        _build_controller(CONTROLLER, quad, trajectory_dt),
        quad,
        trajectory,
        frequency,
    )
    control_step = partial(
        _wind_control_step, simulation, wind, trajectory_controller)
    reset_control = partial(
        _reset_wind_control, simulation, trajectory_controller)

    print(f"Planner: GCOPTER; controller: {CONTROLLER}.")
    print(f"Wind force: steady {np.asarray(wind.steady_force)} N NED, "
          f"gust amplitude {np.asarray(wind.gust_force)} N.")
    print("Press Backspace in the MuJoCo viewer to replay the wind-disturbed flight.")
    simulation.run_interactive(control_step, reset_control)
    distance = np.linalg.norm(quad.position - simulation.goal_position)
    result = "reached" if distance < min_distance_target else "missed"
    print(f"Wind demo finished {distance:.2f} m away from the goal ({result}).")
    if simulation.collision_detected:
        print("At least one collision occurred during the wind demo.")


if __name__ == "__main__":
    main()
