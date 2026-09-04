import os
from typing import Literal

# GCOPTER solves many tiny dense/banded systems. Starting a BLAS worker pool
# for them costs more than the arithmetic on both desktop and onboard CPUs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from uav_ac import utils
from uav_ac.control import CascadedController, TrajectoryController
from uav_ac.planning.pipeline import (
    build_gcs_corridor as _build_gcs_corridor,
    build_mission_corridor as _build_firi_corridor,
    gcs_controller_trajectory as _gcs_controller_trajectory,
    generate_gcopter_mission as _generate_gcopter_trajectory,
    generate_gcs_trajectory as _generate_gcs_trajectory,
    generate_minimum_snap_mission as _generate_mission_trajectory,
)
from uav_ac.simulation.mujoco_sim import (
    DEFAULT_SCENE_PATH,
    GCS_BUILDING_SCENE_PATH,
    MujocoSimulation,
)


PlannerName = Literal["mini_snap", "gcopter", "gcs"]
ControllerName = Literal["cascaded", "mpc", "rl"]
VISUALIZE: bool = False
PLANNER: PlannerName = "gcopter"
CONTROLLER: ControllerName = "rl"


def _trajectory_after_takeoff(
        trajectory: np.ndarray,
        takeoff_waypoint: np.ndarray,
) -> np.ndarray:
    """Return the trajectory from the sample nearest the takeoff waypoint."""
    distances = np.linalg.norm(trajectory[:, :3] - takeoff_waypoint, axis=1)
    return trajectory[np.argmin(distances):]


def _plan_trajectory(
        planner: PlannerName,
        simulation: MujocoSimulation,
        velocity: float,
        trajectory_dt: float,
        *,
        visualize: bool | None = None,
) -> np.ndarray:
    """Run exactly one of the interchangeable planning examples."""
    visualize = VISUALIZE if visualize is None else visualize
    if planner == "mini_snap":
        print("Planner: MinimumSnap.")
        return _generate_mission_trajectory(
            simulation.mission_waypoints,
            simulation.obstacles,
            velocity,
            trajectory_dt,
        )
    if planner == "gcopter":
        corridor = _build_firi_corridor(
            simulation, simulation.mission_waypoints, visualize=visualize
        )
        return _generate_gcopter_trajectory(
            simulation.mission_waypoints,
            corridor,
            simulation.quad,
            velocity,
            trajectory_dt,
        )
    if planner == "gcs":
        corridor = _build_gcs_corridor(simulation, visualize=visualize)
        geometric = _generate_gcs_trajectory(
            simulation.mission_waypoints[[0, -1]], corridor
        )
        controller_trajectory = _gcs_controller_trajectory(
            geometric, velocity, trajectory_dt
        )
        duration = (len(controller_trajectory) - 1) * trajectory_dt
        peak_speed = np.max(np.linalg.norm(
            controller_trajectory[:, 3:6], axis=1
        ))
        print(
            f"GCS: {len(corridor.regions)} room regions, "
            f"{geometric.segment_count} selected segments, "
            f"{duration:.2f} s at {peak_speed:.2f} m/s peak speed."
        )
        return controller_trajectory
    raise ValueError(f"unsupported planner: {planner}")


def _build_controller(controller_name: ControllerName, quad, trajectory_dt: float):
    """Construct one controller behind the common tracking interface."""
    if controller_name == "cascaded":
        return CascadedController(quad.g, trajectory_dt)

    if controller_name == "mpc":
        # Lazy import: selecting the cascaded controller does not require acados.
        from uav_ac.control.mpc_controller import MPCConfig, MPCController

        return MPCController(
            quad,
            MPCConfig(
                dt=trajectory_dt,
                horizon_steps=10,
                nlp_solver_type="SQP_RTI",
            ),
        )

    if controller_name == "rl":
        run_dir = os.environ.get(
            "UAV_AC_RL_RUN",
            "runs/ppo_trajectory/exp01",
        )
        if not run_dir:
            raise ValueError(
                "the rl controller requires UAV_AC_RL_RUN to name a training run")
        # Loading SB3 and Torch is deferred until RL is explicitly selected.
        from uav_ac.control.rl_controller import RLController

        return RLController.from_run(run_dir, quad, device="cpu")

    raise ValueError(f"unsupported controller: {controller_name}")


def main() -> None:
    cfg, cfg_flight = utils.get_config()

    # Existing config name is kept for compatibility. Semantically this value
    # is the number of MuJoCo steps per trajectory/reference (and MPC) update.
    steps_per_reference = cfg.getint("frequency")

    velocity = cfg_flight.getfloat("velocity")
    min_distance_target = cfg_flight.getfloat("min_dist_target")

    scene_path = (
        GCS_BUILDING_SCENE_PATH if PLANNER == "gcs" else DEFAULT_SCENE_PATH
    )
    simulation = MujocoSimulation(scene_path)
    quad = simulation.quad

    trajectory_dt = quad.dt * steps_per_reference
    controller = _build_controller(CONTROLLER, quad, trajectory_dt)

    global_trajectory = _plan_trajectory(
        PLANNER, simulation, velocity, trajectory_dt
    )

    visible_trajectory = (
        global_trajectory if PLANNER == "gcs" else
        _trajectory_after_takeoff(
            global_trajectory, simulation.mission_waypoints[1]
        )
    )
    simulation.set_trajectory_visualization(visible_trajectory[:, :3])

    trajectory_controller = TrajectoryController(
        controller=controller,
        quad=quad,
        trajectory=global_trajectory,
        steps_per_reference=steps_per_reference,
    )

    print(f"Controller: {CONTROLLER}.")
    print("Press Backspace in the MuJoCo viewer to replay the flight.")
    simulation.run_interactive(
        trajectory_controller.step,
        trajectory_controller.reset,
    )

    distance_to_goal = np.linalg.norm(
        quad.position - simulation.goal_position
    )
    goal_has_been_reached = distance_to_goal < min_distance_target
    print(
        f"Flight finished {distance_to_goal:.2f} m away from the goal "
        f"({'reached' if goal_has_been_reached else 'missed'})."
    )
    if simulation.collision_detected:
        print("At least one collision occurred during the flight.")


if __name__ == "__main__":
    main()
