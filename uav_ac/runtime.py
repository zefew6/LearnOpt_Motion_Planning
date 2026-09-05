"""Runtime composition helpers for the interactive UAV example."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np

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
    ENU_TO_NED,
    GCS_BUILDING_SCENE_PATH,
    OPEN_FIELD_SCENE_PATH,
    MujocoSimulation,
)
from uav_ac.simulation.wind_disturb import GustingCrosswind

PlannerName = Literal["mini_snap", "gcopter", "gcs"]
ControllerName = Literal["cascaded", "mpc", "rl"]


def trajectory_after_takeoff(trajectory: np.ndarray, takeoff_waypoint: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(trajectory[:, :3] - takeoff_waypoint, axis=1)
    return trajectory[np.argmin(distances):]


def plan_trajectory(
        planner: PlannerName,
        simulation: MujocoSimulation,
        velocity: float,
        trajectory_dt: float,
        *,
        visualize: bool = False,
        waypoints: np.ndarray | None = None,
) -> np.ndarray:
    if waypoints is None:
        waypoints = np.asarray(simulation.mission_waypoints, dtype=float).copy()
        waypoints[0] = simulation.start_position
    else:
        waypoints = np.asarray(waypoints, dtype=float)
    if planner == "mini_snap":
        print("Planner: MinimumSnap.")
        return _generate_mission_trajectory(
            waypoints, simulation.obstacles, velocity, trajectory_dt)
    if planner == "gcopter":
        corridor = _build_firi_corridor(simulation, waypoints, visualize=visualize)
        return _generate_gcopter_trajectory(
            waypoints, corridor, simulation.quad, velocity, trajectory_dt)
    if planner == "gcs":
        corridor = _build_gcs_corridor(simulation, visualize=visualize)
        geometric = _generate_gcs_trajectory(waypoints[[0, -1]], corridor)
        trajectory = _gcs_controller_trajectory(geometric, velocity, trajectory_dt)
        duration = (len(trajectory) - 1) * trajectory_dt
        peak_speed = np.max(np.linalg.norm(trajectory[:, 3:6], axis=1))
        print(
            f"GCS: {len(corridor.regions)} room regions, "
            f"{geometric.segment_count} selected segments, "
            f"{duration:.2f} s at {peak_speed:.2f} m/s peak speed."
        )
        return trajectory
    raise ValueError(f"unsupported planner: {planner}")


def sample_open_field_mission(simulation: MujocoSimulation, seed: int | None) -> np.ndarray:
    from uav_ac.rl.trajectory_bank import sample_open_field_waypoints

    rng = np.random.default_rng(seed)
    patterns = ("s_curve", "arc", "zigzag", "random_turns", "climb_dive")
    return sample_open_field_waypoints(
        simulation.start_position,
        navigation_waypoint_count=int(rng.integers(5, 11)),
        rng=rng,
        config={
            "horizontal_bounds": [[-90.0, -90.0], [90.0, 90.0]],
            "altitude": [0.8, 4.5],
            "takeoff_altitude": 1.2,
            "segment_length": [12.0, 50.0],
            "path_length": [60.0, 300.0],
        },
        pattern=str(rng.choice(patterns)),
    )


def wind_control_callbacks(
        simulation: MujocoSimulation,
        trajectory_controller: TrajectoryController,
        enabled: bool,
        wind: GustingCrosswind | None = None,
) -> tuple[Callable[[], None], Callable[[], None]]:
    if not enabled:
        return trajectory_controller.step, trajectory_controller.reset
    wind = GustingCrosswind() if wind is None else wind

    def control_step() -> None:
        force_ned = wind.force_ned(float(simulation.data.time))
        simulation.set_external_force_world(ENU_TO_NED @ force_ned)
        trajectory_controller.step()

    def reset_control() -> None:
        trajectory_controller.reset()
        simulation.set_external_force_world(np.zeros(3))

    return control_step, reset_control


def build_controller(
        controller_name: ControllerName,
        quad,
        trajectory_dt: float,
        run_dir: str | Path | None = None,
):
    if controller_name == "cascaded":
        return CascadedController(quad.g, trajectory_dt)
    if controller_name == "mpc":
        from uav_ac.control.mpc_controller import MPCConfig, MPCController

        return MPCController(
            quad,
            MPCConfig(dt=trajectory_dt, horizon_steps=10, nlp_solver_type="SQP_RTI"),
        )
    if controller_name == "rl":
        selected_run_dir = run_dir or os.environ.get(
            "UAV_AC_RL_RUN", "runs/ppo_trajectory/exp01")
        if not selected_run_dir:
            raise ValueError(
                "the rl controller requires UAV_AC_RL_RUN to name a training run")
        from uav_ac.control.rl_controller import RLController

        return RLController.from_run(selected_run_dir, quad, device="cpu")
    raise ValueError(f"unsupported controller: {controller_name}")


def scene_path(planner: PlannerName, scene: str) -> Path:
    if planner == "gcs":
        return GCS_BUILDING_SCENE_PATH
    if scene == "open_field":
        return OPEN_FIELD_SCENE_PATH
    return DEFAULT_SCENE_PATH


def mission_for_scene(
        simulation: MujocoSimulation,
        selected_scene: Path,
        seed: int | None,
        open_field_speed: float,
) -> tuple[np.ndarray, float | None]:
    if selected_scene == OPEN_FIELD_SCENE_PATH:
        waypoints = sample_open_field_mission(simulation, seed)
        simulation.set_goal_position(waypoints[-1])
        return waypoints, open_field_speed
    return simulation.mission_waypoints, None


def plan_scene_trajectory(
        planner: PlannerName,
        simulation: MujocoSimulation,
        selected_scene: Path,
        waypoints: np.ndarray,
        trajectory_dt: float,
        velocity: float,
        *,
        visualize: bool = False,
) -> np.ndarray:
    if planner in ("gcopter", "mini_snap") and len(waypoints) > 2:
        # Scene missions insert waypoint_00 directly above the initialized
        # vehicle position for takeoff.  Let one planner solve span the full
        # mission without that artificial intermediate waypoint.
        mission_waypoints = np.vstack((waypoints[0], waypoints[2:]))
        return plan_trajectory(
            planner,
            simulation,
            velocity,
            trajectory_dt,
            visualize=visualize,
            waypoints=mission_waypoints,
        )
    return plan_trajectory(
        planner,
        simulation,
        velocity,
        trajectory_dt,
        visualize=visualize,
        waypoints=waypoints,
    )


def run_interactive_flight(
        simulation: MujocoSimulation,
        controller: object,
        trajectory: np.ndarray,
        steps_per_reference: int,
        min_distance_target: float,
        selected_scene: Path,
        controller_name: ControllerName,
        wind_enabled: bool,
) -> None:
    tracker = TrajectoryController(
        controller=controller,
        quad=simulation.quad,
        trajectory=trajectory,
        steps_per_reference=steps_per_reference,
    )
    control_step, reset_control = wind_control_callbacks(
        simulation, tracker, wind_enabled)
    print(
        f"Controller: {controller_name}; scene: {selected_scene.stem}; "
        f"wind: {wind_enabled}."
    )
    print("Press Backspace in the MuJoCo viewer to replay the flight.")
    simulation.run_interactive(
        control_step,
        reset_control,
        chase_camera=(selected_scene == OPEN_FIELD_SCENE_PATH),
    )
    distance_to_goal = float(np.linalg.norm(
        simulation.quad.position - simulation.goal_position))
    result = "reached" if distance_to_goal < min_distance_target else "missed"
    print(f"Flight finished {distance_to_goal:.2f} m away from the goal ({result}).")
    if simulation.collision_detected:
        print("At least one collision occurred during the flight.")


__all__ = [
    "ControllerName", "PlannerName", "build_controller", "mission_for_scene",
    "plan_scene_trajectory", "plan_trajectory", "run_interactive_flight",
    "sample_open_field_mission", "scene_path", "trajectory_after_takeoff",
    "wind_control_callbacks",
]
