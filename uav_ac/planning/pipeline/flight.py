"""Compose the selected planner into a controller-ready flight trajectory."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from uav_ac.planning.pipeline import (
    CorridorPlanningConfig,
    build_gcs_corridor,
    build_mission_corridor,
    gcopter_controller_trajectory,
    gcs_controller_trajectory,
    generate_gcs_trajectory,
    generate_minimum_snap_mission,
)
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def plan_trajectory(config: dict, simulation: MujocoSimulation,
                    dt: float) -> np.ndarray:
    """Run the selected planner directly against one MuJoCo scene."""
    planner = config["planner"]
    speed = float(config["speed"])
    waypoints = np.asarray(simulation.mission_waypoints, dtype=float).copy()
    minimum = 3 if planner == "mini_snap" else 2
    if (waypoints.ndim != 2 or waypoints.shape[1] != 3
            or len(waypoints) < minimum or not np.all(np.isfinite(waypoints))):
        raise ValueError(
            f"scene requires at least {minimum} finite mission waypoints for {planner}")
    waypoints[0] = simulation.start_position

    if planner == "mini_snap":
        return generate_minimum_snap_mission(
            waypoints, simulation.obstacles, speed, dt)

    if planner == "bmtp":
        from uav_ac.planning.pipeline.bmtp_mission import generate_bmtp_mission
        from uav_ac.planning.trajectory.bmtp import BMTPConfig, BMTPLimits

        values = dict(config["bmtp"])
        scene_settings = {
            "initial_route": values.pop("initial_route", 0),
            "segments": values.pop("segments", 8),
            "clearance": values.pop("clearance", 0.05),
        }
        limits = {
            "velocity": speed,
            "acceleration": values.pop("acceleration", 3.0),
            "jerk": values.pop("jerk", 15.0),
            "snap": values.pop("snap", 30.0),
        }
        planner_settings = BMTPConfig(**values)
        limit_settings = BMTPLimits(**limits)
        return generate_bmtp_mission(
            simulation, dt, visualize=True,
            settings={"planner": {
                field.name: getattr(planner_settings, field.name)
                for field in fields(BMTPConfig)
            }, "limits": {
                field.name: getattr(limit_settings, field.name)
                for field in fields(BMTPLimits)
            }, "scene": scene_settings})

    corridor_config = CorridorPlanningConfig(rrt_seed=config["seed"])
    if planner == "gcs":
        from uav_ac.planning.trajectory.gcs import GCSConfig
        corridor = build_gcs_corridor(
            simulation, corridor_config, visualize=config["visualize"])
        geometric = generate_gcs_trajectory(
            waypoints[[0, -1]], corridor, GCSConfig(**config["gcs"]))
        return gcs_controller_trajectory(geometric, speed, dt)

    from uav_ac.planning.trajectory.gcopter import GCOPTER, GCOPTERConfig
    corridor = build_mission_corridor(
        simulation, waypoints, corridor_config, visualize=config["visualize"])
    quad = simulation.quad
    settings = {
        "length_per_piece": 1.5,
        "max_velocity": speed,
        "mass": quad.m,
        "gravity": quad.g,
        "min_thrust": 4.0 * quad.min_thrust,
        "max_thrust": 4.0 * quad.max_thrust,
        "max_tilt_angle": quad.max_tilt_angle,
        **config["gcopter"],
    }
    optimized = GCOPTER(GCOPTERConfig(**settings)).plan(
        waypoints[0], waypoints[-1], corridor.regions,
        fixed_corridor_boundaries=corridor.fixed_boundaries)
    return gcopter_controller_trajectory(optimized, dt)


