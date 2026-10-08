"""Initialized static maps and two-leg planning for an aerial pick/place flight."""

from collections.abc import Callable
import time

import numpy as np

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.trajectory.aerial_manipulator_minco import (
    AerialManipulatorMINCO, AerialManipulatorMINCOConfig,
    AerialManipulatorTrajectory, make_terminal_state,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.search_adapter import AerialAStarMaps
from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import quaternion_yaw


def pick_place_settings(simulation, config):
    scene = simulation.pick_place
    return {**({} if scene is None else scene.as_mapping()), **config["pick_place"]}


def _scene_occupancy(simulation, config):
    bounds = simulation.space_limits
    if bounds is None:
        raise ValueError("aerial pick/place scene requires planning_bounds")
    padding = max(.8, config.obstacle_clearance+.65)
    return GridMap.from_scene_geometries(
        simulation.scene_geometries, bounds[0]-padding, bounds[1]+padding,
        config.esdf_resolution)


def _leg_metrics(plan, planner, result, search_only):
    if search_only:
        return {**result.metrics, "exact_solution": result.exact_solution,
                "path_states": int(len(result.path))}
    metrics = dict(planner.last_metrics)
    metrics.update(validation_passed=bool(plan.validation_passed),
                   optimizer_converged=bool(plan.optimizer_converged),
                   optimizer_iterations=int(plan.iterations),
                   maximum_violation=float(plan.maximum_violation),
                   minimum_clearance=float(plan.minimum_clearance),
                   validation_sample_dt=float(plan.validation_sample_dt))
    states = plan.evaluate(np.linspace(0., plan.total_time, 101))
    metrics.update(
        planned_movement_time_s=float(plan.total_time),
        planned_average_base_speed_mps=float(np.linalg.norm(
            np.diff(states[:, :3], axis=0), axis=1).sum()/plan.total_time),
        planned_average_joint_speed_rad_s=(
            np.sum(np.abs(np.diff(states[:, 4:8], axis=0)), axis=0)/plan.total_time).tolist())
    return metrics


class PickPlacePlanner:
    """Own static planning maps for one scene/configuration lifetime.

    Construct a new instance for a changed scene or planner configuration.
    Planning and execution reset reuse these initialized maps.
    """

    def __init__(self, simulation, config, *, settings=None, planner_config=None):
        started = time.perf_counter()
        self.simulation, self.config = simulation, config
        self.settings = pick_place_settings(simulation, config) if settings is None else settings
        self.planner_config = (AerialManipulatorMINCOConfig.from_mapping(
            config["aerial_manipulator_minco"]) if planner_config is None else planner_config)
        cfg = self.planner_config
        phase = time.perf_counter()
        self.occupancy = _scene_occupancy(simulation, cfg)
        occupancy_seconds = time.perf_counter()-phase
        phase = time.perf_counter()
        self.esdf = ESDF.from_occupancy(self.occupancy)
        esdf_seconds = time.perf_counter()-phase
        phase = time.perf_counter()
        self.astar_maps = (AerialAStarMaps(
            self.occupancy, self.esdf,
            proxy_radius=simulation.robot.base_inscribed_collision_radius(),
            margin=cfg.rrt_obstacle_margin,
            grid_resolution=cfg.astar_grid_resolution,
            fallback_resolution=cfg.astar_fallback_grid_resolution,
            clearance_weight_m=cfg.astar_clearance_weight_m,
            clearance_offset_m=cfg.astar_clearance_offset_m,
            clearance_error_m=cfg.esdf_discretization_margin)
            if cfg.astar_guidance_enabled else None)
        self.initialization_seconds = time.perf_counter()-started
        self.initialization_metrics = dict(
            occupancy_seconds=occupancy_seconds, esdf_seconds=esdf_seconds,
            astar_map_seconds=time.perf_counter()-phase,
            esdf_grid_shape=list(self.occupancy.shape),
            esdf_grid_voxels=int(self.occupancy.occupied.size),
            esdf_grid_bounds=[self.occupancy.origin.tolist(), self.occupancy.upper.tolist()],
            esdf_resolution=float(self.occupancy.resolution))

    def plan(self, *, diagnostics=None, search_only=False,
             on_rrt_path: Callable[[str, np.ndarray], None] | None = None,
             on_minco_trajectory: Callable[
                 [str, AerialManipulatorTrajectory], None] | None = None):
        """Plan and validate both pick/place legs through the production path."""
        simulation, config = self.simulation, self.config
        planner_config, settings = self.planner_config, self.settings
        diagnostics = {} if diagnostics is None else diagnostics
        diagnostics.update(self.initialization_metrics)
        started = time.perf_counter()
        robot, bounds = simulation.robot, simulation.space_limits
        pick, place = (np.asarray(settings[key], dtype=float)
                       for key in ("pick_position_ned", "place_position_ned"))
        gap_open, gap_closed = float(settings["gripper_open"]), float(settings["gripper_closed"])
        occupancy, esdf = self.occupancy, self.esdf
        start_q = robot.configuration.copy()
        start = np.r_[start_q[:3], quaternion_yaw(start_q[3:7]), start_q[7:11]]
        pick_state, place_state = (make_terminal_state(
            robot, position, float(settings[f"{name}_yaw"]),
            np.asarray(settings[f"{name}_nominal_joints"], dtype=float),
            gripper_opening=gap, workspace_bounds=bounds)
            for name, position, gap in (("pick", pick, gap_open), ("place", place, gap_closed)))
        planner = AerialManipulatorMINCO(planner_config)
        plans, searches, legs, optimizer_seconds, optimizer_calls = {}, {}, {}, 0., 0
        requests = (("pick", start, pick_state, gap_open, False),
                    ("place", pick_state, place_state, gap_closed, True))
        for name, leg_start, goal, opening, carry in requests:
            rrt_callback = (None if on_rrt_path is None else
                            lambda states, leg_name=name: on_rrt_path(leg_name, states))
            try:
                result = planner.plan(leg_start, goal, robot=robot, esdf=esdf, quad=simulation.quad,
                    workspace_bounds=bounds, gripper_opening=opening, carry_payload=carry,
                    occupancy=occupancy, astar_maps=self.astar_maps, search_only=search_only,
                    on_rrt_path=rrt_callback)
                if not search_only and on_minco_trajectory is not None:
                    on_minco_trajectory(name, result)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                legs[name] = {**planner.last_metrics, "failure_reason": str(error)}
                diagnostics.update(plans=plans, searches=searches)
                if not search_only:
                    optimizer_seconds += float(planner.last_metrics.get("optimizer_seconds", 0.))
                    optimizer_calls += int(planner.last_metrics.get("objective_calls", 0))
                    diagnostics.update(
                        minco_optimizer_seconds=optimizer_seconds,
                        minco_optimizer_calls=optimizer_calls, legs=legs,
                        planning_seconds=time.perf_counter()-started+self.initialization_seconds)
                    raise
                continue
            if search_only:
                searches[name] = result
            else:
                plans[name] = result
                optimizer_seconds += float(planner.last_metrics.get("optimizer_seconds", 0.))
                optimizer_calls += int(planner.last_metrics.get("objective_calls", 0))
            legs[name] = _leg_metrics(result, planner, result, search_only)
            if not search_only:
                diagnostics.update(plans=plans, legs=legs)
        diagnostics.update(legs=legs, planning_seconds=time.perf_counter()-started+self.initialization_seconds)
        if search_only:
            diagnostics["searches"] = searches
        else:
            diagnostics.update(
                plans=plans, minco_optimizer_seconds=optimizer_seconds,
                minco_optimizer_calls=optimizer_calls)
        return {
            "searches" if search_only else "plans": searches if search_only else plans,
            "pick": pick, "place": place, "gap_open": gap_open, "gap_closed": gap_closed,
            "start_q": start_q, "esdf": esdf, "settings": settings,
        }

