"""Interactive MuJoCo planning and controller deployment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import math
import os
from pathlib import Path

import numpy as np
import yaml

# GCOPTER solves many small systems; extra BLAS workers reduce throughput.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from uav_ac.control import CascadedConfig, CascadedController, RLController, TrajectoryController
from uav_ac.planning.pipeline import (
    CorridorPlanningConfig,
    build_gcs_corridor,
    build_mission_corridor,
    gcopter_controller_trajectory,
    gcs_controller_trajectory,
    generate_gcs_trajectory,
    generate_minimum_snap_mission,
)
from uav_ac.simulation.mujoco_sim import ENU_TO_NED, MujocoSimulation
from uav_ac.simulation.wind_disturb import GustingCrosswind


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = Path(__file__).resolve().parent / "simulation" / "models"
DEFAULT_CONFIG = ROOT / "configs" / "flight.yaml"
PLANNERS = {"none", "mini_snap", "gcopter", "gcs", "bmtp"}
CONTROLLERS = {"cascaded", "mpc", "rl"}
TASKS = {"trajectory_tracking", "gate_racing"}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _only(values, allowed, label):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = values.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} option: {sorted(unknown)[0]}")


def _positive(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{label} must be a positive finite number")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load and validate the intentionally small interactive-flight YAML."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=_UniqueLoader)
    _only(config, {
        "task", "scene", "planner", "controller", "speed", "control_dt", "wind",
        "visualize", "follow_camera", "seed", "rl", "cascaded", "bmtp", "gcopter",
        "gcs", "mpc", "wind_options",
    }, "flight")
    for required in ("scene", "planner", "controller"):
        if required not in config:
            raise ValueError(f"flight.{required} is required")

    scene = config["scene"]
    if (not isinstance(scene, str) or not scene
            or Path(scene).name != scene or Path(scene).suffix not in {"", ".xml"}):
        raise ValueError("scene must be an XML stem or filename from simulation/models")
    scene_path = MODEL_DIRECTORY / (scene if scene.endswith(".xml") else f"{scene}.xml")
    if not scene_path.is_file():
        raise ValueError(f"scene does not exist: {scene_path}")
    config["scene"] = str(scene_path.resolve())

    if config["planner"] not in PLANNERS:
        raise ValueError(f"planner must be one of: {', '.join(sorted(PLANNERS))}")
    if config["controller"] not in CONTROLLERS:
        raise ValueError(f"controller must be one of: {', '.join(sorted(CONTROLLERS))}")
    config.setdefault("task", "trajectory_tracking")
    if config["task"] not in TASKS:
        raise ValueError(f"task must be one of: {', '.join(sorted(TASKS))}")
    if config["task"] == "gate_racing":
        if scene != "gate_racing" or config["planner"] != "none" or config["controller"] != "rl":
            raise ValueError("gate_racing requires scene: gate_racing, planner: none, and controller: rl")
        if config.get("wind", "none") != "none":
            raise ValueError("gate_racing deployment does not support flight wind options")
    elif config["planner"] == "none":
        raise ValueError("planner: none is only valid for task: gate_racing")
    config.setdefault("speed", 3.0)
    config.setdefault("wind", "none")
    config.setdefault("visualize", False)
    config.setdefault("follow_camera", False)
    config.setdefault("seed", 7)
    _positive(config["speed"], "speed")
    if not isinstance(config["visualize"], bool):
        raise ValueError("visualize must be true or false")
    if not isinstance(config["follow_camera"], bool):
        raise ValueError("follow_camera must be true or false")
    if (isinstance(config["seed"], bool) or not isinstance(config["seed"], int)
            or config["seed"] < 0):
        raise ValueError("seed must be a nonnegative integer")
    if config["wind"] not in {"none", "fixed_gust"}:
        raise ValueError("wind must be none or fixed_gust")

    sections = {name: config.setdefault(name, {})
                for name in ("rl", "cascaded", "bmtp", "gcopter", "gcs", "mpc", "wind_options")}
    for name, values in sections.items():
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")
    # All component blocks may coexist in one flight YAML. Runtime dispatch
    # reads only the selected planner, controller, and wind implementation.

    from uav_ac.control.mpc_controller import MPCConfig
    from uav_ac.planning.trajectory.bmtp import BMTPConfig
    from uav_ac.planning.trajectory.gcopter import GCOPTERConfig
    from uav_ac.planning.trajectory.gcs import GCSConfig

    bmtp_special = {"initial_route", "segments", "clearance",
                     "acceleration", "jerk", "snap"}
    _only(sections["bmtp"], bmtp_special | {f.name for f in fields(BMTPConfig)}, "bmtp")
    _only(sections["gcopter"], {
        f.name for f in fields(GCOPTERConfig)
    } - {"max_velocity", "mass", "gravity", "min_thrust", "max_thrust",
         "max_tilt_angle"}, "gcopter")
    _only(sections["gcs"], {f.name for f in fields(GCSConfig)}, "gcs")
    _only(sections["mpc"], {f.name for f in fields(MPCConfig)} - {"dt"}, "mpc")
    _only(sections["wind_options"], {f.name for f in fields(GustingCrosswind)},
          "wind_options")
    _only(sections["rl"], {"checkpoint", "device"}, "rl")
    _only(sections["cascaded"], {f.name for f in fields(CascadedConfig)}, "cascaded")

    if config["controller"] == "rl":
        if "control_dt" in config:
            raise ValueError("control_dt is owned by the RL checkpoint")
        checkpoint = sections["rl"].get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("rl.checkpoint is required for controller: rl")
        checkpoint_path = (path.parent / checkpoint).resolve()
        if not checkpoint_path.is_file() or checkpoint_path.suffix != ".zip":
            raise ValueError("rl.checkpoint must name an existing .zip model")
        sections["rl"]["checkpoint"] = str(checkpoint_path)
        sections["rl"].setdefault("device", "cpu")
    else:
        config.setdefault("control_dt", 0.01)
        _positive(config["control_dt"], "control_dt")
    return config


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


def _print_summary(config, trajectory, dt):
    positions = trajectory[:, :3]
    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    peak_speed = float(np.linalg.norm(trajectory[:, 3:6], axis=1).max())
    print(f"Flight: scene={Path(config['scene']).stem} | planner={config['planner']} | "
          f"controller={config['controller']} | wind={config['wind']}")
    print(f"Trajectory: duration={(len(trajectory)-1)*dt:.2f}s | length={length:.2f}m | "
          f"peak_speed={peak_speed:.2f}m/s | samples={len(trajectory)}")


def run(config: dict) -> np.ndarray | dict:
    """Plan and fly once; ordinary viewer runs intentionally write no files."""
    if config.get("task", "trajectory_tracking") == "gate_racing":
        return _run_gate_racing(config)
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


def _run_gate_racing(config: dict):
    """Replay a reference-free gate-racing policy in the native viewer."""
    from uav_ac.rl.tasks.gate_racing.evaluation import replay

    checkpoint = Path(config["rl"]["checkpoint"])
    result = replay(
        checkpoint.parent,
        device=config["rl"]["device"],
        seed=config["seed"],
    )
    print(f"Finished gate racing: success={result['success_rate']:.3f} | "
          f"gates={result['mean_gates_passed']:.2f} | "
          f"collision={result['collision_rate']:.3f}")
    return result


def main(config_path: str | Path = DEFAULT_CONFIG) -> np.ndarray | dict:
    return run(load_config(config_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
