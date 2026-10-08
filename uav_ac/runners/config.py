"""Load and validate interactive-flight YAML configuration."""

from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path

import yaml

from uav_ac.control import CascadedConfig
from uav_ac.simulation.wind_disturb import GustingCrosswind


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "simulation" / "models"
DEFAULT_CONFIG = ROOT / "configs" / "flight.yaml"
PLANNERS = {"none", "mini_snap", "gcopter", "gcs", "bmtp",
            "aerial_manipulator_minco"}
CONTROLLERS = {"cascaded", "mpc", "rl"}
TASKS = {"trajectory_tracking", "gate_racing", "aerial_pick_place"}


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
    flight_options = {
        "task", "scene", "planner", "controller", "speed", "control_dt", "wind",
        "visualize", "follow_camera", "duration", "rl", "cascaded", "bmtp", "gcopter",
        "gcs", "mpc", "wind_options", "pick_place", "aerial_manipulator_minco",
    }
    if config.get("task", "trajectory_tracking") != "aerial_pick_place":
        flight_options.add("seed")
    _only(config, flight_options, "flight")
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
    elif config["task"] == "aerial_pick_place":
        if (config["planner"] != "aerial_manipulator_minco"
                or config["controller"] != "cascaded"):
            raise ValueError(
                "aerial_pick_place requires an aerial manipulator pick/place scene, "
                "aerial_manipulator_minco, and cascaded controller")
        if config.get("wind", "none") != "none":
            raise ValueError("aerial pick/place requires wind: none")
        from uav_ac.scenes.loader import load_scene
        metadata = load_scene(scene_path)
        if metadata.space_limits is None or metadata.pick_place is None:
            raise ValueError(
                "aerial pick/place XML requires planning_bounds and pick/place numerics")
    elif config["planner"] == "none":
        raise ValueError("planner: none is only valid for task: gate_racing")
    config.setdefault("speed", 3.0)
    config.setdefault("duration", 20.0)
    _positive(config["duration"], "duration")
    config.setdefault("wind", "none")
    config.setdefault("visualize", False)
    config.setdefault("follow_camera", config["task"] == "gate_racing")
    _positive(config["speed"], "speed")
    if not isinstance(config["visualize"], bool):
        raise ValueError("visualize must be true or false")
    if not isinstance(config["follow_camera"], bool):
        raise ValueError("follow_camera must be true or false")
    if config["task"] != "aerial_pick_place":
        config.setdefault("seed", 7)
        if (isinstance(config["seed"], bool) or not isinstance(config["seed"], int)
                or config["seed"] < 0):
            raise ValueError("seed must be a nonnegative integer")
    if config["wind"] not in {"none", "fixed_gust"}:
        raise ValueError("wind must be none or fixed_gust")

    sections = {name: config.setdefault(name, {})
                for name in ("rl", "cascaded", "bmtp", "gcopter", "gcs", "mpc",
                             "wind_options", "aerial_manipulator_minco")}
    for name, values in sections.items():
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")
    # All component blocks may coexist in one flight YAML. Runtime dispatch
    # reads only the selected planner, controller, and wind implementation.

    from uav_ac.control.mpc_controller import MPCConfig
    from uav_ac.planning.trajectory.bmtp import BMTPConfig
    from uav_ac.planning.trajectory.gcopter import GCOPTERConfig
    from uav_ac.planning.trajectory.aerial_manipulator_minco import (
        AerialManipulatorMINCOConfig,
    )
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
    _only(sections["aerial_manipulator_minco"],
          {f.name for f in fields(AerialManipulatorMINCOConfig)},
          "aerial_manipulator_minco")
    AerialManipulatorMINCOConfig.from_mapping(sections["aerial_manipulator_minco"])
    pick_place = config.setdefault("pick_place", {})
    _only(pick_place, {
        "position_tolerance", "velocity_tolerance", "settle_time",
        "angular_velocity_tolerance", "record_joint_trace",
    }, "pick_place")
    if config["task"] == "aerial_pick_place":
        for name, default in (("position_tolerance", .02), ("velocity_tolerance", .05),
                              ("angular_velocity_tolerance", .2),
                              ("settle_time", .30)):
            pick_place.setdefault(name, default)
            _positive(pick_place[name], f"pick_place.{name}")
        if not isinstance(pick_place.get("record_joint_trace", False), bool):
            raise ValueError("pick_place.record_joint_trace must be true or false")
        pick_place.setdefault("record_joint_trace", False)
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


