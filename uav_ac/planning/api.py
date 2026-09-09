"""Unified configuration adapter for controller-ready planning."""

from dataclasses import fields

import numpy as np


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict")
    return value


def _keys(values, allowed, label):
    unknown = values.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(map(str, unknown)))}")


def _requires(simulation, names, label):
    missing = [name for name in names if getattr(simulation, name, None) is None]
    if missing:
        raise ValueError(f"{label} scene requires: {', '.join(missing)}")


def plan(config: dict, simulation, dt: float, *, seed=42, visualize=False) -> np.ndarray:
    """Return N x 10 samples (position, velocity, acceleration, yaw).

    Config contains ``name``, ``limits`` (velocity defaults to 3 m/s),
    ``options`` (native BMTP/GCOPTER/GCS dataclass fields; none for mini_snap),
    and ``initial_path``. The latter selects ``source=scene_waypoints`` or
    ``random_open_field``; BMTP uses ``scene_route`` with ``index=0``,
    ``segments=8`` and ``clearance=0.15``. BMTP limits additionally accept
    acceleration, jerk and snap. Its optimized timing is preserved.
    """
    _mapping(config, "config")
    _keys(config, ("name", "limits", "options", "initial_path"), "planning fields")
    name = config.get("name")
    if name not in ("mini_snap", "gcopter", "gcs", "bmtp"):
        raise ValueError(f"unsupported planner: {name}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    limits = _mapping(config.get("limits", {}), "limits")
    options = _mapping(config.get("options", {}), "options")
    initial = _mapping(config.get("initial_path", {}), "initial_path")
    _keys(limits, ("velocity", "acceleration", "jerk", "snap")
          if name == "bmtp" else ("velocity",), f"{name} limits")
    velocity = limits.get("velocity", 3.0)
    if not np.isfinite(velocity) or velocity <= 0:
        raise ValueError("limits.velocity must be finite and positive")

    from .trajectory.bmtp import BMTPConfig, BMTPLimits
    from .trajectory.gcopter import GCOPTERConfig
    from .trajectory.gcs import GCSConfig

    native = {"bmtp": BMTPConfig, "gcopter": GCOPTERConfig, "gcs": GCSConfig}
    _keys(options, [field.name for field in fields(native[name])]
          if name in native else (), f"{name} planner options")
    if name == "gcopter" and "max_velocity" in options:
        raise ValueError("use limits.velocity instead of options.max_velocity")
    if name in native:
        native[name](**options)
    _keys(initial, ("source", "index", "segments", "clearance")
          if name == "bmtp" else ("source",), "initial_path fields")
    source = initial.get("source", "scene_route" if name == "bmtp" else "scene_waypoints")
    if name == "bmtp":
        if source != "scene_route":
            raise ValueError("bmtp initial_path.source must be scene_route")
        BMTPLimits(**limits)
        _requires(simulation, ("model", "data", "_body_id", "space_limits",
                               "obstacles", "start_position", "goal_position"), "bmtp")
        from .pipeline.bmtp_mission import generate_bmtp_mission

        return generate_bmtp_mission(simulation, dt, visualize=visualize, settings={
            "planner": dict(options), "limits": dict(limits),
            "scene": {"initial_route": initial.get("index", 0),
                      "segments": initial.get("segments", 8),
                      "clearance": initial.get("clearance", 0.15)},
        })
    if source not in ("scene_waypoints", "random_open_field"):
        raise ValueError(f"{name} initial_path.source must be scene_waypoints or random_open_field")
    _requires(simulation, ("start_position", "obstacles"), name)
    if name == "gcopter":
        _requires(simulation, ("quad", "space_limits", "get_planning_obstacle_points"), name)
    elif name == "gcs":
        _requires(simulation, ("space_limits", "gcs_guide_paths"), name)

    from uav_ac import runtime

    if source == "random_open_field":
        if np.asarray(simulation.obstacles).size:
            raise ValueError("random_open_field requires an obstacle-free scene")
        waypoints = runtime.sample_open_field_mission(simulation, seed)
    else:
        _requires(simulation, ("mission_waypoints",), source)
        waypoints = np.asarray(simulation.mission_waypoints, dtype=float).copy()
    waypoints = np.asarray(waypoints, dtype=float).copy()
    minimum = 3 if name == "mini_snap" else 2
    if (waypoints.ndim != 2 or waypoints.shape[1] != 3
            or len(waypoints) < minimum or not np.all(np.isfinite(waypoints))):
        raise ValueError(f"{source} requires at least {minimum} finite 3D waypoints")
    waypoints[0] = simulation.start_position
    if not options:
        return runtime.plan_trajectory(name, simulation, velocity, dt,
                                       visualize=visualize, waypoints=waypoints)

    from . import pipeline

    if name == "gcs":
        corridor = pipeline.build_gcs_corridor(simulation, visualize=visualize)
        geometric = pipeline.generate_gcs_trajectory(
            waypoints[[0, -1]], corridor, GCSConfig(**options))
        return pipeline.gcs_controller_trajectory(geometric, velocity, dt)

    from .trajectory.gcopter import GCOPTER

    quad = simulation.quad
    settings = dict(length_per_piece=1.5, max_velocity=velocity,
                    mass=quad.m, gravity=quad.g, min_thrust=4 * quad.min_thrust,
                    max_thrust=4 * quad.max_thrust, max_tilt_angle=quad.max_tilt_angle)
    settings.update(options)
    corridor = pipeline.build_mission_corridor(simulation, waypoints, visualize=visualize)
    trajectory = GCOPTER(GCOPTERConfig(**settings)).plan(
        waypoints[0], waypoints[-1], corridor.regions,
        fixed_corridor_boundaries=corridor.fixed_boundaries)
    return pipeline.gcopter_controller_trajectory(trajectory, dt)
