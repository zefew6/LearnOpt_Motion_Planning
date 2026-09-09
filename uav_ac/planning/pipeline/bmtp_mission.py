"""MuJoCo/mission adapter; the BMTP library itself is simulator-independent."""

from pathlib import Path
import re

import numpy as np
import yaml

from ..trajectory.bmtp import BMTPConfig, BMTPLimits, BMTPPlanner, BMTPTrajectory
from ..trajectory.bmtp.collision import box, collisions, offset


DEFAULT_BMTP_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "bmtp.yaml"


def load_bmtp_settings(path: str | Path = DEFAULT_BMTP_CONFIG) -> dict:
    with Path(path).open() as stream:
        settings = yaml.safe_load(stream)
    BMTPConfig(**settings["planner"])
    BMTPLimits(**settings["limits"][settings["limits_preset"]])
    if settings["scene"]["clearance"] < 0 or settings["scene"]["segments"] < 1:
        raise ValueError("BMTP clearance must be nonnegative and segments positive")
    return settings


def subdivide_path(path: np.ndarray, segments: int) -> np.ndarray:
    """Preserve every corner while splitting longest current pieces to fixed M."""
    path = np.asarray(path, float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if segments < len(lengths) or np.any(lengths <= 1e-8):
        raise ValueError("not enough segments or duplicate seed waypoint")
    counts = np.ones(len(lengths), dtype=int)
    for _ in range(segments-len(lengths)):
        counts[np.argmax(lengths/counts)] += 1
    return np.vstack([path[0], *[np.linspace(a, b, count+1)[1:]
                               for a, b, count in zip(path[:-1], path[1:], counts)]])


def scene_seed_paths(simulation) -> list[np.ndarray]:
    import mujoco
    from uav_ac.simulation.mujoco_sim import ENU_TO_NED

    groups = {}
    for site_id in range(simulation.model.nsite):
        name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if not name.startswith("bmtp_route_"):
            continue
        match = re.fullmatch(r"bmtp_route_(\d{2})_(\d{2})", name)
        if match is None:
            raise ValueError(f"invalid BMTP route site: {name}")
        route, point = map(int, match.groups())
        groups.setdefault(route, {})[point] = ENU_TO_NED @ simulation.data.site_xpos[site_id]
    if not groups or sorted(groups) != list(range(len(groups))):
        raise ValueError("scene requires consecutively numbered bmtp_route sites")
    paths = []
    for route in sorted(groups):
        points = groups[route]
        if sorted(points) != list(range(len(points))) or len(points) < 2:
            raise ValueError("BMTP route point indices must be consecutive")
        path = np.asarray([points[i] for i in range(len(points))])
        if not (np.allclose(path[0], simulation.start_position) and np.allclose(path[-1], simulation.goal_position)):
            raise ValueError("BMTP routes must share scene start and goal")
        paths.append(path)
    return paths


def scene_problem(simulation, settings):
    # The XML vehicle uses box/capsule collision geoms; geom_rbound plus offset
    # from body origin encloses every collidable component at every attitude.
    model = simulation.model
    ids = np.flatnonzero((model.geom_bodyid == simulation._body_id)
                         & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    if not len(ids):
        raise ValueError("BMTP scene requires collidable vehicle geometry")
    radius = float(np.max(np.linalg.norm(model.geom_pos[ids], axis=1)+model.geom_rbound[ids]))
    margin = radius + float(settings["scene"]["clearance"])
    domain = offset(box(*simulation.space_limits), -margin)
    obstacles = [offset(box(obstacle[::2], obstacle[1::2]), margin) for obstacle in simulation.obstacles]
    return domain, obstacles, margin


def bmtp_controller_trajectory(trajectory: BMTPTrajectory, dt: float) -> np.ndarray:
    samples = trajectory.sample(dt)
    velocities = samples[:, 3:6]
    yaw = np.zeros(len(samples))
    moving = np.linalg.norm(velocities[:, :2], axis=1) > 1e-6
    last = 0.
    for i in range(len(yaw)):
        if moving[i]:
            last = np.arctan2(velocities[i, 1], velocities[i, 0])
        yaw[i] = last
    return np.column_stack((samples, np.unwrap(yaw)))


def generate_bmtp_mission(simulation, dt: float, config_path=DEFAULT_BMTP_CONFIG,
                          *, visualize: bool = False, settings: dict | None = None) -> np.ndarray:
    """Plan without retiming; explicit settings bypass the legacy YAML loader.

    Settings use ``planner``, ``scene`` and flat ``limits`` dictionaries, or
    the legacy ``limits_preset``/named-limits layout used by the demo.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    settings = load_bmtp_settings(config_path) if settings is None else settings
    config = BMTPConfig(**settings["planner"])
    limit_values = settings["limits"]
    if "limits_preset" in settings:
        limit_values = limit_values[settings["limits_preset"]]
    limits = BMTPLimits(**limit_values)
    scene = settings["scene"]
    if not np.isfinite(scene["clearance"]) or scene["clearance"] < 0:
        raise ValueError("BMTP clearance must be finite and nonnegative")
    if type(scene["segments"]) is not int or scene["segments"] < 1:
        raise ValueError("BMTP segments must be a positive integer")
    if type(scene["initial_route"]) is not int or scene["initial_route"] < 0:
        raise ValueError("initial_route must be a nonnegative integer")
    domain, obstacles, margin = scene_problem(simulation, settings)
    seeds = scene_seed_paths(simulation)
    route = settings["scene"]["initial_route"]
    if not 0 <= route < len(seeds):
        raise ValueError("initial_route index outside scene routes")
    path = subdivide_path(seeds[route], settings["scene"]["segments"])
    result = BMTPPlanner(config).plan(path, obstacles, domain, limits)
    if not result.success:
        raise RuntimeError(f"BMTP has no certified executable trajectory: {result.status}: {result.message}")
    # Recheck every obstacle at a tighter resolution before handing off to flight.
    if collisions(result.trajectory.control_points, obstacles, config.collision_tolerance/10, config.collision_max_depth):
        raise RuntimeError("BMTP final collision certification failed")
    simulation.bmtp_result = result
    if visualize:
        seeds = [subdivide_path(seed, settings["scene"]["segments"])
                 for seed in scene_seed_paths(simulation)]
        colors = [(0.0, 0.45, 0.70, 0.8), (0.9, 0.4, 0.0, 0.8),
                  (0.0, 0.6, 0.5, 0.8), (0.8, 0.4, 0.7, 0.8),
                  (0.1, 0.1, 0.1, 0.95)]
        final_path = result.trajectory.evaluate(
            np.linspace(0.0, result.trajectory.duration, 501))
        simulation.set_planning_paths(
            [*seeds, final_path], colors, dashed=[True, True, True, True, False])
    print(f"BMTP: {result.status}; T={result.trajectory.duration:.3f}s; "
          f"planning={result.timings['total_seconds']:.3f}s; inflation={margin:.3f}m; "
          f"limits={settings.get('limits_preset', 'explicit')}")
    return bmtp_controller_trajectory(result.trajectory, dt)
