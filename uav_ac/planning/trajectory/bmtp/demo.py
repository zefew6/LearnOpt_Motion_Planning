"""Reproducible multi-initialization BMTP evaluation and MuJoCo flight.

    .venv/bin/python -m uav_ac.planning.trajectory.bmtp.demo
    .venv/bin/python -m uav_ac.planning.trajectory.bmtp.demo --replay RUN --viewer
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import clarabel
import cvxpy
import numpy as np

from uav_ac.planning.pipeline.bmtp_mission import (
    DEFAULT_BMTP_CONFIG, bmtp_controller_trajectory, load_bmtp_settings,
    scene_problem, scene_seed_paths, subdivide_path,
)
from uav_ac.simulation.mujoco_sim import BMTP_VILLAGE_SCENE_PATH, MujocoSimulation

from .collision import collisions
from .config import BMTPConfig, BMTPLimits
from .planner import BMTPPlanner, constraint_residuals
from .types import BMTPTrajectory


def initializations(simulation, settings, domain, obstacles):
    seeds = [subdivide_path(p, settings["scene"]["segments"]) for p in scene_seed_paths(simulation)]
    count = settings["demo"]["random_initializations"]
    deviation = settings["demo"]["perturbation"]
    if count < 0 or not np.isfinite(deviation) or deviation < 0:
        raise ValueError("random initializations and perturbation must be nonnegative")
    rng = np.random.default_rng(settings["demo"]["random_seed"])
    rejected = 0
    fixed_count = len(seeds)
    for i in range(count):
        for _ in range(500):
            seed = seeds[i % fixed_count].copy()
            seed[1:-1] += rng.uniform(-deviation, deviation, size=seed[1:-1].shape)
            if (np.all(domain.contains(seed)) and not collisions(np.stack((seed[:-1], seed[1:]), axis=1), obstacles)):
                seeds.append(seed)
                break
            rejected += 1
        else:
            raise RuntimeError("could not generate a valid perturbed initial path in 500 attempts")
    return seeds, rejected


def _store_curve(arrays, key, trajectory):
    arrays[key] = trajectory.evaluate(np.linspace(0, trajectory.duration, 501))
    arrays[key+"_control"] = trajectory.control_points
    arrays[key+"_segment_time"] = np.asarray(trajectory.segment_time)


def _restore_curve(arrays, key):
    return BMTPTrajectory(arrays[key+"_control"], float(arrays[key+"_segment_time"]))


def evaluate_initializations(simulation, settings):
    domain, obstacles, inflation = scene_problem(simulation, settings)
    config = BMTPConfig(**settings["planner"])
    limits = BMTPLimits(**settings["limits"][settings["limits_preset"]])
    seeds, rejected = initializations(simulation, settings, domain, obstacles)
    arrays, records = {}, []
    metadata = dict(settings=settings, obstacles=simulation.obstacles.tolist(),
                    bounds=simulation.space_limits.tolist(), inflation=inflation,
                    rejected_invalid_initial_paths=rejected,
                    scene_sha256=hashlib.sha256(BMTP_VILLAGE_SCENE_PATH.read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  cvxpy=cvxpy.__version__, clarabel=clarabel.__version__),
                    records=records)
    for i, seed in enumerate(seeds):
        result = BMTPPlanner(config).plan(seed, obstacles, domain, limits)
        certified = result.success
        if certified:
            residuals = constraint_residuals(result.trajectory, seed[0], seed[-1], domain, limits, config)
            certified = max(residuals.values()) <= config.feasibility_tolerance and not collisions(
                result.trajectory.control_points, obstacles, config.collision_tolerance/10, config.collision_max_depth)
        record = dict(id=i, kind="fixed" if i < 4 else "perturbed", initial_path=seed.tolist(),
                      status=result.status if certified or not result.success else "certification_failure",
                      message=result.message, success=bool(certified), converged=result.converged and bool(certified),
                      duration=result.trajectory.duration if certified else None,
                      initial_length=float(np.linalg.norm(np.diff(seed, axis=0), axis=1).sum()),
                      final_length=None, final_key=None, timings=result.timings,
                      obstacle_free_reference_duration=result.lower_bound_duration, history=[])
        if certified:
            key = f"run_{i:03d}_final"
            _store_curve(arrays, key, result.trajectory)
            record["final_key"] = key
            record["final_length"] = float(np.linalg.norm(np.diff(arrays[key], axis=0), axis=1).sum())
        for step in result.history:
            key = f"run_{i:03d}_step_{step.iteration:03d}"
            _store_curve(arrays, key, step.trajectory)
            record["history"].append(dict(iteration=step.iteration, key=key, accepted=step.accepted,
                collisions=step.collisions, new_tags=step.new_tags, duration=step.trajectory.duration,
                elapsed_seconds=step.elapsed_seconds, residuals=step.residuals))
        records.append(record)
        print(f"[{i+1}/{len(seeds)}] {record['kind']} {record['status']}: "
              f"T={record['duration']}; {result.timings['total_seconds']:.2f}s; {result.message}", flush=True)
    fixed = records[:len(scene_seed_paths(simulation))]
    durations = [r["duration"] for r in fixed if r["success"]]
    metadata["summary"] = dict(total=len(records), certified=sum(r["success"] for r in records),
        converged=sum(r["converged"] for r in records), fixed_certified=len(durations),
        fixed_duration_ratio=max(durations)/min(durations) if durations else None,
        demonstration_target_met=len(durations) == len(fixed) and bool(durations))
    return metadata, arrays


def save_run(output_dir, metadata, arrays):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir/"results.json").open("w") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    np.savez_compressed(output_dir/"trajectories.npz", **arrays)


def load_run(output_dir):
    with (Path(output_dir)/"results.json").open() as stream:
        metadata = json.load(stream)
    with np.load(Path(output_dir)/"trajectories.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return metadata, arrays


def show_overlays(simulation, metadata, arrays):
    colors = [(0.0, 0.45, 0.70, 0.8), (0.9, 0.4, 0.0, 0.8),
              (0.0, 0.6, 0.5, 0.8), (0.8, 0.4, 0.7, 0.8)]
    paths, rgba, dashed = [], [], []
    for index, record in enumerate(metadata["records"]):
        if record["kind"] != "fixed":
            continue
        color = colors[index % len(colors)]
        paths.append(record["initial_path"])
        rgba.append(color)
        dashed.append(True)
        if record["success"]:
            paths.append(arrays[record["final_key"]])
            rgba.append(color)
            dashed.append(False)
    simulation.set_planning_paths(paths, rgba, dashed=dashed)


def fly(metadata, arrays, route: int = 0, *, viewer: bool = False, video_path: Path | None = None):
    """Actual cascaded-controller flight; never move the vehicle by teleportation."""
    from uav_ac.control import CascadedController, TrajectoryController
    from uav_ac.simulation.recording import Mp4Recorder, default_camera
    import mujoco

    if metadata["scene_sha256"] != hashlib.sha256(BMTP_VILLAGE_SCENE_PATH.read_bytes()).hexdigest():
        raise ValueError("saved run scene differs from installed XML; refuse flight replay")
    if not 0 <= route < len(metadata["records"]):
        raise ValueError("flight route outside saved records")
    record = metadata["records"][route]
    if not record["success"]:
        raise ValueError("cannot fly an uncertified result")
    simulation = MujocoSimulation(BMTP_VILLAGE_SCENE_PATH, planning_path_capacity=8,
                                   record_actual_trajectory=viewer or video_path is not None)
    show_overlays(simulation, metadata, arrays)
    trajectory = _restore_curve(arrays, record["final_key"])
    domain, obstacles, _ = scene_problem(simulation, metadata["settings"])
    cfg = BMTPConfig(**metadata["settings"]["planner"])
    limits = BMTPLimits(**metadata["settings"]["limits"][metadata["settings"]["limits_preset"]])
    residuals = constraint_residuals(trajectory, simulation.start_position, simulation.goal_position, domain, limits, cfg)
    if max(residuals.values()) > cfg.feasibility_tolerance or collisions(trajectory.control_points, obstacles, cfg.collision_tolerance/10):
        raise ValueError("saved trajectory failed fresh flight certification")
    stride, dt = 10, simulation.quad.dt*10
    controller = CascadedController(simulation.quad.g, dt)
    samples = bmtp_controller_trajectory(trajectory, dt)
    tracker = TrajectoryController(controller, simulation.quad, samples, stride)
    if viewer:
        print("Dashed: initial paths. Solid, matching color: optimized paths. Blue trail: actual flight.")
        simulation.run_interactive(tracker.step, tracker.reset)
        return None
    recorder, renderer = None, None
    errors = []
    try:
        if video_path is not None:
            simulation.model.vis.global_.offwidth = 960
            simulation.model.vis.global_.offheight = 720
            renderer = mujoco.Renderer(simulation.model, height=720, width=960)
            recorder = Mp4Recorder(video_path, 960, 720, 30).__enter__()
            camera = default_camera(simulation.model)
        next_frame = 0.
        steps = int(np.ceil((trajectory.duration+3.)/simulation.quad.dt))
        for _ in range(steps):
            tracker.step()
            simulation.step()
            t = min(float(simulation.data.time), trajectory.duration)
            errors.append(float(np.linalg.norm(simulation.quad.position-trajectory.evaluate(t))))
            if recorder is not None and simulation.data.time >= next_frame:
                renderer.update_scene(simulation.data, camera=camera)
                recorder.write(renderer.render())
                next_frame += 1/30
    finally:
        if renderer is not None:
            renderer.close()
        if recorder is not None:
            recorder.close()
    return dict(route=route, collision=bool(simulation.collision_detected),
                goal_error=float(np.linalg.norm(simulation.quad.position-simulation.goal_position)),
                max_tracking_error=max(errors), rms_tracking_error=float(np.sqrt(np.mean(np.square(errors)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BMTP_CONFIG)
    parser.add_argument("--replay", type=Path, help="load a saved run without re-optimizing")
    parser.add_argument("--viewer", action="store_true", help="open MuJoCo after generating comparison artifacts")
    parser.add_argument("--check-flight", action="store_true", help="simulate every fixed result headlessly")
    parser.add_argument("--video", action="store_true", help="record actual flight to MP4 (requires an OpenGL backend)")
    args = parser.parse_args()
    if args.replay:
        output = args.replay
        metadata, arrays = load_run(output)
    else:
        settings = load_bmtp_settings(args.config)
        simulation = MujocoSimulation(BMTP_VILLAGE_SCENE_PATH, record_actual_trajectory=False)
        metadata, arrays = evaluate_initializations(simulation, settings)
        output = Path(settings["demo"]["output_dir"])/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_run(output, metadata, arrays)
    from uav_ac.visualization.bmtp import plot_results
    plot_results(metadata, arrays, output, animation=metadata["settings"]["demo"]["animation"])
    print(json.dumps(metadata["summary"], indent=2), flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    if args.check_flight:
        fixed = [record for record in metadata["records"] if record["kind"] == "fixed"]
        flights = [fly(metadata, arrays, i) if record["success"]
                   else dict(route=i, skipped="no certified trajectory")
                   for i, record in enumerate(fixed)]
        with (output/"flight_checks.json").open("w") as stream:
            json.dump(flights, stream, indent=2)
        print(json.dumps(flights, indent=2), flush=True)
    route = metadata["settings"]["scene"]["initial_route"]
    if args.video:
        report = fly(metadata, arrays, route, video_path=output/"flight.mp4")
        with (output/"recorded_flight.json").open("w") as stream:
            json.dump(report, stream, indent=2)
    if args.viewer:
        fly(metadata, arrays, route, viewer=True)


if __name__ == "__main__":
    main()
