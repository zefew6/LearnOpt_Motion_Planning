"""Generate deterministic, offscreen comparison videos and FIRI images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from uav_ac import utils
from uav_ac.main import _build_controller, _plan_trajectory, _trajectory_after_takeoff
from uav_ac.control import TrajectoryController
from uav_ac.planning.pipeline import build_mission_corridor
from uav_ac.simulation.recording import default_camera, render_offscreen_frame, save_png
from uav_ac.simulation.mujoco_sim import (
    DEFAULT_SCENE_PATH,
    GCS_BUILDING_SCENE_PATH,
    MujocoSimulation,
)


ExperimentName = Literal["gcopter-cascaded", "minisnap-cascaded", "gcs-mpc"]

EXPERIMENTS: dict[ExperimentName, tuple[str, str]] = {
    "gcopter-cascaded": ("gcopter", "cascaded"),
    "minisnap-cascaded": ("mini_snap", "cascaded"),
    "gcs-mpc": ("gcs", "mpc"),
}

RECORDING_FOVY = 45.0
GCS_RECORDING_LOOKAT = (12.0, -7.0, 3.0)
GCS_RECORDING_AZIMUTH = -5.0
GCS_RECORDING_ELEVATION = -5.0
DEFAULT_RECORDING_AZIMUTH = 45.0
DEFAULT_RECORDING_ELEVATION = -32.0


def _recording_camera(
        model,
        *,
        lookat: tuple[float, float, float] | None = None,
        azimuth: float = DEFAULT_RECORDING_AZIMUTH,
        elevation: float = DEFAULT_RECORDING_ELEVATION,
) -> object:
    """Return the fixed camera used by a comparison video."""
    camera = default_camera(model)
    model.vis.global_.fovy = RECORDING_FOVY
    if lookat is not None:
        camera.lookat[:] = lookat
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def record_experiment(
        experiment: ExperimentName,
        output_path: str | Path,
        *,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        hold_seconds: float = 2.0,
) -> Path:
    """Plan and record one experiment without starting a MuJoCo viewer."""
    if hold_seconds < 0.0:
        raise ValueError("hold_seconds must be non-negative")

    planner, controller_name = EXPERIMENTS[experiment]
    cfg, flight_cfg = utils.get_config()
    steps_per_reference = cfg.getint("frequency")
    velocity = flight_cfg.getfloat("velocity")
    scene_path = GCS_BUILDING_SCENE_PATH if planner == "gcs" else DEFAULT_SCENE_PATH

    simulation = MujocoSimulation(scene_path)
    trajectory_dt = simulation.quad.dt * steps_per_reference
    trajectory = _plan_trajectory(
        planner, simulation, velocity, trajectory_dt, visualize=False,
    )
    visible_trajectory = (
        trajectory if planner == "gcs" else
        _trajectory_after_takeoff(trajectory, simulation.mission_waypoints[1])
    )
    simulation.set_trajectory_visualization(visible_trajectory[:, :3])

    tracker = TrajectoryController(
        controller=_build_controller(controller_name, simulation.quad, trajectory_dt),
        quad=simulation.quad,
        trajectory=trajectory,
        steps_per_reference=steps_per_reference,
    )
    trajectory_seconds = (len(trajectory) - 1) * trajectory_dt
    total_seconds = trajectory_seconds + hold_seconds
    max_steps = max(1, int(round(total_seconds / simulation.quad.dt)))
    output_path = Path(output_path)
    camera = (
        _recording_camera(
            simulation.model,
            lookat=GCS_RECORDING_LOOKAT,
            azimuth=GCS_RECORDING_AZIMUTH,
            elevation=GCS_RECORDING_ELEVATION,
        )
        if experiment == "gcs-mpc"
        else _recording_camera(simulation.model)
    )
    print(
        f"Recording {experiment}: {trajectory_seconds:.2f} s flight + "
        f"{hold_seconds:.2f} s terminal hold -> {output_path}")
    simulation.run_recorded(
        tracker.step,
        max_steps=max_steps,
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
        camera=camera,
    )
    distance = float(np.linalg.norm(
        simulation.quad.position - simulation.goal_position))
    print(
        f"Finished {experiment}: {distance:.2f} m from goal, "
        f"collision={'yes' if simulation.collision_detected else 'no'}.")
    return output_path


def generate_firi_images(
        output_dir: str | Path,
        *,
        width: int = 1920,
        height: int = 1080,
) -> list[Path]:
    """Render several camera views of the FIRI corridor as lossless PNGs."""
    simulation = MujocoSimulation(DEFAULT_SCENE_PATH)
    corridor = build_mission_corridor(
        simulation, simulation.mission_waypoints, visualize=True)
    output_dir = Path(output_dir)
    views = {
        "firi_corridor_overview.png": (45.0, -32.0),
        "firi_corridor_top.png": (0.0, -78.0),
        "firi_corridor_side.png": (90.0, -12.0),
    }
    outputs = []
    for filename, (azimuth, elevation) in views.items():
        camera = default_camera(simulation.model)
        camera.azimuth = azimuth
        camera.elevation = elevation
        frame = render_offscreen_frame(
            simulation.model, simulation.data, width, height, camera)
        output_path = output_dir / filename
        save_png(frame, output_path)
        outputs.append(output_path)
    print(f"FIRI: {len(corridor.regions)} convex regions rendered to {output_dir}.")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=[*EXPERIMENTS, "all"], default="all",
        help="experiment to record (default: all)",
    )
    parser.add_argument(
        "--firi-images", action="store_true",
        help="also render overview/top/side PNGs of the FIRI corridor",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/videos"),
        help="directory for MP4 files (default: docs/videos)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--hold-seconds", type=float, default=2.0,
        help="seconds to keep applying terminal feedback after the trajectory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiments = EXPERIMENTS if args.experiment == "all" else {args.experiment: None}
    for experiment in experiments:
        filename = f"{experiment.replace('-', '_')}.mp4"
        record_experiment(
            experiment, args.output_dir / filename, fps=args.fps,
            width=args.width, height=args.height, hold_seconds=args.hold_seconds,
        )
    if args.firi_images:
        generate_firi_images(
            args.output_dir.parent, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
