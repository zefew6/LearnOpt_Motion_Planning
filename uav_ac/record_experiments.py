"""Generate deterministic, offscreen comparison videos and FIRI images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from uav_ac import utils
from uav_ac.main import _build_controller, _plan_trajectory
from uav_ac.control import TrajectoryController
from uav_ac.planning.pipeline import build_mission_corridor
from uav_ac.simulation.recording import default_camera, render_offscreen_frame, save_png
from uav_ac.simulation.mujoco_sim import (
    DEFAULT_SCENE_PATH,
    GCS_BUILDING_SCENE_PATH,
    MujocoSimulation,
)


ExperimentName = Literal[
    "gcopter-cascaded",
    "gcopter-mpc",
    "gcopter-rl",
    "minisnap-cascaded",
    "minisnap-mpc",
    "gcs-mpc",
]
CameraMode = Literal["fixed", "follow"]

EXPERIMENTS: dict[ExperimentName, tuple[str, str]] = {
    "gcopter-cascaded": ("gcopter", "cascaded"),
    "gcopter-mpc": ("gcopter", "mpc"),
    "gcopter-rl": ("gcopter", "rl"),
    "minisnap-cascaded": ("mini_snap", "cascaded"),
    "minisnap-mpc": ("mini_snap", "mpc"),
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


def _follow_drone_camera(simulation: MujocoSimulation):
    def update_camera(camera) -> None:
        body_position = simulation.data.xpos[simulation._body_id]
        body_rotation = simulation.data.xmat[simulation._body_id].reshape(3, 3)
        body_forward = body_rotation[:, 0]
        camera.lookat[:] = body_position + np.array([0.0, 0.0, 0.5])
        camera.azimuth = np.rad2deg(np.arctan2(body_forward[1], body_forward[0]))

    return update_camera


def _trajectory_path_length(trajectory: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1).sum())


def _trajectory_yaws(velocities: np.ndarray) -> np.ndarray:
    horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
    moving = np.flatnonzero(horizontal_speeds >= 1.0e-3)
    if len(moving) == 0:
        return np.zeros(len(velocities))
    moving_yaws = np.unwrap(np.arctan2(velocities[moving, 1], velocities[moving, 0]))
    previous = np.searchsorted(moving, np.arange(len(velocities)), side="right") - 1
    previous = np.clip(previous, 0, len(moving) - 1)
    return moving_yaws[previous]


def _retime_trajectory_average_speed(
        trajectory: np.ndarray,
        dt: float,
        average_speed: float,
) -> np.ndarray:
    if average_speed <= 0.0:
        raise ValueError("average_speed must be positive")
    arc_length = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1))))
    total_length = float(arc_length[-1])
    if total_length <= 1.0e-10:
        raise ValueError("trajectory must have non-zero path length")
    keep = np.concatenate(([True], np.diff(arc_length) > 1.0e-10))
    arc_length = arc_length[keep]
    positions = trajectory[keep, :3]
    duration = total_length / average_speed
    intervals = max(1, int(round(duration / dt)))
    times = np.arange(intervals + 1) * dt
    normalized_time = np.clip(times / (intervals * dt), 0.0, 1.0)
    eased_distance = total_length * 0.5 * (1.0 - np.cos(np.pi * normalized_time))
    position_samples = np.column_stack([
        np.interp(eased_distance, arc_length, positions[:, axis])
        for axis in range(3)
    ])
    edge_order = 2 if len(position_samples) >= 3 else 1
    velocity_samples = np.gradient(position_samples, dt, axis=0, edge_order=edge_order)
    acceleration_samples = np.gradient(velocity_samples, dt, axis=0, edge_order=edge_order)
    velocity_samples[[0, -1]] = 0.0
    acceleration_samples[[0, -1]] = 0.0
    yaws = _trajectory_yaws(velocity_samples)
    return np.hstack((
        position_samples, velocity_samples, acceleration_samples, yaws[:, None]))


def record_experiment(
        experiment: ExperimentName,
        output_path: str | Path,
        *,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        hold_seconds: float = 2.0,
        camera_mode: CameraMode = "fixed",
        rl_run_dir: str | Path | None = None,
        gcopter_velocity: float = 3.0,
        gcopter_average_speed: float | None = None,
        minisnap_velocity: float = 4.0,
) -> Path:
    """Plan and record one experiment without starting a MuJoCo viewer."""
    if hold_seconds < 0.0:
        raise ValueError("hold_seconds must be non-negative")

    planner, controller_name = EXPERIMENTS[experiment]
    cfg, flight_cfg = utils.get_config()
    steps_per_reference = cfg.getint("frequency")
    velocity = (
        gcopter_velocity if planner == "gcopter"
        else minisnap_velocity if planner == "mini_snap"
        else flight_cfg.getfloat("velocity")
    )
    scene_path = GCS_BUILDING_SCENE_PATH if planner == "gcs" else DEFAULT_SCENE_PATH

    simulation = MujocoSimulation(scene_path)
    trajectory_dt = simulation.quad.dt * steps_per_reference
    trajectory = _plan_trajectory(
        planner, simulation, velocity, trajectory_dt, visualize=False,
    )
    if planner == "gcopter" and gcopter_average_speed is not None:
        trajectory = _retime_trajectory_average_speed(
            trajectory, trajectory_dt, gcopter_average_speed)
        path_length = _trajectory_path_length(trajectory)
        duration = (len(trajectory) - 1) * trajectory_dt
        print(
            f"Retimed GCOPTER: {duration:.2f} s, "
            f"{path_length / duration:.2f} m/s average speed.")
    # Keep the takeoff segment visible so the planned path starts at the
    # initialized vehicle position in recordings as well.
    simulation.set_trajectory_visualization(trajectory[:, :3])

    tracker = TrajectoryController(
        controller=_build_controller(
            controller_name, simulation.quad, trajectory_dt, run_dir=rl_run_dir),
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
    update_camera = None
    if camera_mode == "follow":
        camera.distance = 8.0
        camera.elevation = -20.0
        update_camera = _follow_drone_camera(simulation)
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
        update_camera=update_camera,
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
    parser.add_argument(
        "--camera", choices=("fixed", "follow"), default="fixed",
        help="recording camera mode: fixed scene view or drone-following view",
    )
    parser.add_argument(
        "--rl-run-dir", type=Path, default=Path("runs/ppo_trajectory/multitraj01"),
        help="trained PPO run directory used by RL controller experiments",
    )
    parser.add_argument(
        "--gcopter-velocity", type=float, default=3.0,
        help="maximum speed constraint used when planning GCOPTER trajectories",
    )
    parser.add_argument(
        "--gcopter-average-speed", type=float,
        help="retime GCOPTER recordings to this average path speed",
    )
    parser.add_argument(
        "--minisnap-velocity", type=float, default=4.0,
        help="speed used when planning MinimumSnap trajectories",
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
            camera_mode=args.camera, rl_run_dir=args.rl_run_dir,
            gcopter_velocity=args.gcopter_velocity,
            gcopter_average_speed=args.gcopter_average_speed,
            minisnap_velocity=args.minisnap_velocity,
        )
    if args.firi_images:
        generate_firi_images(
            args.output_dir.parent, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
