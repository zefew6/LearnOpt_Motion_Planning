"""Persistent multi-trajectory assets for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uav_ac.control.trajectory_controller import TrajectoryController
from uav_ac.planning.pipeline import build_mission_corridor, generate_gcopter_mission
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH, MujocoSimulation

from .assets import InitializationLibrary


TRAJECTORY_BANK_DIRECTORY = "trajectory_bank"
TRAJECTORY_BANK_SCHEMA_VERSION = 2
_TRAJECTORIES_FILE = "trajectories.npy"
_STATES_FILE = "initialization_states.npy"
_MOTORS_FILE = "motor_speeds.npy"
_OFFSETS_FILE = "offsets.npy"
_WAYPOINTS_FILE = "waypoints.npy"
_WAYPOINT_OFFSETS_FILE = "waypoint_offsets.npy"
_METADATA_FILE = "metadata.json"


class TrajectoryBankGenerationError(RuntimeError):
    """Raised when strict dataset quotas cannot be satisfied."""


@dataclass(frozen=True)
class TrajectoryBank:
    """Variable-length trajectories and aligned MPC initialization snapshots."""

    trajectories: np.ndarray
    states: np.ndarray
    motor_speeds: np.ndarray
    offsets: np.ndarray
    waypoints: np.ndarray
    waypoint_offsets: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        trajectories = np.asarray(self.trajectories)
        states = np.asarray(self.states)
        motors = np.asarray(self.motor_speeds)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        waypoints = np.asarray(self.waypoints)
        waypoint_offsets = np.asarray(self.waypoint_offsets, dtype=np.int64)
        entries = self.metadata.get("entries", [])
        if (trajectories.ndim != 2 or trajectories.shape[1] < 10
                or not np.all(np.isfinite(trajectories))):
            raise ValueError("bank trajectories must have finite shape (n, m), m >= 10")
        if states.shape != (len(trajectories), 13) or not np.all(np.isfinite(states)):
            raise ValueError("bank states must have shape (n, 13) and be finite")
        if (motors.shape != (len(trajectories), 4)
                or not np.all(np.isfinite(motors)) or np.any(motors < 0.0)):
            raise ValueError("bank motor speeds must have finite non-negative shape (n, 4)")
        _validate_offsets(offsets, len(trajectories), len(entries), "trajectory")
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or not np.all(np.isfinite(waypoints)):
            raise ValueError("bank waypoints must have finite shape (n, 3)")
        _validate_offsets(waypoint_offsets, len(waypoints), len(entries), "waypoint")
        if int(self.metadata.get("schema_version", -1)) != TRAJECTORY_BANK_SCHEMA_VERSION:
            raise ValueError("unsupported trajectory-bank schema")
        identifiers = [int(entry.get("trajectory_id", -1)) for entry in entries]
        if identifiers != list(range(len(entries))):
            raise ValueError("trajectory metadata identifiers must be consecutive from zero")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "waypoint_offsets", waypoint_offsets)

    def __len__(self) -> int:
        return len(self.metadata["entries"])

    def trajectory(self, trajectory_id: int) -> np.ndarray:
        start, stop = self._slice(self.offsets, trajectory_id)
        return self.trajectories[start:stop]

    def initialization_library(self, trajectory_id: int) -> InitializationLibrary:
        start, stop = self._slice(self.offsets, trajectory_id)
        return InitializationLibrary(self.states[start:stop], self.motor_speeds[start:stop])

    def trajectory_waypoints(self, trajectory_id: int) -> np.ndarray:
        start, stop = self._slice(self.waypoint_offsets, trajectory_id)
        return self.waypoints[start:stop]

    def entry(self, trajectory_id: int) -> dict[str, Any]:
        self._slice(self.offsets, trajectory_id)
        return self.metadata["entries"][trajectory_id]

    def indices(self, split: str) -> np.ndarray:
        indices = [
            index for index, entry in enumerate(self.metadata["entries"])
            if entry["split"] == split
        ]
        if not indices:
            raise ValueError(f"trajectory bank contains no '{split}' trajectories")
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _slice(offsets: np.ndarray, trajectory_id: int) -> tuple[int, int]:
        trajectory_id = int(trajectory_id)
        if not 0 <= trajectory_id < len(offsets) - 1:
            raise ValueError("trajectory_id lies outside the trajectory bank")
        return int(offsets[trajectory_id]), int(offsets[trajectory_id + 1])

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _TRAJECTORIES_FILE, self.trajectories)
        np.save(directory / _STATES_FILE, self.states)
        np.save(directory / _MOTORS_FILE, self.motor_speeds)
        np.save(directory / _OFFSETS_FILE, self.offsets)
        np.save(directory / _WAYPOINTS_FILE, self.waypoints)
        np.save(directory / _WAYPOINT_OFFSETS_FILE, self.waypoint_offsets)
        with (directory / _METADATA_FILE).open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, mmap_mode: str | None = "r") -> "TrajectoryBank":
        directory = Path(directory)
        with (directory / _METADATA_FILE).open(encoding="utf-8") as file:
            metadata = json.load(file)
        return cls(
            trajectories=np.load(directory / _TRAJECTORIES_FILE, mmap_mode=mmap_mode),
            states=np.load(directory / _STATES_FILE, mmap_mode=mmap_mode),
            motor_speeds=np.load(directory / _MOTORS_FILE, mmap_mode=mmap_mode),
            offsets=np.load(directory / _OFFSETS_FILE, mmap_mode=mmap_mode),
            waypoints=np.load(directory / _WAYPOINTS_FILE, mmap_mode=mmap_mode),
            waypoint_offsets=np.load(
                directory / _WAYPOINT_OFFSETS_FILE, mmap_mode=mmap_mode),
            metadata=metadata,
        )

    @classmethod
    def from_sequences(
            cls,
            trajectories: Sequence[np.ndarray],
            libraries: Sequence[InitializationLibrary],
            entries: Sequence[dict[str, Any]],
            *,
            waypoints: Sequence[np.ndarray] | None = None,
            fingerprint: str = "test",
    ) -> "TrajectoryBank":
        if not trajectories or len(trajectories) != len(libraries) or len(entries) != len(trajectories):
            raise ValueError("trajectories, libraries, and entries must have equal non-zero length")
        trajectories = [np.asarray(value, dtype=float) for value in trajectories]
        if any(len(trajectory) != len(library.states)
               for trajectory, library in zip(trajectories, libraries)):
            raise ValueError("each initialization library must align with its trajectory")
        if waypoints is None:
            waypoints = [trajectory[[0, -1], :3] for trajectory in trajectories]
        if len(waypoints) != len(trajectories):
            raise ValueError("waypoints must align with trajectories")
        normalized_entries = []
        for index, entry in enumerate(entries):
            normalized = dict(entry)
            normalized["trajectory_id"] = index
            normalized.setdefault("split", "train")
            normalized_entries.append(normalized)
        return cls(
            trajectories=np.concatenate(trajectories),
            states=np.concatenate([library.states for library in libraries]),
            motor_speeds=np.concatenate([library.motor_speeds for library in libraries]),
            offsets=_length_offsets([len(value) for value in trajectories]),
            waypoints=np.concatenate([np.asarray(value, dtype=float) for value in waypoints]),
            waypoint_offsets=_length_offsets([len(value) for value in waypoints]),
            metadata={
                "schema_version": TRAJECTORY_BANK_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "entries": normalized_entries,
            },
        )


def trajectory_bank_fingerprint(
        generation_config: dict[str, Any],
        *,
        scene_path: str | Path = OPEN_FIELD_SCENE_PATH,
        steps_per_action: int,
) -> str:
    """Hash every input that makes the generated bank reusable."""
    scene_path = Path(scene_path)
    payload = {
        "schema_version": TRAJECTORY_BANK_SCHEMA_VERSION,
        "generation": generation_config,
        "steps_per_action": int(steps_per_action),
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_open_field_waypoints(
        start_position: np.ndarray,
        navigation_waypoint_count: int,
        rng: np.random.Generator,
        config: dict[str, Any],
        *,
        pattern: str,
) -> np.ndarray:
    """Sample a long, bounded, maneuver-rich open-field mission in NED."""
    if navigation_waypoint_count < 1:
        raise ValueError("navigation_waypoint_count must be positive")
    lower_xy, upper_xy = np.asarray(config["horizontal_bounds"], dtype=float)
    altitude_min, altitude_max = map(float, config["altitude"])
    segment_min, segment_max = map(float, config["segment_length"])
    path_min, path_max = map(float, config["path_length"])
    takeoff_altitude = float(config["takeoff_altitude"])
    patterns = {"s_curve", "arc", "zigzag", "random_turns", "climb_dive", "sweep"}
    if pattern not in patterns:
        raise ValueError(f"unknown waypoint pattern: {pattern}")

    if pattern == "sweep":
        return _sample_sweep_waypoints(
            start_position,
            navigation_waypoint_count,
            rng,
            lower_xy,
            upper_xy,
            altitude_min,
            altitude_max,
            takeoff_altitude,
        )

    start_position = np.asarray(start_position, dtype=float)
    for _ in range(256):
        points = [start_position.copy()]
        current = np.array([start_position[0], start_position[1], -takeoff_altitude])
        points.append(current.copy())
        heading = rng.uniform(-np.pi, np.pi)
        arc_sign = rng.choice((-1.0, 1.0))
        for index in range(navigation_waypoint_count):
            accepted = False
            for retry in range(64):
                if pattern == "s_curve":
                    turn = ((-1.0) ** index) * rng.uniform(0.45, 1.05)
                elif pattern == "arc":
                    turn = arc_sign * rng.uniform(0.30, 0.75)
                elif pattern == "zigzag":
                    turn = ((-1.0) ** index) * rng.uniform(0.8, 1.7)
                elif pattern == "climb_dive":
                    turn = rng.uniform(-1.0, 1.0)
                else:
                    turn = rng.uniform(-2.2, 2.2)
                candidate_heading = heading + turn
                if retry >= 16:
                    # Near an edge, turn back through the interior instead of
                    # repeatedly proposing the same outward-curving maneuver.
                    candidate_heading = np.arctan2(-current[1], -current[0]) + rng.uniform(-0.8, 0.8)
                length = rng.uniform(segment_min, segment_max)
                candidate_xy = current[:2] + length * np.array([
                    np.cos(candidate_heading), np.sin(candidate_heading)])
                if np.any(candidate_xy < lower_xy) or np.any(candidate_xy > upper_xy):
                    continue
                if pattern == "climb_dive":
                    fraction = index / max(navigation_waypoint_count - 1, 1)
                    altitude_center = (
                        altitude_min + (altitude_max - altitude_min)
                        * (1.0 - abs(2.0 * fraction - 1.0))
                    )
                    altitude = np.clip(
                        altitude_center + rng.uniform(-0.35, 0.35),
                        altitude_min,
                        altitude_max,
                    )
                else:
                    altitude = np.clip(
                        -current[2] + rng.uniform(-1.3, 1.3),
                        altitude_min,
                        altitude_max,
                    )
                current = np.array([candidate_xy[0], candidate_xy[1], -altitude])
                points.append(current.copy())
                heading = candidate_heading
                accepted = True
                break
            if not accepted:
                break
        if len(points) != navigation_waypoint_count + 2:
            continue
        points_array = np.asarray(points)
        length = float(np.sum(np.linalg.norm(np.diff(points_array, axis=0), axis=1)))
        if not path_min <= length <= path_max:
            continue
        course_vectors = np.diff(points_array[1:, :2], axis=0)
        if len(course_vectors) >= 2:
            headings = np.unwrap(np.arctan2(course_vectors[:, 1], course_vectors[:, 0]))
            meaningful_turns = np.count_nonzero(np.abs(np.diff(headings)) >= np.deg2rad(25.0))
            if meaningful_turns < min(2, len(course_vectors) - 1):
                continue
        return points_array
    raise TrajectoryBankGenerationError(
        f"could not sample a valid {pattern} mission after 256 attempts")


def _sample_sweep_waypoints(
        start_position: np.ndarray,
        count: int,
        rng: np.random.Generator,
        lower_xy: np.ndarray,
        upper_xy: np.ndarray,
        altitude_min: float,
        altitude_max: float,
        takeoff_altitude: float,
) -> np.ndarray:
    """Sample a long, gently curving sweep used by the highest speed stratum."""
    distance = rng.uniform(0.92, 0.98) * (upper_xy[0] - start_position[0])
    progress = np.linspace(1.0 / count, 1.0, count)
    lateral_amplitude = rng.uniform(3.0, 8.0)
    sampled_xy = np.column_stack((
        start_position[0] + progress * distance,
        start_position[1]
        + rng.choice((-1.0, 1.0)) * lateral_amplitude * np.sin(2.0 * np.pi * progress),
    ))
    phase = rng.uniform(-np.pi, np.pi)
    fractions = np.linspace(0.0, 1.0, count)
    altitude_center = rng.uniform(
        altitude_min + 0.25 * (altitude_max - altitude_min),
        altitude_max - 0.25 * (altitude_max - altitude_min),
    )
    altitudes = np.clip(
        altitude_center + 0.45 * np.sin(2.0 * np.pi * fractions + phase),
        altitude_min,
        altitude_max,
    )
    return np.vstack((
        np.asarray(start_position, dtype=float),
        [start_position[0], start_position[1], -takeoff_altitude],
        np.column_stack((sampled_xy, -altitudes)),
    ))


def generate_trajectory_bank(
        directory: str | Path,
        generation_config: dict[str, Any],
        *,
        steps_per_action: int,
        seed: int,
        scene_path: str | Path = OPEN_FIELD_SCENE_PATH,
        progress: Callable[[str], None] = print,
) -> TrajectoryBank:
    """Generate, MPC-validate, and persist the configured trajectory splits."""
    directory = Path(directory)
    fingerprint = trajectory_bank_fingerprint(
        generation_config, scene_path=scene_path, steps_per_action=steps_per_action)
    if (directory / _METADATA_FILE).is_file():
        bank = TrajectoryBank.load(directory)
        if bank.metadata.get("fingerprint") != fingerprint:
            raise ValueError(
                "cached trajectory bank does not match the current configuration; "
                "use a new run directory")
        return bank

    simulation = MujocoSimulation(scene_path, record_actual_trajectory=False)
    if not np.isclose(simulation.quad.dt, 0.001):
        raise ValueError("RL training requires a 1 kHz MuJoCo model timestep")
    try:
        from uav_ac.control.mpc_controller import MPCConfig, MPCController
        controller = MPCController(
            simulation.quad,
            MPCConfig(
                dt=simulation.quad.dt * steps_per_action,
                horizon_steps=int(generation_config["mpc_validation"]["horizon_steps"]),
                nlp_solver_type="SQP_RTI",
            ),
        )
    except ImportError as error:
        raise ImportError(
            "trajectory-bank generation requires acados_template and its Python "
            "dependencies; install the project 'mpc' extra and the local acados interface"
        ) from error

    rng = np.random.default_rng(seed)
    patterns = ("s_curve", "arc", "zigzag", "random_turns", "climb_dive")
    waypoint_low, waypoint_high = map(int, generation_config["navigation_waypoints"])
    speed_edges = np.asarray(generation_config["average_speed_bins"], dtype=float)
    split_targets: list[tuple[str, int, int, str]] = []
    for split, count in generation_config["splits"].items():
        split_entries = []
        for index in range(int(count)):
            speed_bin = index % (len(speed_edges) - 1)
            pattern = patterns[
                (index // (len(speed_edges) - 1) + speed_bin) % len(patterns)
            ]
            if speed_bin >= len(speed_edges) - 3:
                pattern = "sweep"
            split_entries.append((
                split,
                waypoint_low + (
                    index // (len(speed_edges) - 1) + speed_bin
                )
                % (waypoint_high - waypoint_low + 1),
                speed_bin,
                pattern,
            ))
        rng.shuffle(split_entries)
        split_targets.extend(split_entries)

    trajectories: list[np.ndarray] = []
    libraries: list[InitializationLibrary] = []
    entries: list[dict[str, Any]] = []
    waypoints: list[np.ndarray] = []
    rejected: dict[str, int] = {}
    max_attempts = int(generation_config["max_attempts"])
    attempts = 0
    for split, waypoint_count, speed_bin, pattern in split_targets:
        accepted = False
        while attempts < max_attempts and not accepted:
            attempts += 1
            candidate_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            candidate_rng = np.random.default_rng(candidate_seed)
            try:
                candidate_waypoints = sample_open_field_waypoints(
                    simulation.start_position,
                    waypoint_count,
                    candidate_rng,
                    generation_config,
                    pattern=pattern,
                )
                bin_low, bin_high = speed_edges[speed_bin:speed_bin + 2]
                maximum_speed = float(generation_config["maximum_speed"])
                is_last_speed_bin = speed_bin == len(speed_edges) - 2
                planner_speed = (
                    maximum_speed if is_last_speed_bin
                    else float(candidate_rng.uniform(3.0, 4.5))
                )
                takeoff_waypoints = candidate_waypoints[:2]
                course_waypoints = candidate_waypoints[1:]
                takeoff = generate_gcopter_mission(
                    takeoff_waypoints,
                    build_mission_corridor(
                        simulation, takeoff_waypoints, visualize=False),
                    simulation.quad,
                    float(generation_config["takeoff_max_speed"]),
                    simulation.quad.dt * steps_per_action,
                    length_per_piece=float(generation_config["length_per_piece"]),
                    max_acceleration=float(
                        generation_config["takeoff_max_acceleration"]),
                    verbose=False,
                )
                course = generate_gcopter_mission(
                    course_waypoints,
                    build_mission_corridor(
                        simulation, course_waypoints, visualize=False),
                    simulation.quad,
                    planner_speed,
                    simulation.quad.dt * steps_per_action,
                    length_per_piece=float(generation_config["length_per_piece"]),
                    max_acceleration=float(
                        generation_config["gcopter_max_acceleration"]),
                    verbose=False,
                )
                gcopter_trajectory = np.vstack((takeoff, course[1:]))
                original_duration = (
                    (len(gcopter_trajectory) - 1)
                    * simulation.quad.dt * steps_per_action
                )
                if is_last_speed_bin:
                    target_average_speed = bin_low
                    trajectory = gcopter_trajectory
                else:
                    target_upper = bin_high - 0.05
                    if speed_bin == len(speed_edges) - 3:
                        target_upper = min(target_upper, bin_low + 0.30)
                    target_average_speed = float(candidate_rng.uniform(
                        bin_low + 0.05, target_upper))
                    dt = simulation.quad.dt * steps_per_action
                    takeoff_duration = (len(takeoff) - 1) * dt
                    course_path_length = float(np.sum(np.linalg.norm(
                        np.diff(course[:, :3], axis=0), axis=1)))
                    complete_path_length = float(np.sum(np.linalg.norm(
                        np.diff(gcopter_trajectory[:, :3], axis=0), axis=1)))
                    requested_course_duration = (
                        complete_path_length / target_average_speed - takeoff_duration)
                    if requested_course_duration <= 0.0:
                        raise _CandidateRejected("retiming_duration")
                    retimed_course = _retime_trajectory(
                        course,
                        dt,
                        course_path_length / requested_course_duration,
                    )
                    trajectory = np.vstack((takeoff, retimed_course[1:]))
                duration = (len(trajectory) - 1) * simulation.quad.dt * steps_per_action
                path_length = float(np.sum(np.linalg.norm(
                    np.diff(trajectory[:, :3], axis=0), axis=1)))
                average_speed = path_length / max(duration, np.finfo(float).eps)
                peak_speed = float(np.max(np.linalg.norm(trajectory[:, 3:6], axis=1)))
                in_speed_bin = (
                    bin_low <= average_speed <= bin_high
                    if is_last_speed_bin else bin_low <= average_speed < bin_high
                )
                if not in_speed_bin or peak_speed > maximum_speed + 1.0e-6:
                    raise _CandidateRejected("speed_bin")
                _validate_reference_feasibility(trajectory, simulation.quad)
                library, metrics = _mpc_initialization_rollout(
                    simulation,
                    controller,
                    trajectory,
                    steps_per_action,
                    generation_config["mpc_validation"],
                )
            except _CandidateRejected as error:
                rejected[error.reason] = rejected.get(error.reason, 0) + 1
                continue
            except (RuntimeError, ValueError) as error:
                reason = f"planning:{type(error).__name__}"
                rejected[reason] = rejected.get(reason, 0) + 1
                continue

            trajectory_id = len(entries)
            trajectories.append(trajectory)
            libraries.append(library)
            waypoints.append(candidate_waypoints)
            entries.append({
                "trajectory_id": trajectory_id,
                "split": split,
                "seed": candidate_seed,
                "pattern": pattern,
                "navigation_waypoint_count": waypoint_count,
                "planner_maximum_speed": planner_speed,
                "gcopter_original_duration": original_duration,
                "target_average_speed": target_average_speed,
                "average_speed": average_speed,
                "peak_speed": peak_speed,
                "duration": duration,
                "path_length": path_length,
                **metrics,
            })
            accepted = True
            progress(
                f"Accepted {trajectory_id + 1}/{len(split_targets)}: {split}, "
                f"{pattern}, {waypoint_count} waypoints, {average_speed:.2f} m/s mean")
        if not accepted:
            raise TrajectoryBankGenerationError(
                f"generated {len(entries)}/{len(split_targets)} trajectories after "
                f"{attempts} attempts; rejected={json.dumps(rejected, sort_keys=True)}")

    bank = TrajectoryBank.from_sequences(
        trajectories,
        libraries,
        entries,
        waypoints=waypoints,
        fingerprint=fingerprint,
    )
    bank.metadata["generation"] = generation_config
    bank.metadata["rejected_candidates"] = rejected
    bank.metadata["attempts"] = attempts
    bank.save(directory)
    return bank


class _CandidateRejected(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _mpc_initialization_rollout(
        simulation: MujocoSimulation,
        controller: Any,
        trajectory: np.ndarray,
        steps_per_action: int,
        thresholds: dict[str, Any],
) -> tuple[InitializationLibrary, dict[str, float]]:
    simulation.reset()
    controller.reset()
    tracker = TrajectoryController(
        controller, simulation.quad, trajectory, steps_per_action)
    states = []
    motor_speeds = []
    errors = []
    maximum_tilt = 0.0

    for reference_index in range(len(trajectory)):
        states.append(simulation.quad.X.copy())
        motor_speeds.append(simulation.quad.omega.copy())
        errors.append(float(np.linalg.norm(
            simulation.quad.position - trajectory[reference_index, :3])))
        for _ in range(steps_per_action):
            tracker.step()
            if controller.last_status not in (None, 0):
                raise _CandidateRejected("mpc_solver")
            simulation.step()
            maximum_tilt = max(maximum_tilt, _tilt(simulation.quad.R()))
            _validate_rollout_state(simulation)

    hold_steps = int(round(float(thresholds["terminal_hold_seconds"]) / simulation.quad.dt))
    for _ in range(hold_steps):
        tracker.step()
        if controller.last_status not in (None, 0):
            raise _CandidateRejected("mpc_solver")
        simulation.step()
        maximum_tilt = max(maximum_tilt, _tilt(simulation.quad.R()))
        _validate_rollout_state(simulation)

    position_rmse = float(np.sqrt(np.mean(np.square(errors))))
    maximum_position_error = float(np.max(errors))
    final_position_error = float(np.linalg.norm(
        simulation.quad.position - trajectory[-1, :3]))
    final_velocity_error = float(np.linalg.norm(
        simulation.quad.velocity - trajectory[-1, 3:6]))
    checks = (
        (position_rmse <= float(thresholds["position_rmse_max"]), "mpc_position_rmse"),
        (maximum_position_error <= float(thresholds["position_error_max"]), "mpc_position_error"),
        (final_position_error < float(thresholds["final_position_error_max"]), "mpc_final_position"),
        (final_velocity_error < float(thresholds["final_velocity_error_max"]), "mpc_final_velocity"),
        (maximum_tilt < np.deg2rad(float(thresholds["maximum_tilt_degrees"])), "mpc_tilt"),
    )
    for passed, reason in checks:
        if not passed:
            raise _CandidateRejected(reason)
    return InitializationLibrary(np.asarray(states), np.asarray(motor_speeds)), {
        "mpc_position_rmse": position_rmse,
        "mpc_maximum_position_error": maximum_position_error,
        "mpc_final_position_error": final_position_error,
        "mpc_final_velocity_error": final_velocity_error,
        "mpc_maximum_tilt_degrees": float(np.rad2deg(maximum_tilt)),
    }


def _validate_rollout_state(simulation: MujocoSimulation) -> None:
    if not np.all(np.isfinite(simulation.quad.X)):
        raise _CandidateRejected("non_finite_state")
    if simulation.collision_detected:
        raise _CandidateRejected("collision")
    if (np.any(simulation.quad.position < simulation.space_limits[0])
            or np.any(simulation.quad.position > simulation.space_limits[1])):
        raise _CandidateRejected("out_of_bounds")


def _validate_reference_feasibility(trajectory: np.ndarray, quad: Any) -> None:
    force = np.column_stack((
        -trajectory[:, 6],
        -trajectory[:, 7],
        quad.g - trajectory[:, 8],
    ))
    force_norm = np.linalg.norm(force, axis=1)
    tilt = np.arccos(np.clip(
        force[:, 2] / np.maximum(force_norm, np.finfo(float).eps), -1.0, 1.0))
    thrust = quad.m * force_norm
    if np.max(tilt) > quad.max_tilt_angle + 1.0e-3:
        raise _CandidateRejected("reference_tilt")
    if (np.min(thrust) < 4.0 * quad.min_thrust - 1.0e-6
            or np.max(thrust) > 4.0 * quad.max_thrust + 1.0e-6):
        raise _CandidateRejected("reference_thrust")


def _tilt(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))


def _retime_trajectory(
        trajectory: np.ndarray,
        dt: float,
        target_average_speed: float,
) -> np.ndarray:
    """Uniformly retime a GCOPTER path while preserving its derivative fields."""
    path_length = float(np.sum(np.linalg.norm(
        np.diff(trajectory[:, :3], axis=0), axis=1)))
    source_duration = (len(trajectory) - 1) * dt
    requested_duration = path_length / target_average_speed
    intervals = max(1, int(np.ceil(requested_duration / dt)))
    duration = intervals * dt
    time_scale = source_duration / duration
    source_times = np.arange(len(trajectory)) * dt
    query_times = np.minimum(np.arange(intervals + 1) * dt * time_scale, source_duration)
    retimed = np.empty((intervals + 1, trajectory.shape[1]), dtype=float)
    for column in range(3):
        retimed[:, column] = np.interp(query_times, source_times, trajectory[:, column])
        retimed[:, 3 + column] = time_scale * np.interp(
            query_times, source_times, trajectory[:, 3 + column])
        retimed[:, 6 + column] = time_scale**2 * np.interp(
            query_times, source_times, trajectory[:, 6 + column])
    yaw = np.unwrap(trajectory[:, 9])
    retimed[:, 9] = (np.interp(query_times, source_times, yaw) + np.pi) % (2.0 * np.pi) - np.pi
    if trajectory.shape[1] > 10:
        for column in range(10, trajectory.shape[1]):
            retimed[:, column] = np.interp(
                query_times, source_times, trajectory[:, column])
    retimed[0, :3] = trajectory[0, :3]
    retimed[-1, :3] = trajectory[-1, :3]
    return retimed


def _length_offsets(lengths: Sequence[int]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))


def _validate_offsets(
        offsets: np.ndarray, total: int, entry_count: int, label: str,
) -> None:
    if (offsets.shape != (entry_count + 1,) or offsets[0] != 0
            or offsets[-1] != total or np.any(np.diff(offsets) <= 0)):
        raise ValueError(f"invalid {label} offsets")


__all__ = [
    "TRAJECTORY_BANK_DIRECTORY",
    "TRAJECTORY_BANK_SCHEMA_VERSION",
    "TrajectoryBank",
    "TrajectoryBankGenerationError",
    "generate_trajectory_bank",
    "sample_open_field_waypoints",
    "trajectory_bank_fingerprint",
]
