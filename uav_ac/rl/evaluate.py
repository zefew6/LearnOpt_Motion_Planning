"""Evaluate, view, or record a trained PPO trajectory controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from stable_baselines3 import PPO

from uav_ac.control import RLController, TrajectoryController
from uav_ac.control.rl_controller import RL_CONFIG_FILENAME
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation

from .assets import InitializationLibrary
from .environment import MujocoTrajectoryTrackingEnv


EvaluationMode = Literal["metrics", "interactive", "record"]


def _load_assets(run_dir: Path) -> tuple[dict[str, Any], np.ndarray, InitializationLibrary]:
    with (run_dir / RL_CONFIG_FILENAME).open(encoding="utf-8") as file:
        config = json.load(file)
    trajectory = np.load(run_dir / config["trajectory_file"])
    library = InitializationLibrary.load(
        run_dir / config["initialization_library_file"])
    return config, trajectory, library


def evaluate_metrics(
        run_dir: str | Path,
        *,
        seeds: tuple[int, ...] = tuple(range(20)),
        perturb_initial_state: bool = True,
        device: str = "cpu",
) -> dict[str, Any]:
    """Run deterministic full-course episodes and persist aggregate metrics."""
    run_dir = Path(run_dir)
    config, trajectory, library = _load_assets(run_dir)
    environment = MujocoTrajectoryTrackingEnv(
        trajectory,
        library,
        steps_per_action=int(config["steps_per_action"]),
        random_start=False,
        perturb_initial_state=perturb_initial_state,
        curriculum_progress=1.0,
    )
    model = PPO.load(run_dir / "best_model.zip", device=device)
    episodes = []
    maximum_steps = len(trajectory) + int(round(1.0 / environment.control_dt)) + 1
    try:
        for seed in seeds:
            observation, _ = environment.reset(
                seed=seed,
                options={
                    "start_index": 0,
                    "perturbation_scale": 1.0 if perturb_initial_state else 0.0,
                },
            )
            position_errors = []
            final_info: dict[str, Any] = {}
            for step_index in range(maximum_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, final_info = environment.step(action)
                position_errors.append(final_info["position_error"])
                if terminated or truncated:
                    break
            episodes.append({
                "seed": int(seed),
                "success": bool(final_info.get("success", False)),
                "collision": bool(final_info.get("collision", False)),
                "failure_reason": final_info.get("failure_reason"),
                "steps": step_index + 1,
                "position_rmse": float(np.sqrt(np.mean(np.square(position_errors)))),
                "final_position_error": float(final_info.get("position_error", np.inf)),
            })
    finally:
        environment.close()

    successful_rmses = [episode["position_rmse"] for episode in episodes if episode["success"]]
    results = {
        "episode_count": len(episodes),
        "success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "successful_mean_position_rmse": (
            float(np.mean(successful_rmses)) if successful_rmses else None
        ),
        "perturbed_initial_state": bool(perturb_initial_state),
        "episodes": episodes,
    }
    with (run_dir / "evaluation_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")
    return results


def _deployment_tracker(
        run_dir: Path,
        device: str,
) -> tuple[MujocoSimulation, TrajectoryController, np.ndarray]:
    config, trajectory, _ = _load_assets(run_dir)
    simulation = MujocoSimulation(DEFAULT_SCENE_PATH)
    controller = RLController.from_run(run_dir, simulation.quad, device=device)
    tracker = TrajectoryController(
        controller,
        simulation.quad,
        trajectory,
        steps_per_reference=int(config["steps_per_action"]),
    )
    simulation.set_trajectory_visualization(trajectory[:, :3])
    return simulation, tracker, trajectory


def run_interactive(run_dir: str | Path, *, device: str = "cpu") -> None:
    simulation, tracker, _ = _deployment_tracker(Path(run_dir), device)
    simulation.run_interactive(tracker.step, tracker.reset)


def record_run(
        run_dir: str | Path,
        output_path: str | Path,
        *,
        device: str = "cpu",
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
) -> Path:
    simulation, tracker, trajectory = _deployment_tracker(Path(run_dir), device)
    maximum_steps = (
        (len(trajectory) - 1) * tracker.steps_per_reference
        + int(round(1.0 / simulation.quad.dt))
    )
    output_path = Path(output_path)
    simulation.run_recorded(
        tracker.step,
        maximum_steps,
        output_path,
        fps=fps,
        width=width,
        height=height,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--mode", choices=("metrics", "interactive", "record"), default="metrics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.mode == "metrics":
        results = evaluate_metrics(
            arguments.run_dir,
            seeds=tuple(range(arguments.episodes)),
            perturb_initial_state=not arguments.nominal,
            device=arguments.device,
        )
        print(json.dumps({key: value for key, value in results.items() if key != "episodes"}, indent=2))
    elif arguments.mode == "interactive":
        run_interactive(arguments.run_dir, device=arguments.device)
    else:
        output = arguments.output or arguments.run_dir / "evaluation.mp4"
        print(record_run(
            arguments.run_dir,
            output,
            device=arguments.device,
            fps=arguments.fps,
            width=arguments.width,
            height=arguments.height,
        ))


if __name__ == "__main__":
    main()
