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
from uav_ac.simulation.mujoco_sim import (
    DEFAULT_SCENE_PATH,
    ENU_TO_NED,
    OPEN_FIELD_SCENE_PATH,
    MujocoSimulation,
)
from uav_ac.simulation.wind_disturb import RandomWindConfig, sample_gusting_crosswind

from .assets import InitializationLibrary
from .environment import MujocoTrajectoryTrackingEnv
from .trajectory_bank import TrajectoryBank


EvaluationMode = Literal["metrics", "interactive", "record"]
WindMode = Literal["nominal", "random", "both"]


def _load_run_assets(
        run_dir: Path,
) -> tuple[dict[str, Any], TrajectoryBank | np.ndarray, InitializationLibrary | None]:
    with (run_dir / RL_CONFIG_FILENAME).open(encoding="utf-8") as file:
        config = json.load(file)
    bank_directory = config.get("trajectory_bank_directory")
    if bank_directory is not None:
        return config, TrajectoryBank.load(run_dir / bank_directory), None
    trajectory = np.load(run_dir / config["trajectory_file"])
    library = InitializationLibrary.load(run_dir / config["initialization_library_file"])
    return config, trajectory, library


def evaluate_metrics(
        run_dir: str | Path,
        *,
        seeds: tuple[int, ...] = tuple(range(20)),
        perturb_initial_state: bool = True,
        device: str = "cpu",
        split: str = "test",
        trajectory_id: int | None = None,
        wind: WindMode = "both",
) -> dict[str, Any]:
    """Evaluate full trajectories and persist nominal/windy aggregate metrics."""
    run_dir = Path(run_dir)
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    config, assets, legacy_library = _load_run_assets(run_dir)
    is_bank = isinstance(assets, TrajectoryBank)
    scene_path = OPEN_FIELD_SCENE_PATH if is_bank else DEFAULT_SCENE_PATH
    wind_config = (
        RandomWindConfig(**config["wind"])
        if is_bank else None
    )
    environment = MujocoTrajectoryTrackingEnv(
        assets,
        legacy_library,
        model_path=scene_path,
        steps_per_action=int(config["steps_per_action"]),
        random_start=False,
        perturb_initial_state=perturb_initial_state,
        curriculum_progress=1.0,
        split=split if is_bank else "train",
        wind_config=wind_config,
    )
    model = PPO.load(run_dir / "best_model.zip", device=device)
    if is_bank:
        trajectory_ids = (
            np.array([trajectory_id], dtype=np.int64)
            if trajectory_id is not None else assets.indices(split)
        )
        if len(seeds) < len(trajectory_ids):
            trajectory_ids = trajectory_ids[:len(seeds)]
    else:
        trajectory_ids = np.zeros(len(seeds), dtype=np.int64)
    wind_conditions = {
        "nominal": (False,),
        "random": (True,),
        "both": (False, True),
    }[wind]
    if not is_bank:
        wind_conditions = (False,)

    episodes = []
    try:
        for index, selected_id in enumerate(trajectory_ids):
            for windy in wind_conditions:
                episode_seed = int(seeds[index % len(seeds)] + 100_000 * int(windy))
                observation, _ = environment.reset(
                    seed=episode_seed,
                    options={
                        "trajectory_id": int(selected_id),
                        "start_index": 0,
                        "perturbation_scale": 1.0 if perturb_initial_state else 0.0,
                        "wind_enabled": windy,
                        "wind_scale": 1.0,
                    },
                )
                maximum_steps = len(environment.trajectory) + int(round(
                    1.0 / environment.control_dt)) + 1
                episode_return = 0.0
                final_info: dict[str, Any] = {}
                step_index = -1
                for step_index in range(maximum_steps):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, final_info = environment.step(action)
                    episode_return += float(reward)
                    if terminated or truncated:
                        break
                episodes.append({
                    "trajectory_id": int(selected_id),
                    "split": split if is_bank else "legacy",
                    "seed": episode_seed,
                    "wind_enabled": windy,
                    "success": bool(final_info.get("success", False)),
                    "collision": bool(final_info.get("collision", False)),
                    "failure_reason": final_info.get("failure_reason"),
                    "steps": step_index + 1,
                    "return": episode_return,
                    "position_rmse": float(final_info.get("position_rmse", np.inf)),
                    "final_position_error": float(final_info.get("position_error", np.inf)),
                })
    finally:
        environment.close()

    results = _aggregate_results(episodes)
    results.update({
        "split": split if is_bank else "legacy",
        "wind_mode": wind,
        "perturbed_initial_state": bool(perturb_initial_state),
        "acceptance": {
            "nominal_success_rate_min": 0.9,
            "windy_success_rate_min": 0.8,
            "successful_mean_position_rmse_max": 0.75,
        },
        "episodes": episodes,
    })
    results["acceptance_passed"] = bool(
        (results["nominal_success_rate"] is None
         or results["nominal_success_rate"] >= 0.9)
        and (results["windy_success_rate"] is None
             or results["windy_success_rate"] >= 0.8)
        and results["successful_mean_position_rmse"] is not None
        and results["successful_mean_position_rmse"] < 0.75
    )
    output_path = run_dir / f"evaluation_{results['split']}_{wind}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")
    return results


def _aggregate_results(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("evaluation requires at least one episode")
    successful_rmses = [
        episode["position_rmse"] for episode in episodes if episode["success"]]
    nominal = [episode for episode in episodes if not episode["wind_enabled"]]
    windy = [episode for episode in episodes if episode["wind_enabled"]]
    return {
        "episode_count": len(episodes),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "nominal_success_rate": (
            float(np.mean([episode["success"] for episode in nominal])) if nominal else None),
        "windy_success_rate": (
            float(np.mean([episode["success"] for episode in windy])) if windy else None),
        "successful_mean_position_rmse": (
            float(np.mean(successful_rmses)) if successful_rmses else None),
    }


def _deployment_tracker(
        run_dir: Path,
        device: str,
        *,
        split: str,
        trajectory_id: int | None,
) -> tuple[MujocoSimulation, TrajectoryController, np.ndarray, dict[str, Any]]:
    config, assets, _ = _load_run_assets(run_dir)
    if isinstance(assets, TrajectoryBank):
        allowed = assets.indices(split)
        selected_id = int(allowed[0] if trajectory_id is None else trajectory_id)
        if selected_id not in allowed:
            raise ValueError(f"trajectory_id {selected_id} is not in split '{split}'")
        trajectory = assets.trajectory(selected_id)
        scene_path = OPEN_FIELD_SCENE_PATH
    else:
        if trajectory_id not in (None, 0):
            raise ValueError("legacy runs only contain trajectory_id 0")
        trajectory = assets
        scene_path = DEFAULT_SCENE_PATH
    simulation = MujocoSimulation(scene_path)
    simulation.set_goal_position(trajectory[-1, :3])
    controller = RLController.from_run(run_dir, simulation.quad, device=device)
    tracker = TrajectoryController(
        controller,
        simulation.quad,
        trajectory,
        steps_per_reference=int(config["steps_per_action"]),
    )
    simulation.set_trajectory_visualization(trajectory[:, :3])
    return simulation, tracker, trajectory, config


def _wind_callbacks(
        simulation: MujocoSimulation,
        tracker: TrajectoryController,
        config: dict[str, Any],
        *,
        windy: bool,
        seed: int,
):
    if not windy:
        return tracker.step, tracker.reset
    wind = sample_gusting_crosswind(
        np.random.default_rng(seed),
        RandomWindConfig(**config.get("wind", {})),
        scale=1.0,
    )

    def step() -> None:
        simulation.set_external_force_world(
            ENU_TO_NED @ wind.force_ned(float(simulation.data.time)))
        tracker.step()

    def reset() -> None:
        tracker.reset()
        simulation.set_external_force_world(np.zeros(3))

    return step, reset


def run_interactive(
        run_dir: str | Path,
        *,
        device: str = "cpu",
        split: str = "test",
        trajectory_id: int | None = None,
        windy: bool = False,
        seed: int = 0,
) -> None:
    simulation, tracker, _, config = _deployment_tracker(
        Path(run_dir), device, split=split, trajectory_id=trajectory_id)
    step, reset = _wind_callbacks(
        simulation, tracker, config, windy=windy, seed=seed)
    simulation.run_interactive(step, reset)


def record_run(
        run_dir: str | Path,
        output_path: str | Path,
        *,
        device: str = "cpu",
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        split: str = "test",
        trajectory_id: int | None = None,
        windy: bool = False,
        seed: int = 0,
) -> Path:
    simulation, tracker, trajectory, config = _deployment_tracker(
        Path(run_dir), device, split=split, trajectory_id=trajectory_id)
    step, _ = _wind_callbacks(
        simulation, tracker, config, windy=windy, seed=seed)
    maximum_steps = (
        (len(trajectory) - 1) * tracker.steps_per_reference
        + int(round(1.0 / simulation.quad.dt))
    )
    output_path = Path(output_path)
    simulation.run_recorded(
        step,
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
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--trajectory-id", type=int)
    parser.add_argument("--wind", choices=("nominal", "random", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--nominal", "--nominal-start", dest="nominal_start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    windy = arguments.wind == "random"
    if arguments.mode == "metrics":
        results = evaluate_metrics(
            arguments.run_dir,
            seeds=tuple(arguments.seed + index for index in range(arguments.episodes)),
            perturb_initial_state=not arguments.nominal_start,
            device=arguments.device,
            split=arguments.split,
            trajectory_id=arguments.trajectory_id,
            wind=arguments.wind,
        )
        print(json.dumps({
            key: value for key, value in results.items() if key != "episodes"
        }, indent=2))
    elif arguments.mode == "interactive":
        if arguments.wind == "both":
            raise ValueError("interactive mode requires --wind nominal or --wind random")
        run_interactive(
            arguments.run_dir,
            device=arguments.device,
            split=arguments.split,
            trajectory_id=arguments.trajectory_id,
            windy=windy,
            seed=arguments.seed,
        )
    else:
        if arguments.wind == "both":
            raise ValueError("record mode requires --wind nominal or --wind random")
        output = arguments.output or arguments.run_dir / "evaluation.mp4"
        print(record_run(
            arguments.run_dir,
            output,
            device=arguments.device,
            fps=arguments.fps,
            width=arguments.width,
            height=arguments.height,
            split=arguments.split,
            trajectory_id=arguments.trajectory_id,
            windy=windy,
            seed=arguments.seed,
        ))


if __name__ == "__main__":
    main()
