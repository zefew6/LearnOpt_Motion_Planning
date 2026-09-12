"""Unified evaluation, viewer, and recording entrypoint for RL tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from .common.registry import get_workflow, task_name_from_run

EvaluationMode = Literal["metrics", "interactive", "record"]
WindMode = Literal["nominal", "random", "both"]


def _workflow(run_dir: str | Path):
    return get_workflow(task_name_from_run(run_dir))


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
    workflow = _workflow(run_dir)
    if wind not in workflow.metric_wind_modes:
        raise ValueError(f"{workflow.name.replace('_', ' ')} does not support wind mode {wind}")
    return workflow.evaluate_metrics(
        run_dir, seeds=seeds, perturb_initial_state=perturb_initial_state,
        device=device, split=split, trajectory_id=trajectory_id, wind=wind)


def run_interactive(
        run_dir: str | Path,
        *,
        device: str = "cpu",
        split: str = "test",
        trajectory_id: int | None = None,
        windy: bool = False,
        seed: int = 0,
) -> None:
    return _workflow(run_dir).run_interactive(
        run_dir, device=device, split=split, trajectory_id=trajectory_id,
        windy=windy, seed=seed)


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
    return _workflow(run_dir).record_run(
        run_dir, output_path, device=device, fps=fps, width=width, height=height,
        split=split, trajectory_id=trajectory_id, windy=windy, seed=seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("metrics", "interactive", "record"), default="metrics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--trajectory-id", type=int)
    parser.add_argument("--wind", choices=("nominal", "random", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--nominal", "--nominal-start", dest="nominal_start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    workflow = _workflow(arguments.run_dir)
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
        print(json.dumps({key: value for key, value in results.items() if key != "episodes"}, indent=2))
        return
    if arguments.wind not in workflow.replay_wind_modes:
        raise ValueError(f"{workflow.name.replace('_', ' ')} does not support wind mode {arguments.wind}")
    windy = arguments.wind == "random"
    if arguments.mode == "interactive":
        workflow.run_interactive(
            arguments.run_dir, device=arguments.device, split=arguments.split,
            trajectory_id=arguments.trajectory_id, windy=windy, seed=arguments.seed)
        return
    output = arguments.output or arguments.run_dir / "evaluation.mp4"
    print(workflow.record_run(
        arguments.run_dir, output, device=arguments.device, fps=arguments.fps,
        width=arguments.width, height=arguments.height, split=arguments.split,
        trajectory_id=arguments.trajectory_id, windy=windy, seed=arguments.seed))


if __name__ == "__main__":
    main()
