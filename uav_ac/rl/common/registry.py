"""Explicit registry for task-owned RL workflows."""

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TaskWorkflow:
    name: str
    settings_from: Callable
    train: Callable
    prepare: Callable | None
    evaluate_metrics: Callable
    run_interactive: Callable
    record_run: Callable
    metric_wind_modes: tuple[str, ...]
    replay_wind_modes: tuple[str, ...]


def _build_registry(entries: tuple[tuple[str, str], ...]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for name, module in entries:
        if name in modules:
            raise RuntimeError(f"duplicate RL task workflow registration: {name}")
        modules[name] = module
    return modules


_TASK_MODULES = _build_registry((
    ("trajectory_tracking", "uav_ac.rl.tasks.trajectory_tracking"),
    ("gate_racing", "uav_ac.rl.tasks.gate_racing"),
))


def task_name(values: dict) -> str:
    name = values.get("task", "trajectory_tracking")
    if name not in _TASK_MODULES:
        raise ValueError(f"unknown training task: {name}")
    return name


def task_name_from_run(run_dir: str | Path) -> str:
    with (Path(run_dir) / "rl_config.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    return task_name(metadata)


def get_workflow(name: str) -> TaskWorkflow:
    if name not in _TASK_MODULES:
        raise ValueError(f"unknown training task: {name}")
    module = import_module(_TASK_MODULES[name])
    workflow = getattr(module, "WORKFLOW", None)
    if not isinstance(workflow, TaskWorkflow) or workflow.name != name:
        raise RuntimeError(f"invalid RL task workflow registration: {name}")
    return workflow


def registered_tasks() -> tuple[str, ...]:
    return tuple(_TASK_MODULES)
