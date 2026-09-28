"""Reproducible cold-plan timing for the whole-body pick/place planner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import scipy

from uav_ac.main import load_config
from uav_ac.planning.trajectory.aerial_manipulator_minco import (
    AerialManipulatorMINCOConfig,
)
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.tasks.aerial_pick_place import plan_pick_place


def _json_default(value):
    """Convert NumPy scalar diagnostics to standard JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _processor_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.uname().processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aerial_manipulator_pick_place.yaml")
    parser.add_argument("--repeats-seed7", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()
    if args.repeats_seed7 < 0:
        parser.error("--repeats-seed7 cannot be negative")
    try:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    except ValueError:
        parser.error("--seeds must be comma-separated nonnegative integers")
    if any(seed < 0 for seed in seeds):
        parser.error("seeds must be nonnegative")

    config = load_config(args.config)
    planner_config = AerialManipulatorMINCOConfig(
        **config["aerial_manipulator_minco"])
    budget = float(planner_config.planning_budget_s)
    runs = [7]*args.repeats_seed7+seeds
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mujoco": mujoco.__version__,
        "machine": platform.machine(),
        "processor": _processor_name(),
        "logical_cpus": os.cpu_count(),
        "budget_s": budget,
        "planner_config": asdict(planner_config),
        "seeds": runs,
    }
    print(json.dumps({"environment": environment}, sort_keys=True,
                     default=_json_default), flush=True)
    failures = 0
    for index, seed in enumerate(runs):
        trial = dict(config)
        trial["seed"] = seed
        simulation = MujocoSimulation(trial["scene"], record_actual_trajectory=False)
        diagnostics = {}
        started = time.perf_counter()
        try:
            result = plan_pick_place(simulation, trial, diagnostics=diagnostics)
            duration = time.perf_counter()-started
            passed = (duration <= budget and all(
                plan.validation_passed for plan in result["plans"].values()))
            row = {
                "run": index,
                "seed": seed,
                "success": passed,
                "planning_seconds": duration,
                "esdf_seconds": diagnostics.get("esdf_seconds"),
                "legs": diagnostics.get("legs", {}),
                "rrt_waypoints": {name: len(plan.rrt_path)
                                  for name, plan in result["plans"].items()},
            }
        except (ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError) as error:
            failures += 1
            row = {
                "run": index,
                "seed": seed,
                "success": False,
                "planning_seconds": time.perf_counter()-started,
                "failure": f"{type(error).__name__}: {error}",
                "esdf_seconds": diagnostics.get("esdf_seconds"),
                "legs": diagnostics.get("legs", {}),
                "partial_validation": {
                    name: {"validation_passed": plan.validation_passed,
                           "optimizer_converged": plan.optimizer_converged,
                           "maximum_violation": plan.maximum_violation,
                           "maximum_violation_kind": diagnostics.get("legs", {}).get(
                               name, {}).get("maximum_violation_kind"),
                           "minimum_clearance": plan.minimum_clearance}
                    for name, plan in diagnostics.get("plans", {}).items()
                },
            }
        if not row["success"]:
            failures += int("failure" not in row)
        print(json.dumps(row, sort_keys=True, default=_json_default), flush=True)
    print(json.dumps({"summary": {"runs": len(runs), "failures": failures,
                                  "passed": failures == 0}}, sort_keys=True,
                     default=_json_default), flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
