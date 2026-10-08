"""Cold-plan timing for the whole-body pick/place planner.

Each isolated run rebuilds the binary RRT occupancy map and signed ESDF.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import scipy
from statistics import median

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


def _minco_summary(rows):
    measured = [row for row in rows
                if row.get("minco_optimizer_seconds") is not None]
    if not measured:
        return None
    times = np.asarray([row["minco_optimizer_seconds"] for row in measured], float)
    call_counts = np.asarray([
        sum(int(metric.get("objective_calls", 0))
            for metric in row.get("legs", {}).values())
        for row in measured], dtype=int)
    piece_counts = {
        leg: [int(row.get("legs", {}).get(leg, {}).get("minco_pieces", 0))
              for row in measured if leg in row.get("legs", {})]
        for leg in ("pick", "place")
    }
    return {
        "measured": len(measured),
        "under_3s": int(np.sum(times < 3.0)),
        "median_s": median(times),
        "p95_s": float(np.percentile(times, 95)),
        "maximum_s": float(np.max(times)),
        "objective_calls_total": int(np.sum(call_counts)),
        "objective_calls_median": float(np.median(call_counts)),
        "objective_calls_p95": float(np.percentile(call_counts, 95)),
        "objective_calls_maximum": int(np.max(call_counts)),
        "pieces_by_leg": {
            leg: {"measured": len(values), "median": float(np.median(values)),
                  "maximum": int(np.max(values))}
            for leg, values in piece_counts.items() if values
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aerial_manipulator_workcell.yaml")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--search-only", action="store_true",
                        help="measure both RRT searches without MINCO optimization")
    parser.add_argument("--in-process", action="store_true",
                        help="run all repeats in one process")
    parser.add_argument("--_single-run-repeat", type=int, default=-1,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 0:
        parser.error("--repeats cannot be negative")

    config = load_config(args.config)
    planner_config = AerialManipulatorMINCOConfig.from_mapping(
        config["aerial_manipulator_minco"])
    runs = ([args._single_run_repeat] if args._single_run_repeat >= 0 else
            list(range(args.repeats)))
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mujoco": mujoco.__version__,
        "machine": platform.machine(),
        "processor": _processor_name(),
        "logical_cpus": os.cpu_count(),
        "planner_config": asdict(planner_config),
        "repeats": len(runs),
        "search_only": args.search_only,
        "process_isolation": not args.in_process,
    }
    print(json.dumps({"environment": environment}, sort_keys=True,
                     default=_json_default), flush=True)
    if not args.in_process and args._single_run_repeat < 0:
        rows = []
        for index, repeat in enumerate(runs):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--config", args.config, "--repeats", "1",
                       "--_single-run-repeat", str(repeat)]
            if args.search_only:
                command.append("--search-only")
            wall_started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True)
            worker_wall = time.perf_counter()-wall_started
            row = None
            for line in completed.stdout.splitlines():
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict) and "run" in decoded:
                    row = decoded
            if row is None:
                row = {
                    "repeat": repeat, "success": False,
                    "function_success": False if not args.search_only else None,
                    "rrt_success": False,
                    "failure": completed.stderr[-2000:] or
                    f"worker exited with status {completed.returncode}",
                    "planning_seconds": None,
                }
            row["run"] = index
            row["process_startup_seconds"] = max(
                0., worker_wall-float(row.get("planning_seconds") or 0.))
            rows.append(row)
            print(json.dumps(row, sort_keys=True, default=_json_default), flush=True)
        failures = sum(not bool(row.get("success")) for row in rows)
        rrt_rows = [row for row in rows if row.get("rrt_seconds_pick_place") is not None]
        durations = np.asarray([row["rrt_seconds_pick_place"] for row in rrt_rows], float)
        rrt_summary = None if not len(durations) else {
            "measured": int(len(durations)),
            "successful_search_pairs": int(sum(bool(row.get("rrt_success"))
                                                for row in rrt_rows)),
            "median_s": median(durations),
            "p95_s": float(np.percentile(durations, 95)),
            "maximum_s": float(np.max(durations)),
            "under_1s": int(sum(bool(row.get("rrt_success"))
                                and row.get("rrt_under_1s") for row in rrt_rows)),
            "under_1s_fraction": float(np.mean([
                bool(row.get("rrt_success")) and bool(row.get("rrt_under_1s"))
                for row in rrt_rows])),
        }
        print(json.dumps({"summary": {
            "runs": len(runs),
            "function_successes": sum(bool(row.get("function_success")) for row in rows),
            "function_failures": None if args.search_only else failures,
            "search_successes": sum(bool(row.get("rrt_success")) for row in rows),
            "search_failures": sum(not bool(row.get("rrt_success")) for row in rows),
            "rrt_pairs_measured": int(sum(bool(row.get("rrt_success"))
                                           for row in rrt_rows)),
            "rrt_pick_place": rrt_summary,
            "minco_optimizer": _minco_summary(rows),
            "passed": failures == 0,
        }}, sort_keys=True, default=_json_default), flush=True)
        return int(failures != 0)
    failures = 0
    measured_rrt = []
    rrt_successes = 0
    function_successes = 0
    for index, repeat in enumerate(runs):
        trial = dict(config)
        simulation = MujocoSimulation(trial["scene"], record_actual_trajectory=False)
        diagnostics = {}
        started = time.perf_counter()
        try:
            result = plan_pick_place(
                simulation, trial, diagnostics=diagnostics,
                search_only=args.search_only)
            duration = time.perf_counter()-started
            if args.search_only:
                passed = (set(result["searches"]) == {"pick", "place"}
                          and all(search.exact_solution
                                  for search in result["searches"].values()))
            else:
                minco_seconds = diagnostics.get("minco_optimizer_seconds")
                passed = (all(plan.validation_passed
                              and plan.optimizer_converged
                              for plan in result["plans"].values())
                          and minco_seconds is not None and minco_seconds < 3.0)
            leg_data = diagnostics.get("legs", {})
            leg_rrt = [leg_data[name].get("rrt_seconds") for name in ("pick", "place")
                       if name in leg_data and "rrt_seconds" in leg_data[name]]
            rrt_total = float(sum(leg_rrt)) if len(leg_rrt) == 2 else None
            leg_search_success = [
                bool(leg_data[name].get("exact_solution",
                                        leg_data[name].get("rrt_path_states", 0) >= 2))
                and int(leg_data[name].get("rrt_path_states", 0)) >= 2
                for name in ("pick", "place") if name in leg_data]
            rrt_success = len(leg_search_success) == 2 and all(leg_search_success)
            if rrt_total is not None:
                measured_rrt.append(rrt_total)
                rrt_successes += int(rrt_success)
            function_successes += int(passed and not args.search_only)
            row = {
                "run": index,
                "repeat": repeat,
                "success": passed,
                "function_success": None if args.search_only else passed,
                "rrt_success": rrt_success,
                "rrt_seconds_pick_place": rrt_total,
                "rrt_under_1s": (None if rrt_total is None else
                                 bool(rrt_success and rrt_total < 1.0)),
                "planning_seconds": duration,
                "minco_optimizer_seconds": diagnostics.get("minco_optimizer_seconds"),
                "occupancy_seconds": diagnostics.get("occupancy_seconds"),
                "esdf_seconds": diagnostics.get("esdf_seconds"),
                "esdf_grid_shape": diagnostics.get("esdf_grid_shape"),
                "esdf_grid_voxels": diagnostics.get("esdf_grid_voxels"),
                "esdf_grid_bounds": diagnostics.get("esdf_grid_bounds"),
                "esdf_resolution": diagnostics.get("esdf_resolution"),
                "legs": diagnostics.get("legs", {}),
                "rrt_waypoints": ({name: len(search.path)
                                   for name, search in result["searches"].items()}
                                  if args.search_only else
                                  {name: len(plan.rrt_path)
                                   for name, plan in result["plans"].items()}),
            }
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures += 1
            leg_data = diagnostics.get("legs", {})
            leg_rrt = [leg_data[name].get("rrt_seconds") for name in ("pick", "place")
                       if name in leg_data and "rrt_seconds" in leg_data[name]]
            rrt_total = float(sum(leg_rrt)) if len(leg_rrt) == 2 else None
            leg_search_success = [
                bool(leg_data[name].get("exact_solution",
                                        leg_data[name].get("rrt_path_states", 0) >= 2))
                and int(leg_data[name].get("rrt_path_states", 0)) >= 2
                for name in ("pick", "place") if name in leg_data]
            rrt_success = len(leg_search_success) == 2 and all(leg_search_success)
            if rrt_total is not None:
                measured_rrt.append(rrt_total)
                rrt_successes += int(rrt_success)
            row = {
                "run": index,
                "repeat": repeat,
                "success": False,
                "function_success": False if not args.search_only else None,
                "rrt_success": rrt_success,
                "rrt_seconds_pick_place": rrt_total,
                "rrt_under_1s": (None if rrt_total is None else
                                 bool(rrt_success and rrt_total < 1.0)),
                "planning_seconds": time.perf_counter()-started,
                "minco_optimizer_seconds": diagnostics.get("minco_optimizer_seconds"),
                "failure": f"{type(error).__name__}: {error}",
                "occupancy_seconds": diagnostics.get("occupancy_seconds"),
                "esdf_seconds": diagnostics.get("esdf_seconds"),
                "esdf_grid_shape": diagnostics.get("esdf_grid_shape"),
                "esdf_grid_voxels": diagnostics.get("esdf_grid_voxels"),
                "esdf_grid_bounds": diagnostics.get("esdf_grid_bounds"),
                "esdf_resolution": diagnostics.get("esdf_resolution"),
                "legs": leg_data,
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
    rrt_summary = None
    if measured_rrt:
        rrt_summary = {
            "measured": len(measured_rrt),
            "successful_search_pairs": rrt_successes,
            "median_s": median(measured_rrt),
            "p95_s": float(np.percentile(measured_rrt, 95)),
            "maximum_s": float(np.max(measured_rrt)),
            "under_1s": int(np.sum(np.asarray(measured_rrt) < 1.0)),
            "under_1s_fraction": float(np.mean(np.asarray(measured_rrt) < 1.0)),
        }
    print(json.dumps({"summary": {"runs": len(runs), "function_successes": function_successes,
                                  "function_failures": None if args.search_only else failures,
                                  "search_successes": rrt_successes,
                                  "search_failures": len(runs)-rrt_successes,
                                  "rrt_pairs_measured": rrt_successes,
                                  "rrt_pick_place": rrt_summary,
                                  "minco_optimizer": _minco_summary(rows),
                                  "passed": failures == 0}}, sort_keys=True,
                     default=_json_default), flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
