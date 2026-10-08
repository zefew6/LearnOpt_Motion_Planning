"""Interactive MuJoCo planning and controller deployment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# GCOPTER solves many small systems; extra BLAS workers reduce throughput.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Preserve public entrypoint imports for existing callers.
from uav_ac.runners.config import (
    CONTROLLERS, DEFAULT_CONFIG, MODEL_DIRECTORY, PLANNERS, ROOT, TASKS, load_config,
)
# from uav_ac.control.factory import build_controller
# from uav_ac.planning.pipeline.flight import plan_trajectory


def run(config: dict) -> np.ndarray | dict:
    """Dispatch one interactive flight task."""
    task = config.get("task", "trajectory_tracking")
    if task == "gate_racing":
        from uav_ac.runners.gate_racing import run_gate_racing
        return run_gate_racing(config)
    if task == "aerial_pick_place":
        from uav_ac.runners.aerial_pick_place import run_aerial_pick_place
        return run_aerial_pick_place(config)

    from uav_ac.runners.trajectory_tracking import run_trajectory_tracking
    return run_trajectory_tracking(config)


def main(config_path: str | Path = DEFAULT_CONFIG) -> np.ndarray | dict:
    return run(load_config(config_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
