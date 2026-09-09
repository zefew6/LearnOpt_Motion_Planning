"""Single configuration entry point for planning, training and deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

# Stable imports for existing analysis scripts. New applications should use
# ``uav_ac.experiments.runner`` instead of importing business logic from main.
from uav_ac.control import TrajectoryController
from uav_ac.planning.pipeline import generate_minimum_snap_mission as _generate_mission_trajectory
from uav_ac.runtime import (
    build_controller as _build_controller,
    plan_trajectory as _plan_trajectory,
    sample_open_field_mission as _sample_open_field_mission,
    trajectory_after_takeoff as _trajectory_after_takeoff,
    wind_control_callbacks as _wind_control_callbacks,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "lab_gcopter_cascaded.yaml"


def main(config_path: str | Path | None = None):
    """Resolve and execute one experiment YAML file."""
    from uav_ac.experiments.runner import run
    return run(DEFAULT_CONFIG if config_path is None else config_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
