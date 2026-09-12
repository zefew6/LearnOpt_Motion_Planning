"""Compatibility entrypoint for unified RL evaluation."""

from ..evaluate import (
    EvaluationMode,
    WindMode,
    evaluate_metrics,
    main,
    record_run,
    run_interactive,
)
from ..tasks.trajectory_tracking.evaluation import (
    _aggregate_results,
    _deployment_tracker,
    _load_run_assets,
    _wind_callbacks,
)

__all__ = [
    "EvaluationMode", "WindMode", "evaluate_metrics", "main", "record_run",
    "run_interactive",
]


if __name__ == "__main__":
    main()
