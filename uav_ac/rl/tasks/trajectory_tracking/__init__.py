"""Trajectory-tracking RL workflow registration."""

from ...common.registry import TaskWorkflow
from .assets import prepare_assets
from .config import settings_from
from .evaluation import evaluate_metrics, record_run, run_interactive
from .training import train


WORKFLOW = TaskWorkflow(
    name="trajectory_tracking",
    settings_from=settings_from,
    train=train,
    prepare=prepare_assets,
    evaluate_metrics=evaluate_metrics,
    run_interactive=run_interactive,
    record_run=record_run,
    metric_wind_modes=("nominal", "random", "both"),
    replay_wind_modes=("nominal", "random"),
)
