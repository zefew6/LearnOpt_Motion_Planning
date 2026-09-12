"""Compatibility exports for the task-owned gate-racing workflow."""

from ..tasks.gate_racing.config import contract, read_run, scene_path, settings_from
from ..tasks.gate_racing.environment import make_environment
from ..tasks.gate_racing.evaluation import evaluate_metrics, evaluate_model, load_model, replay
from ..tasks.gate_racing.training import RacingEvaluationCallback, train

__all__ = [
    "RacingEvaluationCallback", "contract", "evaluate_metrics", "evaluate_model",
    "load_model", "make_environment", "read_run", "replay", "scene_path",
    "settings_from", "train",
]
