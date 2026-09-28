"""Whole-body aerial-manipulator MINCO planning."""

from .config import AerialManipulatorGCOPTERConfig
from .planner import AerialManipulatorGCOPTER
from .task_targets import make_terminal_state
from .types import AerialManipulatorTrajectory

__all__ = [
    "AerialManipulatorGCOPTER", "AerialManipulatorGCOPTERConfig",
    "AerialManipulatorTrajectory", "make_terminal_state",
]
