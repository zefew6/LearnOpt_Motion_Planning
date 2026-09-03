"""Application-level defaults for composing search, corridor and trajectory layers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CorridorPlanningConfig:
    rrt_step_size: float = 1.5
    rrt_max_iterations: int = 2000
    rrt_seed: int = 7
    obstacle_padding: float = 0.15
    obstacle_point_spacing: float = 0.5
    firi_max_iterations: int = 1
    seed_spacing: float = 5.0
    local_half_size: tuple[float, float, float] = (2.5, 2.5, 1.5)
    seed_window_size: int = 2


__all__ = ["CorridorPlanningConfig"]
