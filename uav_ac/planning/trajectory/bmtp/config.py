"""BMTP settings in SI units. No simulator or upstream planner dependency."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BMTPLimits:
    velocity: float = 3.0
    acceleration: float = 3.0
    jerk: float | None = 15.0
    snap: float | None = 30.0

    def __post_init__(self):
        if self.snap is not None and self.jerk is None:
            raise ValueError("snap requires a jerk limit")
        for value in self.values:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("derivative limits must be finite and positive")

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(v for v in (self.velocity, self.acceleration, self.jerk, self.snap) if v is not None)


@dataclass(frozen=True)
class BMTPConfig:
    degree: int = 8
    plane_degree: int = 1
    continuity_order: int = 4
    terminal_order: int = 2
    relative_tolerance: float = 0.01
    max_iterations: int = 50
    collision_tolerance: float = 1e-4
    collision_max_depth: int = 32
    trajectory_margin: float = 1e-4
    obstacle_margin: float = 1e-6
    feasibility_tolerance: float = 1e-5
    solver_tolerance: float = 1e-8
    solver_max_iterations: int = 300
    trajectory_backend: str = "cvxpy"
    # Fixed Clarabel plane slots per spline segment; unused slots are relaxed.
    active_plane_slots: int = 16

    def __post_init__(self):
        if self.degree < 2*self.terminal_order+1 or self.degree <= self.continuity_order:
            raise ValueError("degree insufficient for terminal/continuity order")
        if min(self.terminal_order, self.continuity_order, self.plane_degree) < 0:
            raise ValueError("orders must be nonnegative")
        if self.max_iterations < 1 or self.collision_max_depth < 1 or self.solver_max_iterations < 1:
            raise ValueError("iteration/depth limits must be positive")
        if self.trajectory_backend not in {"cvxpy", "clarabel"}:
            raise ValueError("trajectory_backend must be 'cvxpy' or 'clarabel'")
        if self.active_plane_slots < 1:
            raise ValueError("active_plane_slots must be positive")
        for value in (self.relative_tolerance, self.collision_tolerance,
                      self.trajectory_margin, self.obstacle_margin,
                      self.feasibility_tolerance, self.solver_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("tolerances and margins must be finite and positive")
