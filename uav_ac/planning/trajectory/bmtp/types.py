"""Serializable planner records and physically timed analytic trajectories."""

from dataclasses import dataclass, field

import numpy as np

from .bezier import derivative_matrix, evaluate


@dataclass(frozen=True)
class BMTPTrajectory:
    control_points: np.ndarray
    segment_time: float

    def __post_init__(self):
        points = np.asarray(self.control_points, float)
        if points.ndim != 3 or points.shape[2] != 3 or min(points.shape[:2]) < 1:
            raise ValueError("control_points must have shape (segments, degree+1, 3)")
        if not np.all(np.isfinite(points)) or not np.isfinite(self.segment_time) or self.segment_time <= 0:
            raise ValueError("trajectory must be finite with positive segment time")
        object.__setattr__(self, "control_points", points.copy())

    @property
    def duration(self) -> float:
        return self.segment_time * len(self.control_points)

    def evaluate(self, times: np.ndarray | float, order: int = 0) -> np.ndarray:
        scalar = np.ndim(times) == 0
        times = np.asarray(times, float).reshape(-1)
        if np.any(~np.isfinite(times)) or np.any(times < -1e-10) or np.any(times > self.duration+1e-10):
            raise ValueError("evaluation time outside trajectory")
        indices = np.minimum((np.maximum(times, 0)/self.segment_time).astype(int), len(self.control_points)-1)
        parameters = np.clip(times/self.segment_time - indices, 0, 1)
        matrix = derivative_matrix(self.control_points.shape[1]-1, order) / self.segment_time**order
        result = np.empty((len(times), 3))
        for i in np.unique(indices):
            mask = indices == i
            result[mask] = evaluate(matrix @ self.control_points[i], parameters[mask])
        return result[0] if scalar else result

    def sample(self, dt: float) -> np.ndarray:
        """p/v/a at exact dt ticks; final fractional tick holds the endpoint."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        ticks = np.arange(int(np.ceil(self.duration / dt)) + 1)*dt
        times = np.minimum(ticks, self.duration)
        values = [self.evaluate(times, k) for k in range(3)]
        for value in values[1:]:
            value[ticks > self.duration] = 0
        return np.hstack(values)


@dataclass
class BMTPIteration:
    iteration: int
    trajectory: BMTPTrajectory
    accepted: bool
    collisions: tuple[tuple[int, int], ...]
    new_tags: tuple[tuple[int, int], ...]
    elapsed_seconds: float
    residuals: dict[str, float]


@dataclass
class BMTPResult:
    trajectory: BMTPTrajectory | None
    status: str
    message: str
    initial_path: np.ndarray
    initialization: BMTPTrajectory | None = None
    history: list[BMTPIteration] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    lower_bound_duration: float | None = None

    @property
    def success(self) -> bool:
        return self.trajectory is not None

    @property
    def converged(self) -> bool:
        return self.status == "converged"
