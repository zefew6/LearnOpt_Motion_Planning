"""GCS solver and trajectory result types."""

from dataclasses import dataclass

import numpy as np

from .bezier import evaluate_bezier


@dataclass(frozen=True)
class GCSRelaxation:
    flows: np.ndarray
    tail_control_points: np.ndarray
    head_control_points: np.ndarray
    objective: float
    iterations: int
    status: str


@dataclass(frozen=True)
class GCSTrajectory:
    control_points: np.ndarray
    region_indices: np.ndarray
    vertex_path: tuple[int, ...]
    edge_path: tuple[int, ...]
    relaxed_objective: float
    objective: float
    relaxation_iterations: int
    restriction_iterations: int
    status: str

    @property
    def segment_count(self) -> int:
        return len(self.control_points)

    def sample(self, samples_per_segment: int = 40) -> np.ndarray:
        if samples_per_segment < 2:
            raise ValueError("samples_per_segment must be at least two")
        times = np.linspace(0.0, 1.0, samples_per_segment)
        pieces = [evaluate_bezier(segment, times) for segment in self.control_points]
        return np.vstack((pieces[0], *(piece[1:] for piece in pieces[1:])))


__all__ = ["GCSRelaxation", "GCSTrajectory"]
