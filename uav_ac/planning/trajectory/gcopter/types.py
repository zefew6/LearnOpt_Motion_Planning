"""Public value types returned by the GCOPTER planner."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .mappings import polynomial_basis_matrix, polynomial_bases


class HalfSpaceRegion(Protocol):
    A: np.ndarray
    b: np.ndarray


def coerce_region(
    region: HalfSpaceRegion | tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(region, "A") and hasattr(region, "b"):
        A, b = region.A, region.b
    else:
        try:
            A, b = region
        except (TypeError, ValueError) as error:
            raise TypeError("each corridor region must expose A/b or be an (A, b) pair") from error
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[1] != 3 or len(A) != len(b) or len(A) < 4:
        raise ValueError("half-spaces must have shapes A=(m, 3), b=(m,), m>=4")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("half-spaces must be finite")
    return A.copy(), b.copy()


@dataclass(frozen=True)
class TrajectorySamples:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    jerks: np.ndarray


@dataclass(frozen=True)
class GCOPTERTrajectory:
    """Piecewise quintic trajectory with coefficients in ascending powers."""

    durations: np.ndarray
    coefficients: np.ndarray
    corridor_indices: np.ndarray
    cost: float
    iterations: int
    converged: bool
    message: str

    @property
    def duration(self) -> float:
        return float(np.sum(self.durations))

    @property
    def piece_count(self) -> int:
        return int(len(self.durations))

    def evaluate(self, times: float | np.ndarray, derivative: int = 0) -> np.ndarray:
        if not 0 <= derivative <= 5:
            raise ValueError("derivative must lie in [0, 5]")
        query = np.asarray(times, dtype=float)
        scalar = query.ndim == 0
        query = np.clip(query.reshape(-1), 0.0, self.duration)
        boundaries = np.cumsum(self.durations)
        pieces = np.minimum(np.searchsorted(boundaries, query, side="right"), self.piece_count - 1)
        starts = np.concatenate(([0.0], boundaries[:-1]))
        local_times = query - starts[pieces]
        basis = polynomial_basis_matrix(local_times, derivative)
        result = np.einsum("nk,nkc->nc", basis, self.coefficients[pieces])
        return result[0] if scalar else result

    def sample(self, dt: float = 0.01) -> TrajectorySamples:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        times = np.arange(0.0, self.duration, dt)
        if len(times) == 0 or times[-1] < self.duration:
            times = np.append(times, self.duration)
        boundaries = np.cumsum(self.durations)
        pieces = np.minimum(
            np.searchsorted(boundaries, times, side="right"), self.piece_count - 1)
        starts = np.concatenate(([0.0], boundaries[:-1]))
        bases = polynomial_bases(times - starts[pieces])[:4]
        values = np.einsum("dnk,nkc->dnc", bases, self.coefficients[pieces])
        return TrajectorySamples(times, values[0], values[1], values[2], values[3])

    def maximum_corridor_violation(
        self,
        corridor: Sequence[HalfSpaceRegion | tuple[np.ndarray, np.ndarray]],
        samples_per_piece: int = 50,
    ) -> float:
        halfspaces = [coerce_region(region) for region in corridor]
        maximum = -np.inf
        for index, duration in enumerate(self.durations):
            local_times = np.linspace(0.0, duration, samples_per_piece + 1)
            basis = polynomial_basis_matrix(local_times, 0)
            points = basis @ self.coefficients[index]
            A, b = halfspaces[int(self.corridor_indices[index])]
            maximum = max(maximum, float(np.max(points @ A.T - b)))
        return maximum


__all__ = ["GCOPTERTrajectory", "HalfSpaceRegion", "TrajectorySamples", "coerce_region"]
