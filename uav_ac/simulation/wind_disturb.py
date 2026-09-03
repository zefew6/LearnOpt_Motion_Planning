"""Reusable wind-disturbance models for flight simulation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GustingCrosswind:
    """Deterministic, bounded wind force expressed in the NED frame."""

    steady_force: tuple[float, float, float] = (0.08, 0.30, 0.0)
    gust_force: tuple[float, float, float] = (0.16, 0.12, 0.04)
    angular_frequency: tuple[float, float, float] = (0.73, 1.37, 0.91)
    phase: tuple[float, float, float] = (0.0, 0.4, 1.1)

    def __post_init__(self) -> None:
        for name in (
                "steady_force", "gust_force", "angular_frequency", "phase"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
        if np.any(np.asarray(self.gust_force) < 0.0):
            raise ValueError("gust_force amplitudes must be non-negative")
        if np.any(np.asarray(self.angular_frequency) < 0.0):
            raise ValueError("angular_frequency values must be non-negative")

    def force_ned(self, time: float) -> np.ndarray:
        """Return the disturbance force in newtons at simulation time ``time``."""
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        steady = np.asarray(self.steady_force, dtype=float)
        amplitude = np.asarray(self.gust_force, dtype=float)
        frequency = np.asarray(self.angular_frequency, dtype=float)
        phase = np.asarray(self.phase, dtype=float)
        return steady + amplitude * np.sin(frequency * time + phase)


__all__ = ["GustingCrosswind"]
