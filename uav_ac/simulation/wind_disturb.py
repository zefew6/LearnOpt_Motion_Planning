"""Reusable deterministic and randomized wind disturbances."""

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


@dataclass(frozen=True)
class RandomWindConfig:
    """Episode-level distribution for force disturbances in the NED frame."""

    probability: float = 0.5
    steady_horizontal_max: float = 1.0
    gust_horizontal_max: float = 0.5
    gust_vertical_max: float = 0.1
    angular_frequency_min: float = 0.3
    angular_frequency_max: float = 1.5
    curriculum_fraction: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("wind probability must be in [0, 1]")
        non_negative = (
            self.steady_horizontal_max,
            self.gust_horizontal_max,
            self.gust_vertical_max,
            self.angular_frequency_min,
            self.angular_frequency_max,
            self.curriculum_fraction,
        )
        if not np.all(np.isfinite(non_negative)) or np.any(np.asarray(non_negative) < 0.0):
            raise ValueError("wind limits must be finite and non-negative")
        if self.angular_frequency_max < self.angular_frequency_min:
            raise ValueError("wind maximum frequency must not be below its minimum")


def sample_gusting_crosswind(
        rng: np.random.Generator,
        config: RandomWindConfig,
        *,
        scale: float = 1.0,
) -> GustingCrosswind:
    """Sample one repeatable episode wind field from ``rng``."""
    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("wind scale must be in [0, 1]")
    steady_angle = rng.uniform(-np.pi, np.pi)
    steady_magnitude = rng.uniform(0.0, config.steady_horizontal_max) * scale
    steady = steady_magnitude * np.array([
        np.cos(steady_angle), np.sin(steady_angle), 0.0])

    gust_angle = rng.uniform(-np.pi, np.pi)
    gust_magnitude = rng.uniform(0.0, config.gust_horizontal_max) * scale
    gust = np.array([
        abs(gust_magnitude * np.cos(gust_angle)),
        abs(gust_magnitude * np.sin(gust_angle)),
        rng.uniform(0.0, config.gust_vertical_max) * scale,
    ])
    frequencies = rng.uniform(
        config.angular_frequency_min,
        config.angular_frequency_max,
        size=3,
    )
    phases = rng.uniform(-np.pi, np.pi, size=3)
    return GustingCrosswind(
        steady_force=tuple(steady),
        gust_force=tuple(gust),
        angular_frequency=tuple(frequencies),
        phase=tuple(phases),
    )


__all__ = ["GustingCrosswind", "RandomWindConfig", "sample_gusting_crosswind"]
