"""Result types produced by FIRI corridor generation."""

from dataclasses import dataclass

import numpy as np

from uav_ac.planning.geometry import ConvexPolytope, Ellipsoid


@dataclass(frozen=True)
class FIRIRegion(ConvexPolytope):
    """A convex FIRI region and the last feasible inflation ellipsoid."""

    iterations: int
    center: np.ndarray
    shape: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        ellipsoid = Ellipsoid(self.center, self.shape)
        object.__setattr__(self, "center", ellipsoid.center)
        object.__setattr__(self, "shape", ellipsoid.shape)

    @property
    def ellipsoid(self) -> Ellipsoid:
        return Ellipsoid(self.center, self.shape)

    @property
    def ellipsoid_volume(self) -> float:
        return self.ellipsoid.volume


@dataclass(frozen=True)
class FreeSpaceCover:
    """Convex cover evaluated on collision-checked free-space samples."""

    regions: list[FIRIRegion]
    samples: np.ndarray
    covered: np.ndarray

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.covered)) if len(self.covered) else 1.0
