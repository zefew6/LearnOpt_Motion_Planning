"""Results passed between the planning pipeline stages."""

from dataclasses import dataclass

import numpy as np

from ..corridor.firi import FIRIRegion


@dataclass(frozen=True)
class MissionCorridor:
    regions: list[FIRIRegion]
    fixed_boundaries: list[tuple[int, np.ndarray]]
    rrt_legs: list[np.ndarray]


__all__ = ["MissionCorridor"]
