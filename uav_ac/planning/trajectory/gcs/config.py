"""Configuration for Bezier Graph-of-Convex-Sets planning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GCSConfig:
    degree: int = 5
    continuity: int = 2
    relaxation_degree: int = 1
    length_weight: float = 1.0
    derivative_order: int = 2
    derivative_weight: float = 0.05
    zero_endpoint_derivatives: int = 2
    regularization: float = 1.0e-8
    flow_tolerance: float = 1.0e-6
    feasibility_tolerance: float = 1.0e-7
    max_iterations: int = 200
    solver: str = "CLARABEL"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if not 0 <= self.continuity < self.degree:
            raise ValueError("continuity must satisfy 0 <= continuity < degree")
        if not 1 <= self.relaxation_degree <= self.degree:
            raise ValueError("relaxation_degree must lie in [1, degree]")
        if not 1 <= self.derivative_order <= self.degree:
            raise ValueError("derivative_order must lie in [1, degree]")
        if self.derivative_weight < 0.0:
            raise ValueError("derivative_weight must be nonnegative")
        if not 0 <= self.zero_endpoint_derivatives <= self.degree:
            raise ValueError("zero_endpoint_derivatives must lie in [0, degree]")
        if self.length_weight <= 0.0 or self.regularization < 0.0:
            raise ValueError("length_weight must be positive and regularization nonnegative")
        if self.flow_tolerance <= 0.0 or self.feasibility_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not self.solver:
            raise ValueError("solver name cannot be empty")


__all__ = ["GCSConfig"]
