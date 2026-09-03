"""Configuration for GCOPTER trajectory optimization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GCOPTERConfig:
    """Settings for geometrically constrained trajectory optimization."""

    time_weight: float = 20.0
    length_per_piece: float = 1.0
    max_velocity: float = 3.0
    max_acceleration: float = 6.0
    mass: float = 0.5
    gravity: float = 9.81
    min_thrust: float = 0.4
    max_thrust: float = 18.0
    max_tilt_angle: float = 0.7
    max_body_rate: float = 2.1
    position_weight: float = 1.0e5
    velocity_weight: float = 1.0e4
    acceleration_weight: float = 1.0e4
    thrust_weight: float = 1.0e4
    tilt_weight: float = 1.0e4
    body_rate_weight: float = 1.0e4
    smoothing_epsilon: float = 1.0e-2
    integral_resolution: int = 12
    max_iterations: int = 80
    gradient_tolerance: float = 1.0e-5
    relative_cost_tolerance: float = 1.0e-5
    lbfgs_memory: int = 16
    minimum_piece_time: float = 1.0e-3
    feasible_iteration_patience: int = 2
    # Accelerated projected-gradient steps used only for the initial
    # polytope-coordinate inverse map, not for the main L-BFGS solve.
    inverse_map_iterations: int = 40

    def __post_init__(self) -> None:
        positive = {
            "time_weight": self.time_weight,
            "length_per_piece": self.length_per_piece,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "mass": self.mass,
            "gravity": self.gravity,
            "max_thrust": self.max_thrust,
            "max_tilt_angle": self.max_tilt_angle,
            "max_body_rate": self.max_body_rate,
            "position_weight": self.position_weight,
            "velocity_weight": self.velocity_weight,
            "acceleration_weight": self.acceleration_weight,
            "thrust_weight": self.thrust_weight,
            "tilt_weight": self.tilt_weight,
            "body_rate_weight": self.body_rate_weight,
            "smoothing_epsilon": self.smoothing_epsilon,
            "gradient_tolerance": self.gradient_tolerance,
            "relative_cost_tolerance": self.relative_cost_tolerance,
            "minimum_piece_time": self.minimum_piece_time,
        }
        if any(value <= 0.0 for value in positive.values()):
            raise ValueError("all continuous GCOPTER settings must be positive")
        if (self.integral_resolution < 2 or self.max_iterations <= 0
                or self.lbfgs_memory <= 0 or self.feasible_iteration_patience <= 0
                or self.inverse_map_iterations <= 0):
            raise ValueError("invalid integral resolution or optimizer iteration setting")
        if self.min_thrust < 0.0 or self.min_thrust >= self.max_thrust:
            raise ValueError("thrust bounds must satisfy 0 <= min_thrust < max_thrust")
        if self.max_tilt_angle >= 0.5 * np.pi:
            raise ValueError("max_tilt_angle must be smaller than pi/2")
