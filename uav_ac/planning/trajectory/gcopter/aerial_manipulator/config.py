"""Small, validated configuration for the 8-D aerial-manipulator planner."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AerialManipulatorGCOPTERConfig:
    pieces: int = 6
    jerk_weights: tuple[float, ...] = (1., 1., 1., .2, .08, .08, .08, .08)
    time_weight: float = 2.0
    max_speed: float = 2.0
    max_acceleration: float = 3.0
    max_body_rate: float = 2.1
    max_yaw_rate: float = 1.2
    max_yaw_acceleration: float = 2.5
    joint_velocity_limits: tuple[float, ...] = (1.5, 1.5, 1.5, 1.5)
    joint_acceleration_limits: tuple[float, ...] = (4., 4., 4., 4.)
    obstacle_clearance: float = .05
    self_clearance: float = .015
    payload_radius: float = .035
    obstacle_weight: float = 2.0e4
    self_collision_weight: float = 1.0e4
    constraint_weight: float = 1.0e3
    smoothing_epsilon: float = .01
    integral_resolution: int = 6
    max_iterations: int = 60
    lbfgs_memory: int = 12
    gradient_tolerance: float = 1.0e-4
    rrt_step_size: float = .25
    rrt_max_iterations: int = 2500
    position_scale: float = .5
    yaw_scale: float = .7
    joint_scales: tuple[float, ...] = (.8, .8, .8, .8)
    edge_position_resolution: float = .04
    edge_yaw_resolution: float = .08
    edge_joint_resolution: float = .08
    validation_dt: float = .025
    minimum_total_time: float = .15

    def __post_init__(self):
        for name in ("pieces", "integral_resolution", "max_iterations",
                     "lbfgs_memory", "rrt_max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.pieces < 1 or self.integral_resolution < 2:
            raise ValueError("pieces must be positive and integral_resolution at least two")
        if self.max_iterations < 1 or self.lbfgs_memory < 1 or self.rrt_max_iterations < 1:
            raise ValueError("iteration counts must be positive")
        for name in ("jerk_weights", "joint_velocity_limits", "joint_acceleration_limits",
                     "joint_scales"):
            values = np.asarray(getattr(self, name), dtype=float)
            expected = 8 if name == "jerk_weights" else 4
            if values.shape != (expected,) or np.any(values < 0) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {expected} finite non-negative values")
        positive = ("time_weight", "max_speed", "max_acceleration", "max_body_rate", "max_yaw_rate",
                    "max_yaw_acceleration", "obstacle_clearance", "self_clearance",
                    "payload_radius", "obstacle_weight", "self_collision_weight",
                    "constraint_weight", "smoothing_epsilon", "gradient_tolerance",
                    "rrt_step_size", "position_scale", "yaw_scale",
                    "edge_position_resolution", "edge_yaw_resolution",
                    "edge_joint_resolution", "validation_dt", "minimum_total_time")
        if any(isinstance(getattr(self, name), bool)
               or not np.isfinite(getattr(self, name))
               or getattr(self, name) <= 0 for name in positive):
            raise ValueError("continuous planner settings must be positive and finite")


__all__ = ["AerialManipulatorGCOPTERConfig"]
