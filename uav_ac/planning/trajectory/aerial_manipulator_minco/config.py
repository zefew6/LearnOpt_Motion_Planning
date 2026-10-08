"""Small, validated configuration for the 8-D aerial-manipulator planner."""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class AerialManipulatorMINCOConfig:
    jerk_weights: tuple[float, ...] = (1., 1., 1., .2, .08, .08, .08, .08)
    time_weight: float = 10.0
    max_speed: float = 3.0
    max_acceleration: float = 3.0
    max_body_rate: float = 2.1
    max_yaw_rate: float = 1.2
    max_yaw_acceleration: float = 2.5
    joint_velocity_limits: tuple[float, ...] = (2., 2., 2., 2.)
    joint_acceleration_limits: tuple[float, ...] = (4., 4., 4., 4.)
    obstacle_clearance: float = .05
    # RRT uses the shared occupancy grid and full collision envelopes.
    rrt_obstacle_margin: float = .01
    esdf_discretization_margin: float = .035
    esdf_resolution: float = .02
    self_clearance: float = .015
    obstacle_weight: float = 2.0e4
    self_collision_weight: float = 1.0e4
    constraint_weight: float = 1.0e3
    smoothing_epsilon: float = .01
    integral_resolution: int = 6
    integral_resolution_floor_unloaded: int = 8
    integral_resolution_floor_loaded: int = 12
    max_iterations: int = 60
    lbfgs_memory: int = 12
    gradient_tolerance: float = 1.0e-4
    relative_cost_tolerance: float = 1.0e-3
    optimizer_waypoint_step_scale: float = .05
    rrt_step_size: float = .5
    rrt_joint_sampling_padding_rad: float = .2
    astar_guidance_enabled: bool = True
    astar_grid_resolution: float = .08
    astar_fallback_grid_resolution: float = .04
    astar_clearance_weight_m: float = .10
    astar_clearance_offset_m: float = .05
    astar_heuristic_weight: float = 2.0
    astar_guide_sample_spacing_m: float = .10
    astar_tube_std_m: float = .10
    rrt_simplify_attempts: int = 32
    rrt_simplify_budget_s: float = .02
    minco_sample_spacing_m: float = .8
    position_scale: float = .5
    yaw_scale: float = .7
    joint_scales: tuple[float, ...] = (.8, .8, .8, .8)
    joint_waypoint_parameterization: str = "tanh"
    edge_position_resolution: float = .04
    edge_yaw_resolution: float = .08
    edge_joint_resolution: float = .08
    validation_dt: float = .025
    minimum_total_time: float = .15
    initial_duration_scale: float = 1.2

    def __post_init__(self):
        for name in ("integral_resolution", "integral_resolution_floor_unloaded",
                     "integral_resolution_floor_loaded", "max_iterations", "lbfgs_memory",
                     "rrt_simplify_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.integral_resolution < 2:
            raise ValueError("integral_resolution must be at least two")
        if (self.integral_resolution_floor_unloaded < 2
                or self.integral_resolution_floor_loaded < 2):
            raise ValueError("integral resolution floors must be at least two")
        if self.max_iterations < 1 or self.lbfgs_memory < 1:
            raise ValueError("iteration counts must be positive")
        if self.rrt_simplify_attempts < 0:
            raise ValueError("rrt_simplify_attempts cannot be negative")
        for name in ("jerk_weights", "joint_velocity_limits", "joint_acceleration_limits",
                     "joint_scales"):
            values = np.asarray(getattr(self, name), dtype=float)
            expected = 8 if name == "jerk_weights" else 4
            if values.shape != (expected,) or np.any(values < 0) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {expected} finite non-negative values")
        positive = ("time_weight", "max_speed", "max_acceleration", "max_body_rate", "max_yaw_rate",
                    "max_yaw_acceleration", "obstacle_clearance",
                    "esdf_discretization_margin", "esdf_resolution", "self_clearance",
                    "obstacle_weight", "self_collision_weight",
                    "constraint_weight", "smoothing_epsilon", "gradient_tolerance",
                    "relative_cost_tolerance",
                    "optimizer_waypoint_step_scale",
                    "rrt_step_size", "minco_sample_spacing_m", "position_scale", "yaw_scale",
                    "rrt_joint_sampling_padding_rad",
                    "astar_grid_resolution", "astar_fallback_grid_resolution",
                    "astar_clearance_weight_m",
                    "astar_clearance_offset_m", "astar_heuristic_weight",
                    "astar_guide_sample_spacing_m",
                    "astar_tube_std_m", "rrt_simplify_budget_s",
                    "edge_position_resolution", "edge_yaw_resolution",
                    "edge_joint_resolution",
                    "validation_dt", "minimum_total_time")
        positive = positive + ("initial_duration_scale",)
        if any(isinstance(getattr(self, name), bool)
               or not np.isfinite(getattr(self, name))
               or getattr(self, name) <= 0 for name in positive):
            raise ValueError("continuous planner settings must be positive and finite")
        if self.esdf_discretization_margin < np.sqrt(3.0)*self.esdf_resolution:
            raise ValueError(
                "esdf_discretization_margin must cover one voxel diagonal")
        if self.astar_heuristic_weight < 1.0:
            raise ValueError("astar_heuristic_weight must be at least one")
        if (isinstance(self.rrt_obstacle_margin, bool)
                or not np.isfinite(self.rrt_obstacle_margin)
                or self.rrt_obstacle_margin < 0.0):
            raise ValueError("rrt_obstacle_margin must be finite and non-negative")
        if self.joint_waypoint_parameterization not in {"direct", "tanh"}:
            raise ValueError("joint_waypoint_parameterization must be 'direct' or 'tanh'")
        if not isinstance(self.astar_guidance_enabled, (bool, np.bool_)):
            raise ValueError("astar_guidance_enabled must be boolean")

    @classmethod
    def from_mapping(cls, settings):
        """Load the compact set of supported planner settings."""
        values = dict(settings)
        accepted = {field.name for field in fields(cls)}
        unknown = sorted(set(values)-accepted)
        if unknown:
            raise ValueError(f"unknown aerial_manipulator_minco settings: {', '.join(unknown)}")
        return cls(**values)


__all__ = ["AerialManipulatorMINCOConfig"]
