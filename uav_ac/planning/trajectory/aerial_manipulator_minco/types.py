"""Public result type and controller-reference conversion for 8-D MINCO."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..gcopter.mappings import polynomial_basis_matrix


@dataclass(frozen=True)
class AerialManipulatorTrajectory:
    durations: np.ndarray
    coefficients: np.ndarray
    rrt_path: np.ndarray
    cost: float
    iterations: int
    optimizer_converged: bool
    optimizer_message: str
    validation_passed: bool = False
    minimum_clearance: float = float("nan")
    maximum_violation: float = float("inf")
    validation_sample_dt: float = float("nan")

    def __post_init__(self):
        durations = np.asarray(self.durations, dtype=float).copy()
        coefficients = np.asarray(self.coefficients, dtype=float).copy()
        path = np.asarray(self.rrt_path, dtype=float).copy()
        if (durations.ndim != 1 or len(durations) < 1 or np.any(durations <= 0)
                or coefficients.shape != (len(durations), 6, 8)
                or path.ndim != 2 or path.shape[1] != 8
                or not np.all(np.isfinite(durations))
                or not np.all(np.isfinite(coefficients))
                or not np.all(np.isfinite(path))
                or not np.allclose(durations, durations[0], rtol=1e-10, atol=1e-12)):
            raise ValueError("trajectory arrays must be finite, shaped correctly, and equally timed")
        for value in (durations, coefficients, path):
            value.setflags(write=False)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "rrt_path", path)

    @property
    def total_time(self):
        return float(np.sum(self.durations))

    def evaluate(self, time, derivative=0):
        query = np.asarray(time, dtype=float)
        scalar = query.ndim == 0
        flat = np.clip(query.reshape(-1), 0., self.total_time)
        boundaries = np.cumsum(self.durations)
        pieces = np.minimum(np.searchsorted(boundaries, flat, side="right"), len(self.durations)-1)
        starts = np.r_[0., boundaries[:-1]]
        local = flat-starts[pieces]
        result = np.einsum("nk,nkd->nd", polynomial_basis_matrix(local, derivative),
                           self.coefficients[pieces])
        return result[0] if scalar else result

    def reference(self, time, robot, *, gripper_opening):
        """Convert flat-output state into a full NED/FRD robot reference."""
        sigma, velocity, acceleration, jerk = (
            self.evaluate(time, order) for order in range(4))
        rotation, angular_velocity = _flatness_attitude(
            acceleration[:3], jerk[:3], sigma[3], velocity[3])
        xyzw = Rotation.from_matrix(rotation).as_quat()
        quaternion = np.r_[xyzw[3], xyzw[:3]]
        if np.dot(quaternion, robot.configuration[3:7]) < 0.0:
            quaternion = -quaternion
        q = np.r_[sigma[:3], quaternion, sigma[4:8], float(gripper_opening)]
        v = np.r_[velocity[:3], angular_velocity, velocity[4:8], 0.]
        a = np.r_[acceleration[:3], np.zeros(3), acceleration[4:8], 0.]
        return robot_reference(q, v, a)


def robot_reference(configuration, velocity, acceleration):
    from uav_ac.robot.aerial_manipulator import AerialManipulatorReference
    return AerialManipulatorReference(configuration, velocity, acceleration)


def _flatness_attitude(acceleration, jerk, yaw, yaw_rate, gravity=9.81):
    force = np.array([-acceleration[0], -acceleration[1], gravity-acceleration[2]])
    force_rate = np.array([-jerk[0], -jerk[1], -jerk[2]])
    magnitude = max(float(np.linalg.norm(force)), 1e-8)
    b3 = force/magnitude
    b3_rate = (force_rate-b3*np.dot(b3, force_rate))/magnitude
    b1c = np.array([np.cos(yaw), np.sin(yaw), 0.])
    b1c_rate = yaw_rate*np.array([-np.sin(yaw), np.cos(yaw), 0.])
    cross = np.cross(b3, b1c)
    cross_rate = np.cross(b3_rate, b1c)+np.cross(b3, b1c_rate)
    norm = max(float(np.linalg.norm(cross)), 1e-8)
    b2 = cross/norm
    b2_rate = (cross_rate-b2*np.dot(b2, cross_rate))/norm
    b1 = np.cross(b2, b3)
    b1_rate = np.cross(b2_rate, b3)+np.cross(b2, b3_rate)
    rotation = np.column_stack((b1, b2, b3))
    derivative = np.column_stack((b1_rate, b2_rate, b3_rate))
    skew = rotation.T@derivative
    omega_body = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return rotation, omega_body


__all__ = ["AerialManipulatorTrajectory"]
