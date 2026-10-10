"""NED/FRD differential-flatness recovery for the simplified planning model."""

import numpy as np
from scipy.spatial.transform import Rotation

def recover_state(derivatives, gripper_opening, gravity=9.81):
    """Recover configuration and FRD tangent velocity from p/v/a/j outputs."""
    values = np.asarray(derivatives, dtype=float)
    if values.shape != (4, 8) or not np.all(np.isfinite(values)):
        raise ValueError('flat outputs must be finite (4, 8) derivatives')
    y, velocity, acceleration, jerk = values
    force = np.array([-acceleration[0], -acceleration[1], gravity-acceleration[2]])
    if np.linalg.norm(force) < 1e-8:
        raise ValueError('flatness singularity: zero thrust direction')
    heading = np.array([np.cos(y[3]), np.sin(y[3]), 0.])
    if np.linalg.norm(np.cross(force / np.linalg.norm(force), heading)) < 1e-8:
        raise ValueError('flatness singularity: heading parallel to thrust')
    rotation, omega = _flatness_attitude(acceleration[:3], jerk[:3], y[3], velocity[3], gravity)
    xyzw = Rotation.from_matrix(rotation).as_quat()
    q = np.r_[y[:3], xyzw[3], xyzw[:3], y[4:8], gripper_opening]
    v = np.r_[velocity[:3], omega, velocity[4:8], 0.]
    return q, v


def stationary_waypoint(robot, waypoint, yaw_joints, gripper_opening):
    """Eliminate base translation at a zero-v/a task event, with analytic pullback."""
    theta = np.asarray(yaw_joints, dtype=float)
    if theta.shape != (5,) or not np.all(np.isfinite(theta)):
        raise ValueError('yaw/joints must be a finite 5-vector')
    y = np.r_[np.zeros(3), theta]
    q, _ = recover_state(np.vstack((y, np.zeros((3, 8)))), gripper_opening)
    position, _ = robot.forward_kinematics(q, frame='grasp', check_limits=False)
    jacobian = robot.jacobian(q, frame='grasp', check_limits=False)
    y[:3] = waypoint.position - position
    derivative = np.zeros((8, 5))
    derivative[3:, :] = np.eye(5)
    derivative[:3, 0] = -jacobian[:3, 5]
    derivative[:3, 1:] = -jacobian[:3, 6:10]
    return y, derivative

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


def _flatness_attitude_tangent_jacobian(acceleration, yaw, gravity=9.81):
    """Map acceleration/yaw perturbations to a world-angle tangent.

    The result is ``d theta_world / d [ax, ay, az, yaw]`` for a left
    perturbation ``dR = [d theta_world]x R``. It differentiates the same
    normalized thrust and heading construction used by ``_flatness_attitude``.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
        raise ValueError("acceleration must be a finite 3-vector")
    force = np.array([-acceleration[0], -acceleration[1], gravity-acceleration[2]])
    force_norm = max(float(np.linalg.norm(force)), 1e-8)
    body_z = force/force_norm
    force_jacobian = np.zeros((3, 4))
    force_jacobian[:, :3] = np.diag([-1., -1., -1.])
    body_z_jacobian = ((np.eye(3)-np.outer(body_z, body_z))/force_norm) @ force_jacobian
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
    heading_jacobian = np.zeros((3, 4))
    heading_jacobian[:, 3] = [-np.sin(yaw), np.cos(yaw), 0.]
    cross = np.cross(body_z, heading)
    cross_norm = max(float(np.linalg.norm(cross)), 1e-8)
    body_y = cross/cross_norm
    cross_jacobian = (-_skew(heading)@body_z_jacobian
                      +_skew(body_z)@heading_jacobian)
    body_y_jacobian = ((np.eye(3)-np.outer(body_y, body_y))/cross_norm) @ cross_jacobian
    body_x = np.cross(body_y, body_z)
    body_x_jacobian = (-_skew(body_z)@body_y_jacobian
                       +_skew(body_y)@body_z_jacobian)
    rotation = np.column_stack((body_x, body_y, body_z))
    tangent_jacobian = np.empty((3, 4))
    for index in range(4):
        derivative = np.column_stack((body_x_jacobian[:, index],
                                      body_y_jacobian[:, index],
                                      body_z_jacobian[:, index]))
        skew = derivative@rotation.T
        tangent_jacobian[:, index] = [skew[2, 1], skew[0, 2], skew[1, 0]]
    return rotation, tangent_jacobian


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])




def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


__all__ = [
    'recover_state',
    'stationary_waypoint',
    'yaw_quaternion',
    'quaternion_yaw',
]
