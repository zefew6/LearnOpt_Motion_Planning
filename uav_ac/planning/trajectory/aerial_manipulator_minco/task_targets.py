"""Fixed 8-D terminal-state construction for pick and place events."""

import numpy as np


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def make_terminal_state(
        robot, target_position_ned, nominal_yaw, nominal_joints, *, gripper_opening,
        workspace_bounds=None):
    """Return ``[base_position, yaw, arm_joints]`` placing grasp frame on target."""
    target = np.asarray(target_position_ned, dtype=float)
    joints = np.asarray(nominal_joints, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_position_ned must be a finite 3-vector")
    if joints.shape != (4,) or not np.all(np.isfinite(joints)):
        raise ValueError("nominal_joints must contain four finite values")
    if (np.any(joints < robot.limits.joint_lower)
            or np.any(joints > robot.limits.joint_upper)):
        raise ValueError("nominal joints exceed robot limits")
    q = robot.configuration.copy()
    q[:3] = 0.0
    q[3:7] = yaw_quaternion(float(nominal_yaw))
    q[7:11] = joints
    q[11] = float(gripper_opening)
    if not robot.limits.gripper_opening[0] <= q[11] <= robot.limits.gripper_opening[1]:
        raise ValueError("gripper opening exceeds robot limits")
    grasp_at_origin, _ = robot.forward_kinematics(q, frame="grasp")
    q[:3] = target-grasp_at_origin
    if workspace_bounds is not None:
        bounds = np.asarray(workspace_bounds, dtype=float)
        if bounds.shape != (2, 3) or np.any(q[:3] < bounds[0]) or np.any(q[:3] > bounds[1]):
            raise ValueError("computed base target lies outside workspace bounds")
    grasp, _ = robot.forward_kinematics(q, frame="grasp")
    if np.linalg.norm(grasp-target) >= 1.0e-6:
        raise ValueError("computed terminal state does not reach grasp target")
    return np.r_[q[:3], float(nominal_yaw), joints]


__all__ = ["make_terminal_state", "quaternion_yaw", "yaw_quaternion"]
