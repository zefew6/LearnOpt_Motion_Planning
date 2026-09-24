"""Baseline whole-body reference controller for the aerial manipulator."""

import numpy as np

from uav_ac.robot.aerial_manipulator import AerialManipulatorCommand, AerialManipulatorReference
from .trajectory_controller import TrajectoryReference


class AerialManipulatorController:
    """Cascaded flight control plus arm inverse-dynamics tracking.

    The flight loop remains the repository's established cascaded controller.
    Arm feedforward uses the reduced MuJoCo mass matrix and bias/passive forces.
    This is a baseline tracking interface, not a whole-body optimizer.
    """

    def __init__(self, flight_controller, robot, quad, *, arm_kp=3.0, arm_kd=0.22):
        self.flight_controller = flight_controller
        self.robot = robot
        self.quad = quad
        self.arm_kp = float(arm_kp)
        self.arm_kd = float(arm_kd)
        self._reference_index = 0

    def reset(self):
        self.flight_controller.reset()
        self._reference_index = 0

    def step(self, reference: AerialManipulatorReference) -> AerialManipulatorCommand:
        if not isinstance(reference, AerialManipulatorReference):
            raise TypeError("reference must be an AerialManipulatorReference")
        qref, vref, aref = reference.configuration, reference.velocity, reference.acceleration
        yaw = _yaw(qref[3:7])
        row = np.r_[qref[:3], vref[:3], aref[:3], yaw]
        rows = np.vstack((row, row))
        flight = self.flight_controller.step(
            self.quad,
            TrajectoryReference(rows, self._reference_index, self.quad.dt),
        )
        self._reference_index ^= 1

        state = self.robot.state
        dynamics = self.robot.dynamics(state.configuration, state.velocity)
        torque = (dynamics["mass_matrix"][6:10] @ aref
                  + dynamics["bias_forces"][6:10]
                  - dynamics["passive_forces"][6:10]
                  + self.arm_kp*(qref[7:11]-state.joint_positions)
                  + self.arm_kd*(vref[6:10]-state.joint_velocities))

        # Compensate gravity torque caused by the arm moving the complete COM
        # away from the base origin. Moments are expressed in body FRD axes.
        com_body = self.quad.R().T @ (self.robot.center_of_mass-self.quad.position)
        gravity_body = self.quad.R().T @ np.array([0., 0., self.robot.mass*self.quad.g])
        com_moment = -np.cross(com_body, gravity_body)
        return AerialManipulatorCommand(
            flight.thrust, flight.moment+com_moment, torque, qref[11])


def _yaw(quaternion):
    w, x, y, z = quaternion
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


__all__ = ["AerialManipulatorController"]
