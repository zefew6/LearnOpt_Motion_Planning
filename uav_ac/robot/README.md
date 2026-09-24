# Robot interfaces

Robot state, dynamics parameters, and commands are maintained in
`uav_ac.robot`. In a running MuJoCo simulation, use `simulation.robot` for the
aerial-manipulator interface.

## Aerial manipulator

The aerial manipulator combines a floating quadrotor, a four-joint arm, and a
parallel gripper. Its public configuration is a 12-element vector:

```text
[position_NED(3), quaternion_FRD_to_NED_wxyz(4), arm_positions(4), gripper_gap(1)]
```

The tangent velocity is an 11-element vector:

```text
[linear_velocity_NED(3), angular_velocity_FRD_body(3), arm_rates(4), gripper_gap_rate(1)]
```

The gripper gap is the distance between inner finger faces, from 0.020 m
closed to 0.070 m open. Its state also reports the left-right synchronization
error.

`simulation.robot` exposes the live `state`, `configuration`, `velocity`, and
`limits`, plus `mass`, `center_of_mass`, `gripper_opening`, and
`gripper_sync_error`. Model queries accept an optional configuration and
include:

```python
robot = simulation.robot
position, quaternion = robot.forward_kinematics(frame="grasp")
jacobian = robot.jacobian(frame="grasp")
properties = robot.mass_properties()
dynamics = robot.dynamics(robot.configuration, robot.velocity)
collision = robot.check_collision(clearance=0.02)
```

Frames are `"tool"` (wrist tool site) and `"grasp"` (gripper center). Pose
quaternions are scalar-first; Jacobian linear and angular rows are both
expressed in world NED. The Jacobian has shape 6×11. Dynamics returns the
reduced mass matrix, bias and passive forces, and actuation matrix. The
gripper's two sliders are represented as one symmetric gap coordinate in this
interface. Query methods use separate MuJoCo data and do not modify the live
simulation state.

Commands use `AerialManipulatorCommand`: collective thrust in newtons, body
moment in FRD Nm, four arm joint torques in Nm, and the requested gripper gap
in metres. Apply commands and advance the simulation once per physics step:

```python
from uav_ac.robot.aerial_manipulator import AerialManipulatorCommand

simulation.robot.apply(AerialManipulatorCommand(
    thrust=thrust,
    moment=body_moment,
    joint_torques=arm_torques,
    gripper_opening=0.060,
))
simulation.step()
```

`AerialManipulatorReference` carries configuration, velocity, and acceleration
vectors with sizes 12, 11, and 11. The aerial-manipulator hover example is
configured in [`configs/aerial_manipulator_hover.yaml`](../../configs/aerial_manipulator_hover.yaml).

The arm dimensions, masses, and inertias are simulation assumptions, not
measured hardware specifications. The hover example demonstrates basic flight
and joint motion; it does not implement whole-body obstacle avoidance or
grasping.

## Quadrotor

`uav_ac.robot.quadrotor.Quad` stores the quadrotor's NED/FRD state, inertia,
flight limits, and rotor allocation and motor-response parameters. Its
defaults are defined in `quad.py`; MuJoCo provides rigid-body dynamics in
simulation. The aerial-manipulator command interface uses the same collective
thrust, body-moment, and rotor model.
