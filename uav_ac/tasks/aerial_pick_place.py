"""Deterministic aerial-manipulator pick, carry and place simulation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar

import numpy as np

from uav_ac.control import CascadedConfig, CascadedController
from uav_ac.control.aerial_manipulator_controller import AerialManipulatorController
from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.trajectory.gcopter.aerial_manipulator import (
    AerialManipulatorGCOPTER, AerialManipulatorGCOPTERConfig, make_terminal_state,
)
from uav_ac.planning.trajectory.gcopter.aerial_manipulator.task_targets import quaternion_yaw
from uav_ac.simulation.mujoco_sim import MujocoSimulation


class PickPlaceState(Enum):
    PLAN_TO_PICK = auto()
    MOVE_TO_PICK = auto()
    GRASP = auto()
    PLAN_TO_PLACE = auto()
    MOVE_TO_PLACE = auto()
    RELEASE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PickPlaceStateMachine:
    state: PickPlaceState = PickPlaceState.PLAN_TO_PICK
    holding_payload: bool = False
    failure_reason: str | None = None
    events: list[str] = field(default_factory=lambda: [PickPlaceState.PLAN_TO_PICK.name])

    _allowed: ClassVar[dict] = {
        PickPlaceState.PLAN_TO_PICK: {PickPlaceState.MOVE_TO_PICK, PickPlaceState.FAILED},
        PickPlaceState.MOVE_TO_PICK: {PickPlaceState.GRASP, PickPlaceState.FAILED},
        PickPlaceState.GRASP: {PickPlaceState.PLAN_TO_PLACE, PickPlaceState.FAILED},
        PickPlaceState.PLAN_TO_PLACE: {PickPlaceState.MOVE_TO_PLACE, PickPlaceState.FAILED},
        PickPlaceState.MOVE_TO_PLACE: {PickPlaceState.RELEASE, PickPlaceState.FAILED},
        PickPlaceState.RELEASE: {PickPlaceState.DONE, PickPlaceState.FAILED},
        PickPlaceState.DONE: set(), PickPlaceState.FAILED: set(),
    }

    def transition(self, target, reason=None):
        if target not in self._allowed[self.state]:
            raise ValueError(f"invalid pick/place transition {self.state.name} -> {target.name}")
        if target is PickPlaceState.FAILED and not reason:
            raise ValueError("failed transition requires a reason")
        self.state = target
        self.events.append(target.name)
        if target is PickPlaceState.PLAN_TO_PLACE:
            self.holding_payload = True
        elif target is PickPlaceState.DONE:
            self.holding_payload = False
        self.failure_reason = reason


def run_aerial_pick_place(config):
    """Plan both legs before moving, then execute both with the baseline controller."""
    simulation = MujocoSimulation(config["scene"], record_actual_trajectory=False)
    robot = simulation.robot
    settings = config["pick_place"]
    planner_config = AerialManipulatorGCOPTERConfig(**config["aerial_manipulator_gcopter"])
    pick = np.asarray(settings["pick_position_ned"], dtype=float)
    place = np.asarray(settings["place_position_ned"], dtype=float)
    pick_yaw, place_yaw = float(settings["pick_yaw"]), float(settings["place_yaw"])
    gap_open, gap_closed = float(settings["gripper_open"]), float(settings["gripper_closed"])
    nominal_pick = np.asarray(settings["pick_nominal_joints"], dtype=float)
    nominal_place = np.asarray(settings["place_nominal_joints"], dtype=float)
    bounds = simulation.space_limits
    esdf = _scene_esdf(simulation, planner_config)
    start_q = robot.configuration.copy()
    start = np.r_[start_q[:3], quaternion_yaw(start_q[3:7]), start_q[7:11]]
    pick_state = make_terminal_state(
        robot, pick, pick_yaw, nominal_pick, gripper_opening=gap_open,
        workspace_bounds=bounds)
    place_state = make_terminal_state(
        robot, place, place_yaw, nominal_place, gripper_opening=gap_closed,
        workspace_bounds=bounds)
    planner = AerialManipulatorGCOPTER(planner_config)
    rng = np.random.default_rng(int(config["seed"]))
    machine = PickPlaceStateMachine()
    plans = {}
    try:
        plans["pick"] = planner.plan(
            start, pick_state, robot=robot, esdf=esdf, quad=simulation.quad,
            workspace_bounds=bounds, gripper_opening=gap_open, rng=rng)
        if not plans["pick"].validation_passed:
            raise RuntimeError("pick trajectory failed dense validation")
        plans["place"] = planner.plan(
            pick_state, place_state, robot=robot, esdf=esdf, quad=simulation.quad,
            workspace_bounds=bounds, gripper_opening=gap_closed,
            carry_payload=True, rng=rng)
        if not plans["place"].validation_passed:
            raise RuntimeError("carry trajectory failed dense validation")
        machine.transition(PickPlaceState.MOVE_TO_PICK)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        if machine.state is not PickPlaceState.FAILED:
            machine.transition(PickPlaceState.FAILED, str(error))
        result = _result(machine, plans, simulation, controller=None)
        _print_result(result)
        return result

    simulation.set_mocap_position_ned("pick_target_marker", pick)
    simulation.set_mocap_position_ned("place_target_marker", place)
    simulation.set_mocap_position_ned("payload_marker", pick)
    control_dt = float(config["control_dt"])
    stride = int(round(control_dt/simulation.quad.dt))
    if stride < 1 or not np.isclose(stride*simulation.quad.dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of the XML timestep")
    cascaded_values = config.get("cascaded", {})
    if cascaded_values:
        CascadedConfig(**cascaded_values).apply_to(simulation.quad)
    flight = CascadedController(simulation.quad.g, control_dt)
    controller = AerialManipulatorController(flight, robot, simulation.quad)
    execution = _Execution(
        simulation, controller, plans, machine, settings, pick, place,
        gap_open, gap_closed, control_dt, stride, start_q)
    if config["visualize"]:
        simulation.run_interactive(execution.step, execution.reset,
                                   chase_camera=config["follow_camera"])
    else:
        max_duration = (plans["pick"].total_time+plans["place"].total_time
                        + 2*float(settings["event_timeout"])
                        + 2*float(settings["settle_time"])+2.)
        max_steps = int(np.ceil(max_duration/simulation.quad.dt))
        for _ in range(max_steps):
            execution.step()
            simulation.step()
            if execution.done:
                break
    if not execution.done:
        machine.transition(PickPlaceState.FAILED, "execution_timeout")
    result = _result(machine, plans, simulation, controller, execution)
    _print_result(result)
    return result


class _Execution:
    def __init__(self, simulation, controller, plans, machine, settings,
                 pick, place, gap_open, gap_closed, control_dt, stride, start_q):
        self.simulation, self.robot = simulation, simulation.robot
        self.controller, self.plans, self.machine = controller, plans, machine
        self.settings, self.pick, self.place = settings, pick, place
        self.gap_open, self.gap_closed = gap_open, gap_closed
        self.control_dt, self.stride, self.start_q = control_dt, stride, start_q.copy()
        self.physics_index = 0
        self.stage_start = simulation.time
        self.stable_time = 0.0
        self.failure_time = simulation.time
        self.command_reference = plans["pick"].reference(
            0., self.robot, gripper_opening=gap_open)
        self.max_joint_error = 0.0
        self.last_settle_metrics = {}

    @property
    def done(self):
        return self.machine.state in {PickPlaceState.DONE, PickPlaceState.FAILED}

    def reset(self):
        self.robot.reset(self.start_q)
        self.controller.reset()
        self.machine = PickPlaceStateMachine(PickPlaceState.MOVE_TO_PICK)
        self.machine.events = [PickPlaceState.PLAN_TO_PICK.name,
                               PickPlaceState.MOVE_TO_PICK.name]
        self.simulation.set_mocap_position_ned("pick_target_marker", self.pick)
        self.simulation.set_mocap_position_ned("place_target_marker", self.place)
        self.simulation.set_mocap_position_ned("payload_marker", self.pick)
        self.physics_index = 0
        self.stage_start = self.simulation.time
        self.stable_time = 0.0
        self.command_reference = self.plans["pick"].reference(
            0., self.robot, gripper_opening=self.gap_open)

    def step(self):
        if self.done:
            return
        now = self.simulation.time
        if self.physics_index % self.stride == 0:
            self.command_reference = self._reference(now)
            error = np.linalg.norm(
                self.command_reference.configuration[7:11]-self.robot.configuration[7:11])
            self.max_joint_error = max(self.max_joint_error, float(error))
        self.robot.apply(self.controller.step(self.command_reference))
        if self.machine.holding_payload:
            self.simulation.set_mocap_position_ned(
                "payload_marker", self.robot.forward_kinematics(frame="grasp")[0])
        if self.simulation.collision_detected:
            self.machine.transition(PickPlaceState.FAILED, "simulation_collision")
        self.physics_index += 1

    def _reference(self, now):
        state = self.machine.state
        if state is PickPlaceState.MOVE_TO_PICK:
            trajectory = self.plans["pick"]
            elapsed = now-self.stage_start
            reference = trajectory.reference(
                min(elapsed, trajectory.total_time), self.robot,
                gripper_opening=self.gap_open)
            if elapsed >= trajectory.total_time:
                reference = self._align_grasp_reference(reference, self.pick)
            if elapsed >= trajectory.total_time and self._settled(self.pick, now):
                self.machine.transition(PickPlaceState.GRASP)
                self.stage_start = now
            elif elapsed > trajectory.total_time+float(self.settings["event_timeout"]):
                self.machine.transition(PickPlaceState.FAILED, "pick_event_timeout")
            return reference
        if state is PickPlaceState.GRASP:
            reference = self._terminal_with_gap(self.gap_closed)
            if (now-self.stage_start >= float(self.settings["settle_time"])
                    and abs(self.robot.gripper_opening-self.gap_closed) <= .005):
                if not self._state_matches(self.plans["pick"].evaluate(
                        self.plans["pick"].total_time), .06):
                    self.machine.transition(PickPlaceState.FAILED, "carry_start_state_mismatch")
                    return reference
                self.machine.transition(PickPlaceState.PLAN_TO_PLACE)
                self.machine.transition(PickPlaceState.MOVE_TO_PLACE)
                self.stage_start = now
            elif now-self.stage_start > float(self.settings["event_timeout"]):
                self.machine.transition(PickPlaceState.FAILED, "gripper_close_timeout")
            return reference
        if state is PickPlaceState.MOVE_TO_PLACE:
            trajectory = self.plans["place"]
            elapsed = now-self.stage_start
            reference = trajectory.reference(
                min(elapsed, trajectory.total_time), self.robot,
                gripper_opening=self.gap_closed)
            if elapsed >= trajectory.total_time:
                reference = self._align_grasp_reference(reference, self.place)
            if elapsed >= trajectory.total_time and self._settled(self.place, now):
                self.machine.transition(PickPlaceState.RELEASE)
                self.stage_start = now
            elif elapsed > trajectory.total_time+float(self.settings["event_timeout"]):
                self.machine.transition(PickPlaceState.FAILED, "place_event_timeout")
            return reference
        if state is PickPlaceState.RELEASE:
            reference = self._terminal_with_gap(self.gap_open)
            if (now-self.stage_start >= float(self.settings["settle_time"])
                    and abs(self.robot.gripper_opening-self.gap_open) <= .005):
                payload_position = self.robot.forward_kinematics(frame="grasp")[0]
                self.simulation.set_mocap_position_ned("payload_marker", payload_position)
                if np.linalg.norm(payload_position-self.place) > .02:
                    self.machine.transition(PickPlaceState.FAILED, "release_position_error")
                else:
                    self.machine.transition(PickPlaceState.DONE)
            elif now-self.stage_start > float(self.settings["event_timeout"]):
                self.machine.transition(PickPlaceState.FAILED, "gripper_open_timeout")
            return reference
        return self.command_reference

    def _settled(self, target, now):
        grasp, _ = self.robot.forward_kinematics(frame="grasp")
        speed = self.robot.jacobian(frame="grasp")[:3]@self.robot.velocity
        q = self.robot.configuration
        trajectory_name = "pick" if self.machine.state is PickPlaceState.MOVE_TO_PICK else "place"
        trajectory = self.plans[trajectory_name]
        goal = trajectory.evaluate(trajectory.total_time)
        arm_error = np.linalg.norm(q[7:11]-goal[4:8])
        arm_speed = np.linalg.norm(self.robot.velocity[6:10])
        stable = (np.linalg.norm(grasp-target) <= float(self.settings["position_tolerance"])
                  and np.linalg.norm(speed) <= float(self.settings["velocity_tolerance"])
                  and arm_error <= .05
                  and arm_speed <= .05
                  and np.linalg.norm(self.robot.velocity[3:6])
                  <= float(self.settings.get("angular_velocity_tolerance", .2)))
        self.last_settle_metrics = {
            "position_error": float(np.linalg.norm(grasp-target)),
            "end_effector_speed": float(np.linalg.norm(speed)),
            "arm_error": float(arm_error),
            "arm_speed": float(arm_speed),
            "body_rate": float(np.linalg.norm(self.robot.velocity[3:6])),
            "stable_time": float(self.stable_time),
        }
        self.stable_time = self.stable_time+self.simulation.quad.dt if stable else 0.0
        return self.stable_time >= float(self.settings["settle_time"])

    def _state_matches(self, expected, tolerance):
        actual = self.robot.configuration
        yaw_error = (quaternion_yaw(actual[3:7])-expected[3]+np.pi)%(2*np.pi)-np.pi
        return (np.linalg.norm(actual[:3]-expected[:3]) <= tolerance
                and abs(yaw_error) <= tolerance
                and np.linalg.norm(actual[7:11]-expected[4:8]) <= tolerance
                and np.linalg.norm(self.robot.velocity[:3]) <= tolerance
                and np.linalg.norm(self.robot.velocity[6:10]) <= tolerance)

    def _align_grasp_reference(self, reference, target):
        return reference

    def _terminal_with_gap(self, gap):
        trajectory_name = "pick" if self.machine.state is PickPlaceState.GRASP else "place"
        trajectory = self.plans[trajectory_name]
        reference = trajectory.reference(trajectory.total_time, self.robot,
                                         gripper_opening=gap)
        from uav_ac.robot.aerial_manipulator import AerialManipulatorReference
        configuration = reference.configuration.copy()
        configuration[11] = gap
        return AerialManipulatorReference(configuration, reference.velocity,
                                          reference.acceleration)


def _scene_esdf(simulation, config):
    bounds = simulation.space_limits
    if bounds is None:
        raise ValueError("aerial pick/place scene requires planning_bounds")
    obstacle = simulation.obstacles
    boxes = np.column_stack((obstacle[:, 0], obstacle[:, 2], obstacle[:, 4],
                             obstacle[:, 1], obstacle[:, 3], obstacle[:, 5]))
    padding = max(.6, config.obstacle_clearance+.45)
    lower, upper = bounds[0]-padding, bounds[1]+padding
    # The NED ground lies at z=0; positive signed distance is above the floor.
    return ESDF.from_axis_aligned_boxes(
        boxes, lower, upper, .08, ground_height=0.)


def _result(machine, plans, simulation, controller, execution=None):
    grasp_jacobian = simulation.robot.jacobian(frame="grasp")
    end_effector_velocity = grasp_jacobian[:3]@simulation.robot.velocity
    return {
        "success": machine.state is PickPlaceState.DONE,
        "state": machine.state.name,
        "event_sequence": machine.events.copy(),
        "failure_reason": machine.failure_reason,
        "holding_payload": machine.holding_payload,
        "payload_position": (simulation.robot.forward_kinematics(frame="grasp")[0].copy()
                             if machine.holding_payload else None),
        "final_grasp_position": simulation.robot.forward_kinematics(frame="grasp")[0].copy(),
        "final_base_position": simulation.robot.configuration[:3].copy(),
        "final_joint_positions": simulation.robot.configuration[7:11].copy(),
        "final_gripper_opening": float(simulation.robot.gripper_opening),
        "final_end_effector_speed": float(np.linalg.norm(end_effector_velocity)),
        "final_body_rate": float(np.linalg.norm(simulation.robot.velocity[3:6])),
        "pick_plan_valid": bool(plans.get("pick") and plans["pick"].validation_passed),
        "place_plan_valid": bool(plans.get("place") and plans["place"].validation_passed),
        "pick_minimum_clearance": (float(plans["pick"].minimum_clearance)
                                   if "pick" in plans else float("nan")),
        "place_minimum_clearance": (float(plans["place"].minimum_clearance)
                                    if "place" in plans else float("nan")),
        "pick_validation_sample_dt": (float(plans["pick"].validation_sample_dt)
                                      if "pick" in plans else float("nan")),
        "place_validation_sample_dt": (float(plans["place"].validation_sample_dt)
                                       if "place" in plans else float("nan")),
        "pick_maximum_violation": (float(plans["pick"].maximum_violation)
                                   if "pick" in plans else float("inf")),
        "place_maximum_violation": (float(plans["place"].maximum_violation)
                                    if "place" in plans else float("inf")),
        "pick_optimizer_converged": bool(plans.get("pick") and plans["pick"].optimizer_converged),
        "place_optimizer_converged": bool(plans.get("place") and plans["place"].optimizer_converged),
        "time": float(simulation.time),
        "collision": bool(simulation.collision_detected),
        "saturation_count": int(getattr(controller, "saturation_count", 0)) if controller else 0,
        "maximum_joint_tracking_error": (
            0.0 if execution is None else float(execution.max_joint_error)),
        "settle_metrics": {} if execution is None else execution.last_settle_metrics.copy(),
    }


def _print_result(result):
    print(f"Aerial pick/place: state={result['state']} | success={result['success']} | "
          f"pick_valid={result['pick_plan_valid']} | place_valid={result['place_plan_valid']} | "
          f"optimizer_converged={result['pick_optimizer_converged']}/"
          f"{result['place_optimizer_converged']} | "
          f"collision={'yes' if result['collision'] else 'no'}")
    if result["failure_reason"]:
        print(f"Failure: {result['failure_reason']}")
    print(f"Clearance={result['pick_minimum_clearance']:.3f}/"
          f"{result['place_minimum_clearance']:.3f}m | "
          f"validation_dt={result['pick_validation_sample_dt']:.3f}/"
          f"{result['place_validation_sample_dt']:.3f}s | "
          f"joint_error={result['maximum_joint_tracking_error']:.4f}rad | "
          f"saturations={result['saturation_count']}")


__all__ = ["PickPlaceState", "PickPlaceStateMachine", "run_aerial_pick_place"]
