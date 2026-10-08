"""Aerial-manipulator task state, execution rules and result semantics."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import quaternion_yaw
from uav_ac.utils import StateMachine


class PickPlaceState(Enum):
    PLAN_TO_PICK, MOVE_TO_PICK, GRASP = auto(), auto(), auto()
    PLAN_TO_PLACE, MOVE_TO_PLACE, RELEASE = auto(), auto(), auto()
    DONE, FAILED = auto(), auto()


_PICK_PLACE_FLOW = tuple(PickPlaceState)
_PICK_PLACE_TRANSITIONS = {
    state: {next_state, PickPlaceState.FAILED}
    for state, next_state in zip(_PICK_PLACE_FLOW[:-2], _PICK_PLACE_FLOW[1:-1])
}
_PICK_PLACE_TRANSITIONS.update({state: set() for state in _PICK_PLACE_FLOW[-2:]})


def new_pick_place_machine():
    return StateMachine(PickPlaceState.PLAN_TO_PICK, _PICK_PLACE_TRANSITIONS)


def transition_pick_place(machine, target, reason=None):
    if target is PickPlaceState.FAILED and not reason:
        raise ValueError("failed transition requires a reason")
    machine.transition(target, reason)


def holding_payload(machine):
    return PickPlaceState.PLAN_TO_PLACE in machine.history and PickPlaceState.DONE not in machine.history


@dataclass
class PickPlacePlanBundle:
    plans: dict
    pick: np.ndarray
    place: np.ndarray
    gap_open: float
    gap_closed: float
    start_q: np.ndarray
    settings: dict

    @classmethod
    def from_mapping(cls, result):
        return cls(**{key: result[key] for key in cls.__dataclass_fields__})


class PickPlaceExecution:
    _TRACE_KEYS = ("time_s", "reference_rad", "actual_rad",
                   "reference_gripper_opening_m", "actual_gripper_opening_m")

    def __init__(self, simulation, controller, bundle, machine, stride,
                 record_joint_trace=False):
        self.simulation, self.robot = simulation, simulation.robot
        self.controller, self.machine = controller, machine
        self.plans, self.settings = bundle.plans, bundle.settings
        self.pick, self.place = bundle.pick, bundle.place
        self.gap_open, self.gap_closed = bundle.gap_open, bundle.gap_closed
        self.start_q, self.stride = bundle.start_q.copy(), stride
        self.record_joint_trace = bool(record_joint_trace)
        self.physics_index, self.stage_start = 0, simulation.time
        self.stable_time, self.stable_since = 0., None
        self.max_joint_error, self.max_base_position_error = 0., 0.
        self.last_settle_metrics = {}
        self.trace, self.base_speed_samples, self.joint_speed_samples = (
            {key: [] for key in self._TRACE_KEYS}, [], [])
        self.movement_durations = []
        self.command_reference = self.plans["pick"].reference(
            0., self.robot, gripper_opening=self.gap_open)

    @property
    def done(self):
        return self.machine.state in {PickPlaceState.DONE, PickPlaceState.FAILED}

    def reset(self):
        self.robot.reset(self.start_q)
        self.controller.reset()
        self.machine = new_pick_place_machine()
        transition_pick_place(self.machine, PickPlaceState.MOVE_TO_PICK)
        for marker, position in (("pick_target_marker", self.pick),
                                 ("place_target_marker", self.place),
                                 ("payload_marker", self.pick)):
            self.simulation.set_mocap_position_ned(marker, position)
        self.physics_index, self.stage_start = 0, self.simulation.time
        self.stable_time, self.stable_since = 0., None
        self.max_joint_error = self.max_base_position_error = 0.
        self.command_reference = self.plans["pick"].reference(
            0., self.robot, gripper_opening=self.gap_open)
        for values in (*self.trace.values(), self.base_speed_samples,
                       self.joint_speed_samples, self.movement_durations):
            values.clear()

    def step(self):
        if self.done:
            return
        now = self.simulation.time
        if self.physics_index % self.stride == 0:
            self._sample_motion(now)
            self.command_reference = self._reference(now)
            self._sample_tracking(now)
        self.robot.apply(self.controller.step(self.command_reference))
        if holding_payload(self.machine):
            position = self.robot.forward_kinematics(frame="grasp")[0]
            self.simulation.set_mocap_position_ned("payload_marker", position)
        if self.simulation.collision_detected:
            transition_pick_place(self.machine, PickPlaceState.FAILED, "simulation_collision")
        self.physics_index += 1

    def _sample_motion(self, now):
        state = self.machine.state
        if state not in {PickPlaceState.MOVE_TO_PICK, PickPlaceState.MOVE_TO_PLACE}:
            return
        name = "pick" if state is PickPlaceState.MOVE_TO_PICK else "place"
        if now-self.stage_start <= self.plans[name].total_time:
            self.base_speed_samples.append(float(np.linalg.norm(self.robot.velocity[:3])))
            self.joint_speed_samples.append(np.abs(self.robot.velocity[6:10]).copy())

    def _sample_tracking(self, now):
        reference, q = self.command_reference.configuration, self.robot.configuration
        self.max_joint_error = max(self.max_joint_error, float(np.linalg.norm(reference[7:11]-q[7:11])))
        self.max_base_position_error = max(
            self.max_base_position_error, float(np.linalg.norm(reference[:3]-q[:3])))
        if self.record_joint_trace:
            for key, sample in zip(self._TRACE_KEYS, (
                    float(now), reference[7:11].copy(), q[7:11].copy(),
                    float(reference[11]), float(self.robot.gripper_opening))):
                self.trace[key].append(sample)

    def _reference(self, now):
        state = self.machine.state
        if state in {PickPlaceState.MOVE_TO_PICK, PickPlaceState.MOVE_TO_PLACE}:
            return self._travel_reference(state, now)
        if state in {PickPlaceState.GRASP, PickPlaceState.RELEASE}:
            return self._gripper_reference(state, now)
        return self.command_reference

    def _travel_reference(self, state, now):
        is_pick = state is PickPlaceState.MOVE_TO_PICK
        name, target = ("pick", self.pick) if is_pick else ("place", self.place)
        gap = self.gap_open if is_pick else self.gap_closed
        trajectory = self.plans[name]
        elapsed = now-self.stage_start
        reference = trajectory.reference(
            min(elapsed, trajectory.total_time), self.robot, gripper_opening=gap)
        if elapsed >= trajectory.total_time and self._settled(target, now):
            self.movement_durations.append(min(elapsed, trajectory.total_time))
            transition_pick_place(self.machine, PickPlaceState.GRASP if is_pick else PickPlaceState.RELEASE)
            self.stage_start, self.stable_since = now, None
        return reference

    def _gripper_reference(self, state, now):
        grasping = state is PickPlaceState.GRASP
        gap = self.gap_closed if grasping else self.gap_open
        reference = self._terminal_with_gap(gap)
        elapsed = now-self.stage_start
        settled = (elapsed >= float(self.settings["settle_time"])
                   and abs(self.robot.gripper_opening-gap) <= .005)
        if settled and grasping:
            goal = self.plans["pick"].evaluate(self.plans["pick"].total_time)
            if not self._state_matches(goal, .06):
                transition_pick_place(self.machine, PickPlaceState.FAILED, "carry_start_state_mismatch")
                return reference
            transition_pick_place(self.machine, PickPlaceState.PLAN_TO_PLACE)
            transition_pick_place(self.machine, PickPlaceState.MOVE_TO_PLACE)
            self.stage_start = now
        elif settled:
            position = self.robot.forward_kinematics(frame="grasp")[0]
            self.simulation.set_mocap_position_ned("payload_marker", position)
            if np.linalg.norm(position-self.place) > .02:
                transition_pick_place(self.machine, PickPlaceState.FAILED, "release_position_error")
            else:
                transition_pick_place(self.machine, PickPlaceState.DONE)
        return reference

    def _settled(self, target, now):
        grasp, _ = self.robot.forward_kinematics(frame="grasp")
        speed = self.robot.jacobian(frame="grasp")[:3]@self.robot.velocity
        q = self.robot.configuration
        name = "pick" if self.machine.state is PickPlaceState.MOVE_TO_PICK else "place"
        goal = self.plans[name].evaluate(self.plans[name].total_time)
        position_error, ee_speed = np.linalg.norm(grasp-target), np.linalg.norm(speed)
        arm_error = np.linalg.norm(q[7:11]-goal[4:8])
        arm_speed, body_rate = np.linalg.norm(self.robot.velocity[6:10]), np.linalg.norm(
            self.robot.velocity[3:6])
        stable = (position_error <= float(self.settings["position_tolerance"])
                  and ee_speed <= float(self.settings["velocity_tolerance"])
                  and arm_error <= .05 and arm_speed <= .05
                  and body_rate <= float(self.settings.get("angular_velocity_tolerance", .2)))
        self.last_settle_metrics = dict(position_error=float(position_error),
            end_effector_speed=float(ee_speed), arm_error=float(arm_error),
            arm_speed=float(arm_speed), body_rate=float(body_rate), stable_time=float(self.stable_time))
        if stable:
            self.stable_since = now if self.stable_since is None else self.stable_since
            self.stable_time = max(0., now-self.stable_since)
        else:
            self.stable_since, self.stable_time = None, 0.
        return self.stable_time >= float(self.settings["settle_time"])

    def _state_matches(self, expected, tolerance):
        actual = self.robot.configuration
        yaw_error = (quaternion_yaw(actual[3:7])-expected[3]+np.pi)%(2*np.pi)-np.pi
        return (np.linalg.norm(actual[:3]-expected[:3]) <= tolerance and abs(yaw_error) <= tolerance
                and np.linalg.norm(actual[7:11]-expected[4:8]) <= tolerance
                and np.linalg.norm(self.robot.velocity[:3]) <= tolerance
                and np.linalg.norm(self.robot.velocity[6:10]) <= tolerance)

    def _terminal_with_gap(self, gap):
        name = "pick" if self.machine.state is PickPlaceState.GRASP else "place"
        trajectory = self.plans[name]
        reference = trajectory.reference(trajectory.total_time, self.robot, gripper_opening=gap)
        from uav_ac.robot.aerial_manipulator import AerialManipulatorReference
        configuration = reference.configuration.copy()
        configuration[11] = gap
        return AerialManipulatorReference(configuration, reference.velocity, reference.acceleration)

    def joint_trace(self):
        return {key: np.asarray(self.trace[key], dtype=float).reshape(-1, 4)
                if key.endswith("_rad") else np.asarray(self.trace[key], dtype=float)
                for key in self._TRACE_KEYS}


def _plan_result_fields(plans):
    fields = {}
    specs = (("plan_valid", "validation_passed", False, bool),
             ("minimum_clearance", "minimum_clearance", float("nan"), float),
             ("validation_sample_dt", "validation_sample_dt", float("nan"), float),
             ("maximum_violation", "maximum_violation", float("inf"), float),
             ("optimizer_converged", "optimizer_converged", False, bool))
    for suffix, attribute, missing, convert in specs:
        for name in ("pick", "place"):
            plan = plans.get(name)
            fields[f"{name}_{suffix}"] = convert(getattr(plan, attribute)) if plan else missing
    return fields


def pick_place_result(machine, plans, simulation, controller, execution=None, planning_metrics=None):
    robot = simulation.robot
    payload = simulation.get_mocap_position_ned("payload_marker")
    held = holding_payload(machine)
    grasp = robot.forward_kinematics(frame="grasp")[0]
    ee_velocity = robot.jacobian(frame="grasp")[:3]@robot.velocity
    result = dict(
        success=machine.state is PickPlaceState.DONE, state=machine.state.name,
        event_sequence=[state.name for state in machine.history], failure_reason=machine.last_reason,
        holding_payload=held, payload_position=grasp.copy() if held else None,
        released_payload_position_ned=payload.copy(),
        place_target_position_ned=execution.place.copy() if execution else None,
        gripper_opening_target=float(execution.gap_open) if execution else None,
        released_payload_error=float(np.linalg.norm(payload-execution.place)) if execution else float("nan"),
        final_grasp_position=grasp.copy(), final_base_position=robot.configuration[:3].copy(),
        final_joint_positions=robot.configuration[7:11].copy(),
        final_gripper_opening=float(robot.gripper_opening), final_end_effector_speed=float(np.linalg.norm(ee_velocity)),
        final_body_rate=float(np.linalg.norm(robot.velocity[3:6])), time=float(simulation.time),
        collision=bool(simulation.collision_detected),
        saturation_count=int(getattr(controller, "saturation_count", 0)) if controller else 0,
        planning_metrics={} if planning_metrics is None else dict(planning_metrics))
    result.update(_plan_result_fields(plans))
    base_speeds = execution.base_speed_samples if execution else []
    joint_speeds = execution.joint_speed_samples if execution else []
    result.update(
        maximum_joint_tracking_error=float(execution.max_joint_error) if execution else 0.,
        maximum_base_position_tracking_error_m=float(execution.max_base_position_error) if execution else 0.,
        joint_execution_trace=execution.joint_trace() if execution else {},
        settle_metrics=execution.last_settle_metrics.copy() if execution else {},
        execution_average_base_speed_mps=float(np.mean(base_speeds)) if base_speeds else 0.,
        execution_average_joint_speed_rad_s=np.mean(joint_speeds, axis=0).tolist()
        if joint_speeds else [0.]*4,
        execution_movement_time_s=float(sum(execution.movement_durations)) if execution else 0.)
    return result


__all__ = ["PickPlaceState", "StateMachine", "PickPlacePlanBundle",
           "PickPlaceExecution", "new_pick_place_machine", "transition_pick_place",
           "pick_place_result"]
