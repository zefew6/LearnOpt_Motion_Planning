"""Deterministic aerial-manipulator pick, carry and place simulation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import time
from typing import ClassVar

import numpy as np

from uav_ac.control import CascadedConfig, CascadedController
from uav_ac.control.aerial_manipulator_controller import AerialManipulatorController
from uav_ac.planning.geometry.esdf import ESDF, InflatedOccupancyGrid
from uav_ac.planning.trajectory.aerial_manipulator_minco import (
    AerialManipulatorMINCO, AerialManipulatorMINCOConfig, make_terminal_state,
)
from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import quaternion_yaw
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
    planner_config = AerialManipulatorMINCOConfig.from_mapping(
        config["aerial_manipulator_minco"])
    machine = PickPlaceStateMachine()
    diagnostics = {}
    planning_started = time.perf_counter()
    try:
        bundle = plan_pick_place(
            simulation, config, deadline=planning_started+planner_config.planning_budget_s,
            diagnostics=diagnostics)
        plans, pick, place = bundle["plans"], bundle["pick"], bundle["place"]
        gap_open, gap_closed = bundle["gap_open"], bundle["gap_closed"]
        start_q = bundle["start_q"]
        machine.transition(PickPlaceState.MOVE_TO_PICK)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, TimeoutError) as error:
        diagnostics.setdefault("planning_seconds", time.perf_counter()-planning_started)
        if machine.state is not PickPlaceState.FAILED:
            machine.transition(PickPlaceState.FAILED, str(error))
        result = _result(machine, diagnostics.get("plans", {}), simulation,
                         controller=None, planning_metrics=_public_metrics(diagnostics))
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
    result = _result(machine, plans, simulation, controller, execution,
                     planning_metrics=_public_metrics(diagnostics))
    _print_result(result)
    return result


def _public_metrics(diagnostics):
    return {key: value for key, value in diagnostics.items() if key != "plans"}


def plan_pick_place(simulation, config, *, deadline=None, seed=None, diagnostics=None,
                    search_only=False):
    """Plan and validate both pick/place legs using the production code path.

    This entry point supports headless planning benchmarks without caching maps,
    routes, or optimized trajectories between calls.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    started = time.perf_counter()
    planner_config = AerialManipulatorMINCOConfig.from_mapping(
        config["aerial_manipulator_minco"])
    deadline = (started+planner_config.planning_budget_s if deadline is None else deadline)
    settings = config["pick_place"]
    robot = simulation.robot
    pick = np.asarray(settings["pick_position_ned"], dtype=float)
    place = np.asarray(settings["place_position_ned"], dtype=float)
    gap_open, gap_closed = float(settings["gripper_open"]), float(settings["gripper_closed"])
    bounds = simulation.space_limits
    phase = time.perf_counter()
    occupancy = _scene_occupancy(simulation, planner_config)
    diagnostics["occupancy_seconds"] = time.perf_counter()-phase
    diagnostics["esdf_grid_shape"] = list(occupancy.occupied.shape)
    diagnostics["esdf_grid_voxels"] = int(occupancy.occupied.size)
    diagnostics["esdf_grid_bounds"] = [
        occupancy.origin.tolist(), occupancy.upper.tolist()]
    diagnostics["esdf_resolution"] = float(occupancy.resolution)
    phase = time.perf_counter()
    esdf = ESDF.from_occupancy(occupancy)
    diagnostics["esdf_seconds"] = time.perf_counter()-phase
    if time.perf_counter() >= deadline:
        raise TimeoutError("aerial_manipulator_minco planning budget exceeded during ESDF build")
    start_q = robot.configuration.copy()
    start = np.r_[start_q[:3], quaternion_yaw(start_q[3:7]), start_q[7:11]]
    pick_state = make_terminal_state(
        robot, pick, float(settings["pick_yaw"]),
        np.asarray(settings["pick_nominal_joints"], dtype=float),
        gripper_opening=gap_open, workspace_bounds=bounds)
    place_state = make_terminal_state(
        robot, place, float(settings["place_yaw"]),
        np.asarray(settings["place_nominal_joints"], dtype=float),
        gripper_opening=gap_closed, workspace_bounds=bounds)
    planner = AerialManipulatorMINCO(planner_config)
    rng = np.random.default_rng(int(config["seed"] if seed is None else seed))
    plans = {}
    searches = {}
    leg_metrics = {}
    for name, leg_start, leg_goal, opening, carry in (
            ("pick", start, pick_state, gap_open, False),
            ("place", pick_state, place_state, gap_closed, True)):
        if not search_only and time.perf_counter() >= deadline:
            raise TimeoutError("aerial_manipulator_minco planning budget exceeded")
        leg_deadline = (time.perf_counter()+planner_config.planning_budget_s
                        if search_only else deadline)
        try:
            if search_only:
                searches[name] = planner.search_initial_path(
                    leg_start, leg_goal, robot=robot, esdf=esdf, quad=simulation.quad,
                    workspace_bounds=bounds, gripper_opening=opening,
                    carry_payload=carry, rng=rng, deadline=leg_deadline,
                    occupancy=occupancy)
                leg_metrics[name] = dict(searches[name].metrics)
                leg_metrics[name]["exact_solution"] = searches[name].exact_solution
                leg_metrics[name]["path_states"] = int(len(searches[name].path))
                diagnostics["legs"] = dict(leg_metrics)
            else:
                plans[name] = planner.plan(
                    leg_start, leg_goal, robot=robot, esdf=esdf, quad=simulation.quad,
                    workspace_bounds=bounds, gripper_opening=opening,
                    carry_payload=carry, rng=rng, deadline=leg_deadline,
                    occupancy=occupancy)
        except (ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError) as error:
            leg_metrics[name] = dict(planner.last_metrics)
            leg_metrics[name]["failure_reason"] = str(error)
            diagnostics["legs"] = leg_metrics
            diagnostics["plans"] = plans
            diagnostics["searches"] = searches
            if search_only:
                continue
            diagnostics["planning_seconds"] = time.perf_counter()-started
            raise
        if search_only:
            continue
        leg_metrics[name] = dict(planner.last_metrics)
        leg_metrics[name].update({
            "validation_passed": bool(plans[name].validation_passed),
            "optimizer_converged": bool(plans[name].optimizer_converged),
            "optimizer_iterations": int(plans[name].iterations),
            "maximum_violation": float(plans[name].maximum_violation),
            "minimum_clearance": float(plans[name].minimum_clearance),
            "validation_sample_dt": float(plans[name].validation_sample_dt),
        })
        joint_samples = plans[name].evaluate(
            np.linspace(0., plans[name].total_time, 101))[:, 4:8]
        midpoint_joints = plans[name].evaluate(
            .5*plans[name].total_time)[4:8]
        leg_metrics[name].update({
            "start_arm_joints": leg_start[4:8].tolist(),
            "target_arm_joints": leg_goal[4:8].tolist(),
            "midpoint_arm_joints": midpoint_joints.tolist(),
            "maximum_arm_deviation_rad": float(np.max(np.linalg.norm(
                joint_samples-leg_start[None, 4:8], axis=1))),
            "joint_peak_to_peak_rad": np.ptp(joint_samples, axis=0).tolist(),
            "joint_trajectory_rad": joint_samples.tolist(),
            "rrt_intermediate_arm_deviation_rad": float(np.max(np.linalg.norm(
                plans[name].rrt_path[1:-1, 4:8]-leg_start[None, 4:8], axis=1),
                initial=0.)),
        })
        diagnostics["plans"] = dict(plans)
        diagnostics["legs"] = dict(leg_metrics)
        if not plans[name].validation_passed:
            raise RuntimeError(f"{name} trajectory failed final validation")
    if search_only:
        diagnostics["searches"] = searches
        diagnostics["legs"] = leg_metrics
        diagnostics["planning_seconds"] = time.perf_counter()-started
        return {"searches": searches, "pick": pick, "place": place,
                "gap_open": gap_open, "gap_closed": gap_closed,
                "start_q": start_q, "esdf": esdf}
    diagnostics["plans"] = plans
    diagnostics["legs"] = leg_metrics
    diagnostics["planning_seconds"] = time.perf_counter()-started
    return {"plans": plans, "pick": pick, "place": place,
            "gap_open": gap_open, "gap_closed": gap_closed,
            "start_q": start_q, "esdf": esdf}


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
        self.stable_since = None
        self.failure_time = simulation.time
        self.command_reference = plans["pick"].reference(
            0., self.robot, gripper_opening=gap_open)
        self.max_joint_error = 0.0
        self.last_settle_metrics = {}
        self.joint_trace_time = []
        self.reference_joint_trace = []
        self.actual_joint_trace = []
        self.reference_gripper_trace = []
        self.actual_gripper_trace = []

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
        self.stable_since = None
        self.command_reference = self.plans["pick"].reference(
            0., self.robot, gripper_opening=self.gap_open)
        self.joint_trace_time.clear()
        self.reference_joint_trace.clear()
        self.actual_joint_trace.clear()
        self.reference_gripper_trace.clear()
        self.actual_gripper_trace.clear()

    def step(self):
        if self.done:
            return
        now = self.simulation.time
        if self.physics_index % self.stride == 0:
            self.command_reference = self._reference(now)
            error = np.linalg.norm(
                self.command_reference.configuration[7:11]-self.robot.configuration[7:11])
            self.max_joint_error = max(self.max_joint_error, float(error))
            self.joint_trace_time.append(float(now))
            self.reference_joint_trace.append(
                self.command_reference.configuration[7:11].copy())
            self.actual_joint_trace.append(self.robot.configuration[7:11].copy())
            self.reference_gripper_trace.append(float(self.command_reference.configuration[11]))
            self.actual_gripper_trace.append(float(self.robot.gripper_opening))
        self.robot.apply(self.controller.step(self.command_reference))
        if self.machine.holding_payload:
            payload_position = self.robot.forward_kinematics(frame="grasp")[0]
            self.simulation.set_mocap_position_ned("payload_marker", payload_position)
            payload_collision = self.robot.check_collision(
                self.robot.configuration, clearance=0.,
                payload_position_ned=payload_position,
                payload_radius=.035)
            if payload_collision["collision"]:
                self.machine.transition(PickPlaceState.FAILED, "payload_collision")
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
            if elapsed >= trajectory.total_time and self._settled(self.pick, now):
                self.machine.transition(PickPlaceState.GRASP)
                self.stage_start = now
                self.stable_since = None
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
            if elapsed >= trajectory.total_time and self._settled(self.place, now):
                self.machine.transition(PickPlaceState.RELEASE)
                self.stage_start = now
                self.stable_since = None
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
        if stable:
            self.stable_since = now if self.stable_since is None else self.stable_since
            self.stable_time = max(0.0, now-self.stable_since)
        else:
            self.stable_since = None
            self.stable_time = 0.0
        return self.stable_time >= float(self.settings["settle_time"])

    def _state_matches(self, expected, tolerance):
        actual = self.robot.configuration
        yaw_error = (quaternion_yaw(actual[3:7])-expected[3]+np.pi)%(2*np.pi)-np.pi
        return (np.linalg.norm(actual[:3]-expected[:3]) <= tolerance
                and abs(yaw_error) <= tolerance
                and np.linalg.norm(actual[7:11]-expected[4:8]) <= tolerance
                and np.linalg.norm(self.robot.velocity[:3]) <= tolerance
                and np.linalg.norm(self.robot.velocity[6:10]) <= tolerance)

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

    def joint_trace(self):
        """Return controller-rate planned/actual arm traces for result inspection."""
        return {
            "time_s": np.asarray(self.joint_trace_time, dtype=float),
            "reference_rad": np.asarray(self.reference_joint_trace, dtype=float).reshape(-1, 4),
            "actual_rad": np.asarray(self.actual_joint_trace, dtype=float).reshape(-1, 4),
            "reference_gripper_opening_m": np.asarray(self.reference_gripper_trace, dtype=float),
            "actual_gripper_opening_m": np.asarray(self.actual_gripper_trace, dtype=float),
        }


def _scene_occupancy(simulation, config):
    """Build the shared RRT occupancy and ESDF source map once per mission."""
    boxes, lower, upper = _scene_geometry(simulation, config)
    return InflatedOccupancyGrid.from_axis_aligned_boxes(
        boxes, lower, upper, config.esdf_resolution, ground_height=0.)


def _scene_geometry(simulation, config):
    bounds = simulation.space_limits
    if bounds is None:
        raise ValueError("aerial pick/place scene requires planning_bounds")
    obstacle = simulation.obstacles
    boxes = np.column_stack((obstacle[:, 0], obstacle[:, 2], obstacle[:, 4],
                             obstacle[:, 1], obstacle[:, 3], obstacle[:, 5]))
    padding = max(.8, config.obstacle_clearance+.65)
    return boxes, bounds[0]-padding, bounds[1]+padding


def _result(machine, plans, simulation, controller, execution=None, planning_metrics=None):
    grasp_jacobian = simulation.robot.jacobian(frame="grasp")
    end_effector_velocity = grasp_jacobian[:3]@simulation.robot.velocity
    payload_position = simulation.get_mocap_position_ned("payload_marker")
    payload_error = (float(np.linalg.norm(payload_position-execution.place))
                     if execution is not None else float("nan"))
    return {
        "success": machine.state is PickPlaceState.DONE,
        "state": machine.state.name,
        "event_sequence": machine.events.copy(),
        "failure_reason": machine.failure_reason,
        "holding_payload": machine.holding_payload,
        "payload_position": (simulation.robot.forward_kinematics(frame="grasp")[0].copy()
                             if machine.holding_payload else None),
        "released_payload_position_ned": payload_position.copy(),
        "released_payload_error": payload_error,
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
        "joint_execution_trace": ({} if execution is None else execution.joint_trace()),
        "settle_metrics": {} if execution is None else execution.last_settle_metrics.copy(),
        "planning_metrics": {} if planning_metrics is None else dict(planning_metrics),
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
          f"released_payload_error={result['released_payload_error']:.3f}m | "
          f"validation_dt={result['pick_validation_sample_dt']:.3f}/"
          f"{result['place_validation_sample_dt']:.3f}s | "
          f"joint_error={result['maximum_joint_tracking_error']:.4f}rad | "
          f"saturations={result['saturation_count']}")
    metrics = result.get("planning_metrics", {})
    if metrics:
        print(f"Planning={metrics.get('planning_seconds', float('nan')):.3f}s | "
              f"ESDF={metrics.get('esdf_seconds', float('nan')):.3f}s | "
              f"AB/BC optimizer calls="
              f"{metrics.get('legs', {}).get('pick', {}).get('objective_calls', 0)}/"
              f"{metrics.get('legs', {}).get('place', {}).get('objective_calls', 0)}")


__all__ = ["PickPlaceState", "PickPlaceStateMachine", "plan_pick_place",
           "run_aerial_pick_place"]
