"""Initialize and run an aerial-manipulator pick/place flight."""

import time

import numpy as np

from uav_ac.control import CascadedConfig, CascadedController
from uav_ac.control.aerial_manipulator_controller import AerialManipulatorController
from uav_ac.planning.pipeline.aerial_pick_place import PickPlacePlanner, pick_place_settings
from uav_ac.planning.trajectory.aerial_manipulator_minco import AerialManipulatorMINCOConfig
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.tasks.aerial_pick_place import (
    PickPlaceExecution, PickPlacePlanBundle, PickPlaceState,
    new_pick_place_machine, transition_pick_place, pick_place_result,
)


def run_aerial_pick_place(config):
    """Solve the joint task before moving, then execute its confirmed events."""
    simulation = MujocoSimulation(
        config["scene"], record_actual_trajectory=False,
        planning_path_capacity=2 if config["visualize"] else 0)
    settings = pick_place_settings(simulation, config)
    planner_config = AerialManipulatorMINCOConfig.from_mapping(
        config["aerial_manipulator_minco"])
    machine, diagnostics = new_pick_place_machine(), {}
    runtime = {"bundle": None, "controller": None, "execution": None}
    rrt_paths, minco_paths = {}, {}

    initialization_started = time.perf_counter()
    try:
        planner = PickPlacePlanner(simulation, config, settings=settings,
                                   planner_config=planner_config)
        diagnostics.update(planner.initialization_metrics)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        diagnostics["planning_seconds"] = time.perf_counter()-initialization_started
        transition_pick_place(machine, PickPlaceState.FAILED, str(error))
        result = pick_place_result(machine, {}, simulation, None,
                                   planning_metrics=diagnostics)
        _print_result(result)
        return result


    for marker, position in (("pick_target_marker", settings["pick_position_ned"]),
                             ("place_target_marker", settings["place_position_ned"]),
                             ("payload_marker", settings["pick_position_ned"])):
        simulation.set_mocap_position_ned(marker, position)

    def joined_path(paths):
        segments = [paths[name] for name in ("pick", "place") if name in paths]
        if not segments:
            return None
        return np.vstack([segments[0], *(segment[1:] for segment in segments[1:])])

    def prepare(sync_viewer):
        def refresh_paths():
            paths, colors, dashed = [], [], []
            for points, color, is_dashed in (
                    (joined_path(rrt_paths), (0.0, 0.55, 0.85, 0.95), True),
                    (joined_path(minco_paths), (0.1, 0.8, 0.25, 0.95), False)):
                if points is not None:
                    paths.append(points)
                    colors.append(color)
            simulation.set_planning_paths(paths, colors)
            sync_viewer()

        def on_rrt_path(leg, states):
            rrt_paths[leg] = np.asarray(states, dtype=float)[:, :3].copy()
            refresh_paths()

        def on_minco_trajectory(leg, trajectory):
            sample_count = max(2, int(np.ceil(trajectory.total_time / .05)) + 1)
            times = np.linspace(0.0, trajectory.total_time, sample_count)
            minco_paths[leg] = trajectory.evaluate(times)[:, :3]
            refresh_paths()

        planning_started = time.perf_counter()
        try:
            planned = planner.plan(
                diagnostics=diagnostics,
                on_rrt_path=on_rrt_path if config["visualize"] else None,
                on_minco_trajectory=on_minco_trajectory if config["visualize"] else None)
            bundle = PickPlacePlanBundle.from_mapping(planned)
            transition_pick_place(machine, PickPlaceState.MOVE_TO_PICK)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            diagnostics.setdefault(
                "planning_seconds", time.perf_counter()-planning_started
                +planner.initialization_seconds)
            if machine.state is not PickPlaceState.FAILED:
                transition_pick_place(machine, PickPlaceState.FAILED, str(error))
            return False

        control_dt = float(config["control_dt"])
        stride = int(round(control_dt/simulation.quad.dt))
        if stride < 1 or not np.isclose(stride*simulation.quad.dt, control_dt):
            raise ValueError("control_dt must be an integer multiple of the XML timestep")
        cascaded = config.get("cascaded", {})
        if cascaded:
            CascadedConfig(**cascaded).apply_to(simulation.quad)
        flight = CascadedController(simulation.quad.g, control_dt)
        controller = AerialManipulatorController(flight, simulation.robot, simulation.quad)
        execution = PickPlaceExecution(simulation, controller, bundle, machine, stride,
                               bool(settings.get("record_joint_trace", False)))
        runtime.update(bundle=bundle, controller=controller, execution=execution)
        return True

    def step():
        if runtime["execution"] is not None:
            runtime["execution"].step()

    def reset():
        if runtime["execution"] is not None:
            runtime["execution"].reset()

    if config["visualize"]:
        prepared = simulation.run_interactive(
            step, reset, chase_camera=config["follow_camera"], prepare=prepare)
    else:
        prepared = prepare(lambda: None)

    if not prepared:
        bundle = runtime["bundle"]
        result = pick_place_result(
            machine, bundle.plans if bundle is not None else diagnostics.get("plans", {}),
            simulation, runtime["controller"], runtime["execution"],
            planning_metrics=_public_metrics(diagnostics))
        _print_result(result)
        return result

    bundle = runtime["bundle"]
    controller = runtime["controller"]
    execution = runtime["execution"]
    if not config["visualize"]:
        while not execution.done:
            execution.step()
            simulation.step()
    result = pick_place_result(machine, bundle.plans, simulation, controller, execution,
                     _public_metrics(diagnostics))
    _print_result(result)
    return result


def _public_metrics(diagnostics):
    return {key: value for key, value in diagnostics.items() if key != "plans"}


def _print_result(result):
    seconds = result["planning_metrics"].get("planning_seconds")
    planning_time = f"{seconds:.2f}s" if seconds is not None and np.isfinite(seconds) else "n/a"
    pick_status = "unvalidated" if result['pick_plan_valid'] is None else str(result['pick_plan_valid'])
    place_status = "unvalidated" if result['place_plan_valid'] is None else str(result['place_plan_valid'])
    print(f"Aerial pick/place: state={result['state']} | success={'yes' if result['success'] else 'no'} | "
          f"collision={'yes' if result['collision'] else 'no'} | plans={pick_status}/"
          f"{place_status} | planning={planning_time}")
    if result["failure_reason"]:
        print(f"Failure: {result['failure_reason']}")
