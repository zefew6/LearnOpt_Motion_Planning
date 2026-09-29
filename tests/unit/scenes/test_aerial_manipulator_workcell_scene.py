import numpy as np

from uav_ac.main import load_config
from uav_ac.planning.trajectory.aerial_manipulator_minco import make_terminal_state
from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import yaw_quaternion
from uav_ac.simulation.mujoco_sim import MujocoSimulation


CONFIG = "configs/aerial_manipulator_workcell.yaml"


def _configuration(state, gap):
    return np.r_[state[:3], yaw_quaternion(state[3]), state[4:], gap]


def test_workcell_loads_exact_layout_and_terminal_targets_are_valid():
    config = load_config(CONFIG)
    simulation = MujocoSimulation(config["scene"], record_actual_trajectory=False)
    np.testing.assert_allclose(
        simulation.space_limits,
        [[-.6, -2.4, -2.05], [5.8, 2.4, -.85]])
    assert simulation.obstacles.shape == (20, 6)
    np.testing.assert_allclose(simulation.robot.configuration[:3], [0., -1.4, -1.45])

    settings = config["pick_place"]
    pick = make_terminal_state(
        simulation.robot, settings["pick_position_ned"], 0., np.zeros(4),
        gripper_opening=settings["gripper_open"],
        workspace_bounds=simulation.space_limits)
    place = make_terminal_state(
        simulation.robot, settings["place_position_ned"], 0., np.zeros(4),
        gripper_opening=settings["gripper_closed"],
        workspace_bounds=simulation.space_limits)
    assert not simulation.robot.check_collision(
        _configuration(pick, settings["gripper_open"]),
        clearance=.05, self_clearance=.015)["collision"]
    assert not simulation.robot.check_collision(
        _configuration(place, settings["gripper_closed"]),
        clearance=.05, self_clearance=.015,
        payload_attached=True)["collision"]


def test_transfer_opening_needs_folding_but_has_a_broad_valid_neighborhood():
    config = load_config(CONFIG)
    simulation = MujocoSimulation(config["scene"], record_actual_trajectory=False)
    robot = simulation.robot

    extended = np.r_[[3.9, 0., -1.4], yaw_quaternion(0.), np.zeros(4), .025]
    assert robot.check_collision(
        extended, clearance=.05, self_clearance=.015,
        payload_attached=True)["collision"]
    folded = extended.copy()
    folded[8] = -1.1
    assert not robot.check_collision(
        folded, clearance=.05, self_clearance=.015,
        payload_attached=True)["collision"]

    rng = np.random.default_rng(2026)
    valid = 0
    for _ in range(500):
        joints = np.array([0., -1.1, 0., 0.]) + rng.uniform(-.2, .2, 4)
        height = rng.uniform(1.40, 1.46)
        lateral = rng.uniform(-.5, .5)
        yaw = rng.uniform(-.4, .4)
        state = np.r_[[3.9, lateral, -height], yaw_quaternion(yaw), joints, .025]
        valid += not robot.check_collision(
            state, clearance=.05, self_clearance=.015,
            payload_attached=True)["collision"]
    assert valid >= 400


def test_loaded_rack_has_clear_routes_on_both_sides():
    config = load_config(CONFIG)
    simulation = MujocoSimulation(config["scene"], record_actual_trajectory=False)
    for lateral in (-1.1, 1.1):
        state = np.r_[[2.675, lateral, -1.2], yaw_quaternion(0.), np.zeros(4), .06]
        assert not simulation.robot.check_collision(
            state, clearance=.05, self_clearance=.015)["collision"]
