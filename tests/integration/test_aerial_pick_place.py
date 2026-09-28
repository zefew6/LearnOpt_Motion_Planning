import numpy as np
import pytest

from uav_ac.main import load_config, run
from uav_ac.planning.trajectory.aerial_manipulator_minco import make_terminal_state
from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import yaw_quaternion
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_bookshelf_pick_requires_l_shaped_arm_configuration():
    config = load_config("configs/aerial_manipulator_pick_place.yaml")
    simulation = MujocoSimulation(config["scene"], record_actual_trajectory=False)
    settings = config["pick_place"]
    target = np.asarray(settings["pick_position_ned"])
    vertical = make_terminal_state(
        simulation.robot, target, settings["pick_yaw"], np.zeros(4),
        gripper_opening=settings["gripper_open"],
        workspace_bounds=simulation.space_limits)
    l_shaped = make_terminal_state(
        simulation.robot, target, settings["pick_yaw"],
        np.asarray(settings["pick_nominal_joints"]),
        gripper_opening=settings["gripper_open"],
        workspace_bounds=simulation.space_limits)

    def full_configuration(state):
        return np.r_[state[:3], yaw_quaternion(state[3]), state[4:],
                     settings["gripper_open"]]

    assert simulation.robot.check_collision(full_configuration(vertical))["collision"]
    assert not simulation.robot.check_collision(full_configuration(l_shaped))["collision"]


@pytest.mark.parametrize("seed", (0, 7, 9))
def test_aerial_manipulator_pick_place_plans_and_executes_headless(seed):
    config = load_config("configs/aerial_manipulator_pick_place.yaml")
    config["seed"] = seed
    config["visualize"] = False
    result = run(config)

    assert result["success"]
    assert result["state"] == "DONE"
    assert result["event_sequence"] == [
        "PLAN_TO_PICK", "MOVE_TO_PICK", "GRASP", "PLAN_TO_PLACE",
        "MOVE_TO_PLACE", "RELEASE", "DONE",
    ]
    assert result["pick_plan_valid"] and result["place_plan_valid"]
    assert 0 < result["pick_validation_sample_dt"] <= .025
    assert 0 < result["place_validation_sample_dt"] <= .025
    assert result["planning_metrics"]["planning_seconds"] <= 5.0
    assert result["planning_metrics"]["legs"]["pick"]["rrt_nodes"] > 2
    assert result["planning_metrics"]["legs"]["place"]["rrt_nodes"] > 2
    pick_metrics = result["planning_metrics"]["legs"]["pick"]
    place_metrics = result["planning_metrics"]["legs"]["place"]
    assert np.allclose(pick_metrics["start_arm_joints"], np.zeros(4))
    assert np.allclose(pick_metrics["target_arm_joints"],
                       config["pick_place"]["pick_nominal_joints"])
    assert pick_metrics["rrt_intermediate_arm_deviation_rad"] > 1.0
    assert pick_metrics["maximum_arm_deviation_rad"] > 0.5
    assert np.allclose(place_metrics["start_arm_joints"],
                       place_metrics["target_arm_joints"], atol=1e-8)
    assert place_metrics["maximum_arm_deviation_rad"] < 0.3
    assert np.linalg.norm(result["final_grasp_position"]-
                          config["pick_place"]["place_position_ned"]) < .02
    assert result["released_payload_error"] <= .02
    assert np.linalg.norm(result["released_payload_position_ned"]-
                          config["pick_place"]["place_position_ned"]) <= .02
    assert abs(result["final_gripper_opening"]-
               config["pick_place"]["gripper_open"]) < .005
    assert not result["collision"]
