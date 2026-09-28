import numpy as np

from uav_ac.main import load_config, run


def test_aerial_manipulator_pick_place_plans_and_executes_headless():
    config = load_config("configs/aerial_manipulator_pick_place.yaml")
    config["aerial_manipulator_gcopter"].update(
        pieces=2, max_iterations=4, integral_resolution=3, rrt_max_iterations=300)
    result = run(config)

    assert result["success"]
    assert result["state"] == "DONE"
    assert result["event_sequence"] == [
        "PLAN_TO_PICK", "MOVE_TO_PICK", "GRASP", "PLAN_TO_PLACE",
        "MOVE_TO_PLACE", "RELEASE", "DONE",
    ]
    assert result["pick_plan_valid"] and result["place_plan_valid"]
    assert result["pick_validation_sample_dt"] == .025
    assert result["place_validation_sample_dt"] == .025
    assert np.linalg.norm(result["final_grasp_position"]-
                          config["pick_place"]["place_position_ned"]) < .02
    assert abs(result["final_gripper_opening"]-
               config["pick_place"]["gripper_open"]) < .005
    assert not result["collision"]
