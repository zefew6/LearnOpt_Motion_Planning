import numpy as np
import pytest

from uav_ac import main
from uav_ac.main import load_config, run


def test_aerial_manipulator_task_plans_and_executes_headless():
    config = load_config("configs/aerial_manipulator_workcell.yaml")
    config["visualize"] = False
    config["pick_place"]["record_joint_trace"] = True
    result = run(config)

    assert result["success"]
    assert result["state"] == "DONE"
    assert result["event_sequence"] == [
        "PLAN_TASK", "MOVE_TO_PICK", "GRASP",
        "MOVE_TO_PLACE", "RELEASE", "DONE",
    ]
    assert result["pick_plan_valid"] is None and result["place_plan_valid"] is None
    assert not result["pick_validation_performed"] and not result["place_validation_performed"]
    assert result['planning_metrics']['joint_problem']
    assert result['planning_metrics']['task_equality_feasible']
    assert np.isnan(result["pick_validation_sample_dt"])
    assert np.isnan(result["place_validation_sample_dt"])
    assert result["planning_metrics"]["planning_seconds"] <= 60.0
    assert (result["planning_metrics"]["minco_optimizer_seconds"]
            <= result["planning_metrics"]["planning_seconds"])
    assert result["planning_metrics"]["joint_optimizer_restarts"] == 0
    assert all(result["planning_metrics"]["legs"][name]["optimizer_iterations"] > 0
               for name in ("pick", "place"))
    assert result["planning_metrics"]["legs"]["pick"]["rrt_nodes"] > 2
    assert result["planning_metrics"]["legs"]["place"]["rrt_nodes"] > 2
    pick_metrics = result["planning_metrics"]["legs"]["pick"]
    place_metrics = result["planning_metrics"]["legs"]["place"]
    assert pick_metrics["astar_expansions"] > 0
    assert place_metrics["astar_expansions"] > 0
    assert "astar_route_region" in pick_metrics["rrt_sampling_regions"]
    assert "astar_route_region" in place_metrics["rrt_sampling_regions"]
    assert pick_metrics["planned_movement_time_s"] > 0.0
    assert place_metrics["planned_movement_time_s"] > 0.0
    assert pick_metrics["planned_average_base_speed_mps"] > 0.0
    assert place_metrics["planned_average_base_speed_mps"] > 0.0
    assert max(pick_metrics["planned_average_joint_speed_rad_s"]) > 0.0
    assert max(place_metrics["planned_average_joint_speed_rad_s"]) > 0.0
    trace = result["joint_execution_trace"]
    assert trace["actual_rad"].shape[1] == 4
    assert np.ptp(trace["reference_rad"], axis=0).max() > 0.5
    assert np.linalg.norm(result["final_grasp_position"]-
                          result["place_target_position_ned"]) < .02
    assert result["released_payload_error"] <= .02
    assert np.linalg.norm(result["released_payload_position_ned"]-
                          result["place_target_position_ned"]) <= .02
    assert abs(result["final_gripper_opening"]-
               result["gripper_opening_target"]) < .005
    assert result["maximum_base_position_tracking_error_m"] < .25
    assert result["execution_movement_time_s"] > 0.0
    assert result["execution_average_base_speed_mps"] > 0.0
    assert np.max(result["execution_average_joint_speed_rad_s"]) > 0.0
    assert not result["collision"]


@pytest.mark.parametrize("primitives", [
    '<geom name="obstacle_rotated_validation" type="box" '
    'pos="5.55 -2.20 1.80" quat="0.9393727 0 0 0.3428978" '
    'size="0.12 0.42 0.16"/>',
    '<geom name="obstacle_sphere_validation" type="sphere" '
    'pos="0.50 -2.20 1.80" size="0.12"/>\n'
    '<geom name="obstacle_cylinder_validation" type="cylinder" '
    'pos="0.85 -2.20 1.80" size="0.12 0.25"/>',
])
def test_xml_only_scene_layout_runs_same_headless_pipeline(tmp_path, primitives):
    source = main.MODEL_DIRECTORY / "aerial_manipulator_workcell.xml"
    xml = source.read_text(encoding="utf-8")
    robot_xml = (main.MODEL_DIRECTORY.parent / "model" /
                 "aerial_manipulator.xml").resolve()
    xml = xml.replace("../model/aerial_manipulator.xml", str(robot_xml))
    xml = xml.replace("</worldbody>", f"{primitives}\n</worldbody>", 1)
    scene_path = tmp_path / "scene_primitives.xml"
    scene_path.write_text(xml, encoding="utf-8")
    config = load_config("configs/aerial_manipulator_workcell.yaml")
    config["scene"] = str(scene_path)
    config["visualize"] = False
    result = run(config)

    assert result["success"]
    assert result["event_sequence"] == [
        "PLAN_TASK", "MOVE_TO_PICK", "GRASP",
        "MOVE_TO_PLACE", "RELEASE", "DONE",
    ]
    assert result["pick_plan_valid"] is None and result["place_plan_valid"] is None
    assert not result["pick_validation_performed"] and not result["place_validation_performed"]
    assert result["released_payload_error"] <= .02
    assert not result["collision"]
