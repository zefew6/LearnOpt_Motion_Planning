import numpy as np
import pytest

from uav_ac.planning.trajectory.aerial_manipulator_minco import make_terminal_state
from uav_ac.planning.trajectory.aerial_manipulator_minco.task_targets import yaw_quaternion
from uav_ac.scenes.loader import ENU_TO_NED
from uav_ac.simulation.mujoco_sim import MujocoSimulation


MODEL = "uav_ac/simulation/models/aerial_manipulator_pick_place.xml"


def _ned_bounds(simulation, name):
    geom = simulation.model.geom(name)
    center = ENU_TO_NED @ simulation.data.geom_xpos[geom.id]
    size = simulation.model.geom_size[geom.id]
    return np.r_[center-size, center+size]


def _full_configuration(state, opening):
    return np.r_[state[:3], yaw_quaternion(state[3]), state[4:], opening]


def test_wall_apertures_and_narrow_shelf_match_the_ned_scene_contract():
    simulation = MujocoSimulation(MODEL, record_actual_trajectory=False)

    expected = {
        "obstacle_wall_1_top": [.88, -.36, -3.00, 1.12, .36, -1.41],
        "obstacle_wall_1_bottom": [.88, -.36, -.99, 1.12, .36, .00],
        "obstacle_wall_2_top": [3.08, -.36, -3.00, 3.32, .36, -1.61],
        "obstacle_wall_2_bottom": [3.08, -.36, -1.19, 3.32, .36, .00],
        "obstacle_shelf_bottom": [1.76, .75, -.91, 2.24, 1.25, -.83],
        "obstacle_shelf_top": [1.76, .84, -1.27, 2.24, 1.25, -1.19],
        "obstacle_shelf_left": [1.76, .75, -1.19, 1.84, 1.25, -.91],
        "obstacle_shelf_right": [2.16, .75, -1.19, 2.24, 1.25, -.91],
        "obstacle_shelf_back": [1.76, 1.25, -1.27, 2.24, 1.33, -.83],
    }
    for name, bounds in expected.items():
        lower_upper = _ned_bounds(simulation, name)
        np.testing.assert_allclose(lower_upper[:3], bounds[:3], atol=1e-12)
        np.testing.assert_allclose(lower_upper[3:], bounds[3:], atol=1e-12)

    assert simulation.space_limits == pytest.approx(
        np.array([[-.8, -2., -2.], [4.8, 2., -.7]]))
    assert not simulation.has_collision


def test_collision_environment_aabbs_preserve_ned_boxes_and_ground_plane():
    simulation = MujocoSimulation(MODEL, record_actual_trajectory=False)
    bounds = simulation.robot.collision_environment_aabbs()
    np.testing.assert_allclose(bounds["obstacle_wall_1_top"], [
        [.88, -.36, -3.00], [1.12, .36, -1.41]], atol=1e-12)
    ground = bounds["ground"]
    assert np.all(np.isneginf(ground[0, :2]))
    assert np.all(np.isposinf(ground[1, :2]))
    assert ground[0, 2] == pytest.approx(0.)
    assert ground[1, 2] == pytest.approx(0.)


def test_static_shelf_blocks_a_vertical_arm_but_admits_the_l_grasp_pose():
    simulation = MujocoSimulation(MODEL, record_actual_trajectory=False)
    target = np.array([2., .85, -1.05])
    vertical = make_terminal_state(
        simulation.robot, target, np.pi/2, np.zeros(4),
        gripper_opening=.06, workspace_bounds=simulation.space_limits)
    l_shaped = make_terminal_state(
        simulation.robot, target, np.pi/2, np.array([0., 0., -np.pi/2, 0.]),
        gripper_opening=.06, workspace_bounds=simulation.space_limits)

    assert simulation.robot.check_collision(_full_configuration(vertical, .06))["collision"]
    assert not simulation.robot.check_collision(_full_configuration(l_shaped, .06))["collision"]


def test_wall_apertures_require_a_compact_arm_configuration():
    """Regression poses only: production RRT still chooses all intermediate joints."""
    simulation = MujocoSimulation(MODEL, record_actual_trajectory=False)

    def configuration(position, joints, opening):
        return np.r_[position, yaw_quaternion(0.), joints, opening]

    compact = np.array([0., -2., -.46, 1.38])
    assert simulation.robot.check_collision(configuration(
        [1., 0., -1.232], np.zeros(4), .06))["collision"]
    assert not simulation.robot.check_collision(configuration(
        [1., 0., -1.232], compact, .06))["collision"]
    assert simulation.robot.check_collision(configuration(
        [3.2, 0., -1.432], [0., 0., -np.pi/2, 0.], .025))["collision"]
    assert not simulation.robot.check_collision(
        configuration([3.2, 0., -1.432], compact, .025),
        payload_attached=True)["collision"]
