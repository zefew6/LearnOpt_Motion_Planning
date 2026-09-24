import numpy as np
import mujoco
import pytest

from uav_ac.robot.aerial_manipulator import AerialManipulatorCommand
from uav_ac.simulation.mujoco_sim import ENU_TO_NED, MujocoSimulation


MODEL = "uav_ac/simulation/models/aerial_manipulator_hover.xml"


def test_model_state_and_queries_are_consistent_and_non_mutating():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    assert (sim.model.nq, sim.model.nv) == (13, 12)
    state = sim.robot_state
    assert state.base_state.shape == (13,)
    assert state.joint_positions.shape == state.joint_velocities.shape == (4,)
    qpos, qvel, time = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.time
    position, _ = sim.end_effector_pose()
    jacobian = sim.end_effector_jacobian()
    assert position.shape == (3,)
    assert jacobian.shape == (6, 12)
    assert 0.020 <= state.gripper_opening <= 0.070
    assert sim.configuration_collision(state.joint_positions) is False
    np.testing.assert_array_equal(sim.data.qpos, qpos)
    np.testing.assert_array_equal(sim.data.qvel, qvel)
    assert sim.data.time == time


def test_configuration_collision_includes_the_arm_and_environment():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    sim.data.qpos[2] = .4  # lower the complete robot until the hanging arm reaches the floor
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.configuration_collision(sim.robot_state.joint_positions) is True


def test_joint_limits_and_nonadjacent_self_collision_are_checked():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    with pytest.raises(ValueError, match="joint limit"):
        sim.configuration_collision([0.0, 2.3, 0.0, 0.0])
    folded = np.full(4, -2.2)
    assert sim.configuration_collision(folded) is True


def test_joint_jacobian_matches_finite_difference_at_rotated_configuration():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    data = sim.data
    data.qpos[3:7] = [np.cos(.2), 0, 0, np.sin(.2)]
    data.qpos[sim._arm_qpos_addresses] = [.25, -.4, .3, -.2]
    mujoco.mj_forward(sim.model, data)
    jacobian = sim.end_effector_jacobian()
    reference_linear = np.zeros((3, sim.model.nv))
    reference_angular = np.zeros_like(reference_linear)
    mujoco.mj_jacSite(sim.model, data, reference_linear, reference_angular,
                      sim._tool_site_id)
    expected = np.vstack((
        ENU_TO_NED @ reference_linear,
        ENU_TO_NED @ reference_angular,
    ))
    np.testing.assert_allclose(jacobian, expected, atol=1e-12)
    baseline, _ = sim.end_effector_pose()
    eps = 1e-6
    for joint_index, qpos_address in enumerate(sim._arm_qpos_addresses):
        data.qpos[qpos_address] += eps
        mujoco.mj_forward(sim.model, data)
        shifted, _ = sim.end_effector_pose()
        data.qpos[qpos_address] -= eps
        mujoco.mj_forward(sim.model, data)
        np.testing.assert_allclose(
            (shifted-baseline)/eps, jacobian[:3, 6+joint_index], atol=2e-5)


def test_torque_limits_and_reset_restore_hover_and_joint_state():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    sim.apply_robot_command(AerialManipulatorCommand(6.8, np.zeros(3), np.full(4, 50.0)))
    ranges = sim.model.actuator_ctrlrange[sim._arm_actuator_ids]
    np.testing.assert_allclose(sim.data.ctrl[sim._arm_actuator_ids], ranges[:, 1])
    sim.data.qpos[sim._arm_qpos_addresses] = [.2, .1, -.1, .3]
    sim.reset()
    np.testing.assert_allclose(sim.robot_state.joint_positions, np.zeros(4))
    assert np.all(sim.quad.omega > 0.0)
    assert sim.robot_state.gripper_opening == pytest.approx(0.020)


def test_joint_motion_reacts_on_floating_base():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    sim.apply_robot_command(AerialManipulatorCommand(
        sim.total_mass*sim.quad.g, np.zeros(3), np.array([0, .5, 0, 0])))
    sim.step()
    assert np.linalg.norm(sim.data.qvel[3:6]) > 0.0


def test_public_reduced_model_has_consistent_configuration_dynamics_and_collision_queries():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    q[:3] = [1.2, -0.4, -1.5]
    q[3:7] = [np.cos(.15), 0, 0, np.sin(.15)]
    q[7:11] = [.2, -.3, .15, -.1]
    q[11] = .052
    before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.time
    pose, jac = robot.forward_kinematics(q, "grasp"), robot.jacobian(q, "grasp")
    assert pose[0].shape == (3,) and pose[1].shape == (4,)
    assert jac.shape == (6, 11)
    eps = 1e-6
    for index in (0, 3, 6, 7, 8, 9):
        delta = np.zeros(11); delta[index] = eps
        shifted, _ = robot.forward_kinematics(robot.integrate(q, delta), "grasp")
        np.testing.assert_allclose((shifted-pose[0])/eps, jac[:3, index], atol=2e-5)
    model = robot.dynamics(q, np.zeros(11))
    np.testing.assert_allclose(model["mass_matrix"], model["mass_matrix"].T, atol=1e-12)
    assert np.linalg.eigvalsh(model["mass_matrix"]).min() > 0
    assert model["actuation_matrix"].shape == (11, 9)
    assert robot.difference(q, robot.integrate(q, np.ones(11)*1e-5)) == pytest.approx(
        np.ones(11)*1e-5, abs=2e-9)
    collision = robot.check_collision(q, clearance=.01)
    assert "minimum_distance" in collision and "pairs" in collision
    np.testing.assert_array_equal(sim.data.qpos, before[0])
    np.testing.assert_array_equal(sim.data.qvel, before[1])
    assert sim.time == before[2]


def test_command_holds_rotor_target_until_each_physics_step_and_full_reset_restores_state():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    initial_speed = sim.quad.omega.copy()
    command = AerialManipulatorCommand(5.5, np.zeros(3), np.zeros(4), .05)
    robot.apply(command)
    robot.apply(command)
    np.testing.assert_array_equal(sim.quad.omega, initial_speed)
    sim.step()
    assert np.any(sim.quad.omega != initial_speed)
    q = robot.configuration.copy(); q[7:11] = [.1, .2, -.1, .05]; q[11] = .06
    v = np.arange(11, dtype=float)*.001
    robot.reset(q, v)
    np.testing.assert_allclose(robot.configuration, q, atol=1e-12)
    np.testing.assert_allclose(robot.velocity, v, atol=1e-12)
    robot.reset()
    np.testing.assert_allclose(robot.configuration[7:11], 0)
    assert robot.gripper_opening == pytest.approx(.02)


def test_full_configuration_collision_reports_environment_and_nonadjacent_self_pairs():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    q = sim.robot.configuration.copy()
    q[2] = -.4  # move the complete robot so the gripper crosses the ground
    environment_result = sim.robot.check_collision(q)
    assert environment_result["collision"]
    assert any("ground" in pair["geoms"] for pair in environment_result["pairs"])

    q = sim.robot.configuration.copy()
    q[7:11] = -2.2
    self_result = sim.robot.check_collision(q)
    assert self_result["collision"]
    for pair in self_result["pairs"]:
        a, b = (mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in pair["geoms"])
        body_a, body_b = sim.model.geom_bodyid[a], sim.model.geom_bodyid[b]
        assert body_a != body_b
        assert sim.model.body_parentid[body_a] != body_b
        assert sim.model.body_parentid[body_b] != body_a


def test_public_jacobian_uses_world_ned_rows_and_public_tangent_columns():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    q = sim.robot.configuration.copy()
    q[3:7] = [np.cos(.2), 0, 0, np.sin(.2)]
    q[7:11] = [.25, -.4, .3, -.2]
    sim.robot.reset(q)
    jac = sim.robot.jacobian(q)
    jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
    mujoco.mj_jacSite(sim.model, sim.data, jp, jr, sim._tool_site_id)
    lift = np.zeros((sim.model.nv, 11))
    lift[:3, :3] = ENU_TO_NED
    lift[3:6, 3:6] = ENU_TO_NED
    lift[sim._arm_dof_addresses, 6:10] = np.eye(4)
    lift[sim._gripper_dof_addresses, 10] = .5
    reference = np.vstack((ENU_TO_NED@jp, ENU_TO_NED@jr))@lift
    np.testing.assert_allclose(jac, reference, atol=1e-12)
