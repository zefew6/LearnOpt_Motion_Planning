import numpy as np
import mujoco
import pytest
from scipy.spatial.transform import Rotation

from uav_ac.robot.aerial_manipulator import AerialManipulatorCommand
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, ENU_TO_NED, MujocoSimulation


MODEL = "tests/fixtures/aerial_manipulator_robot.xml"


def test_aerial_manipulator_physics_step_matches_quadrotor():
    quadrotor = MujocoSimulation(DEFAULT_SCENE_PATH)
    manipulator = MujocoSimulation(MODEL)
    assert manipulator.model.opt.timestep == quadrotor.model.opt.timestep == 0.001


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


def test_opposing_gripper_geometries_keep_more_than_planning_self_clearance():
    sim = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    pairs = (
        ("gripper_pad_left", "gripper_pad_right"),
        ("gripper_finger_left", "gripper_pad_right"),
        ("gripper_pad_left", "gripper_finger_right"),
    )
    q = sim.robot.configuration.copy()
    for opening in (.020, .025, .070):
        q[11] = opening
        for joints in (np.zeros(4), np.array([.3, -.4, .2, -.1])):
            q[7:11] = joints
            result = sim.robot.exact_collision_distances(
                q, pairs=pairs, check_limits=False)
            assert np.all(result["distances"] >= opening-1e-10)


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


def test_attached_payload_collision_check_excludes_gripper_pairs():
    sim = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = sim.robot
    configuration = robot.configuration
    attached = robot.check_collision(configuration, clearance=0., payload_attached=True)
    assert not any("gripper_" in name for pair in attached["pairs"] for name in pair["geoms"]
                   if "payload_marker_geom" in pair["geoms"])


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


def test_public_planning_point_jacobian_supports_soft_limit_queries_without_live_mutation():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    q[:3] = [.4, -.2, -1.1]
    yaw = .35
    q[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    q[7:11] = [.2, -.3, .1, -.15]
    body, local = "arm_link_2_body", np.array([[.02, 0., -.04]])
    before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.time
    positions, jacobians = robot.point_positions_and_jacobians(q, [body], local)
    assert positions.shape == (1, 3) and jacobians.shape == (1, 3, 8)
    epsilon = 1e-6
    for column in range(8):
        shifted = q.copy()
        if column < 3:
            shifted[column] += epsilon
        elif column == 3:
            angle = yaw + epsilon
            shifted[3:7] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        else:
            shifted[7+column-4] += epsilon
        moved = robot.point_positions_and_jacobians(shifted, [body], local)[0]
        np.testing.assert_allclose((moved-positions)[0]/epsilon, jacobians[0, :, column],
                                   atol=3e-5)
    outside = q.copy(); outside[8] = robot.limits.joint_upper[1]+.1
    robot.point_positions_and_jacobians(outside, [body], local, check_limits=False)
    with pytest.raises(ValueError, match="joint limit"):
        robot.point_positions_and_jacobians(outside, [body], local)
    np.testing.assert_array_equal(sim.data.qpos, before[0])
    np.testing.assert_array_equal(sim.data.qvel, before[1])
    assert sim.time == before[2]


def test_exact_collision_distance_query_is_public_signed_and_non_mutating():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    q[:3] = [.3, -.1, -1.1]
    q[7:11] = [.2, -.3, .1, -.15]
    pairs = robot.collision_pairs("world")[:4]
    before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.time
    result = robot.exact_collision_distances(
        q, pairs=pairs, with_jacobians=True)
    assert {frozenset(pair) for pair in result["pairs"]} == {
        frozenset(pair) for pair in pairs}
    assert result["distances"].shape == (len(pairs),)
    assert result["points_ned"].shape == (len(pairs), 2, 3)
    assert result["jacobians"].shape == (len(pairs), 8)
    assert np.all(np.isfinite(result["distances"]))
    assert np.all(np.isfinite(result["jacobians"]))
    np.testing.assert_array_equal(sim.data.qpos, before[0])
    np.testing.assert_array_equal(sim.data.qvel, before[1])
    assert sim.time == before[2]


def test_exact_collision_distance_jacobian_matches_configuration_finite_difference():
    sim = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    q[:3] = [.8, -.1, -1.15]
    q[7:11] = [.35, -.4, .2, -.25]
    pair = ("arm_0", "obstacle_wall_1_left")
    analytic = robot.exact_collision_distances(
        q, pairs=(pair,), with_jacobians=True)["jacobians"][0]
    epsilon = 1.0e-6
    numerical = np.empty(8)
    for column in range(8):
        plus, minus = q.copy(), q.copy()
        if column < 3:
            plus[column] += epsilon
            minus[column] -= epsilon
        elif column == 3:
            plus[3:7] = [np.cos(epsilon/2), 0., 0., np.sin(epsilon/2)]
            minus[3:7] = [np.cos(epsilon/2), 0., 0., -np.sin(epsilon/2)]
        else:
            plus[7+column-4] += epsilon
            minus[7+column-4] -= epsilon
        distances = [robot.exact_collision_distances(
            state, pairs=(pair,))["distances"][0] for state in (plus, minus)]
        numerical[column] = (distances[0]-distances[1])/(2.*epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-4, atol=2e-5)


def test_batch_point_positions_use_kinematics_and_match_scalar_queries():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    configurations = np.repeat(robot.configuration[None, :], 3, axis=0)
    configurations[:, :3] = [[.2, -.3, -1.], [.5, .1, -1.2], [-.1, .2, -.9]]
    for index, angles in enumerate((.2, -.35, .6)):
        configurations[index, 3:7] = [np.cos(angles/2), 0., 0., np.sin(angles/2)]
    configurations[:, 7:11] = [[.1, -.2, .3, -.1], [.2, -.3, .1, -.2],
                                [-.1, .15, -.2, .1]]
    names = ("quadrotor", "arm_link_2_body", "gripper_palm_body")
    local = np.array([[.02, 0., -.03], [0., .01, -.04], [.01, 0., 0.]])
    before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.time
    actual = robot.point_positions_batch(configurations, names, local)
    expected = np.asarray([robot.point_positions(q, names, local)
                           for q in configurations])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(sim.data.qpos, before[0])
    np.testing.assert_array_equal(sim.data.qvel, before[1])
    assert sim.time == before[2]


def test_full_pose_point_jacobian_matches_world_rotation_tangent():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    q[:3] = [.4, -.2, -1.1]
    q[3:7] = Rotation.from_euler("xyz", [.2, -.25, .35]).as_quat()[[3, 0, 1, 2]]
    q[7:11] = [.2, -.3, .1, -.15]
    body, local = "arm_link_2_body", np.array([[.02, 0., -.04]])
    positions, jacobians = robot.point_positions_and_pose_jacobians(q, [body], local)
    assert jacobians.shape == (1, 3, 10)
    epsilon = 1e-7
    for column in range(10):
        shifted = q.copy()
        if column < 3:
            shifted[column] += epsilon
        elif column < 6:
            rotvec = np.zeros(3); rotvec[column-3] = epsilon
            quaternion = Rotation.from_quat(q[4:7].tolist()+[q[3]])
            shifted[3:7] = (Rotation.from_rotvec(rotvec)*quaternion).as_quat()[[3, 0, 1, 2]]
        else:
            shifted[7+column-6] += epsilon
        moved = robot.point_positions_and_pose_jacobians(shifted, [body], local)[0]
        np.testing.assert_allclose((moved-positions)[0]/epsilon,
                                   jacobians[0, :, column], atol=2e-5)


def test_exact_collision_distance_supports_full_pose_tangent_jacobian():
    sim = MujocoSimulation(
        "tests/fixtures/aerial_manipulator_gradient.xml",
        record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    q[:3] = [.8, -.1, -1.15]
    q[3:7] = Rotation.from_euler("xyz", [.15, -.2, .1]).as_quat()[[3, 0, 1, 2]]
    q[7:11] = [.35, -.4, .2, -.25]
    pair = ("arm_0", "obstacle_wall_1_left")
    analytic = robot.exact_collision_distances(
        q, pairs=(pair,), with_jacobians=True,
        with_pose_jacobians=True)["jacobians"][0]
    assert analytic.shape == (10,)
    epsilon = 1e-7
    numerical = np.empty(10)
    for column in range(10):
        plus, minus = q.copy(), q.copy()
        if column < 3:
            plus[column] += epsilon; minus[column] -= epsilon
        elif column < 6:
            rotvec = np.zeros(3); rotvec[column-3] = epsilon
            quaternion = Rotation.from_quat(q[4:7].tolist()+[q[3]])
            plus[3:7] = (Rotation.from_rotvec(rotvec)*quaternion).as_quat()[[3, 0, 1, 2]]
            minus[3:7] = (Rotation.from_rotvec(-rotvec)*quaternion).as_quat()[[3, 0, 1, 2]]
        else:
            plus[7+column-6] += epsilon; minus[7+column-6] -= epsilon
        distances = [robot.exact_collision_distances(state, pairs=(pair,))["distances"][0]
                     for state in (plus, minus)]
        numerical[column] = (distances[0]-distances[1])/(2.*epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-4, atol=3e-5)


def test_compiled_collision_pairs_preserve_order_distances_and_gradients():
    sim = MujocoSimulation('tests/fixtures/aerial_manipulator_gradient.xml',
                           record_actual_trajectory=False)
    robot = sim.robot
    pairs = (('arm_0', 'obstacle_wall_1_left'), ('arm_1', 'ground'))
    compiled = robot.compile_collision_pairs(pairs)
    q = robot.configuration.copy()
    q[:3] = [.8, -.1, -1.15]
    q[7:11] = [.35, -.4, .2, -.25]
    baseline = robot.exact_collision_distances(q, pairs=pairs, with_jacobians=True,
                                              with_pose_jacobians=True)
    actual = robot.exact_collision_distances(q, pairs=compiled, with_jacobians=True,
                                            with_pose_jacobians=True)
    assert actual['pairs'] == baseline['pairs']
    for key in ('distances', 'points_ned', 'jacobians'):
        np.testing.assert_allclose(actual[key], baseline[key], atol=1e-12)
    subset = robot.exact_collision_distances(q, pairs=compiled, pair_indices=[1, 0, 1])
    np.testing.assert_array_equal(subset['distances'], baseline['distances'][[1, 0, 1]])
    assert subset['pairs'] == tuple(baseline['pairs'][i] for i in (1, 0, 1))
    with pytest.raises(ValueError, match='indices'):
        robot.exact_collision_distances(q, pairs=compiled, pair_indices=[2])


def test_query_local_kinematics_shared_and_stale_context_rejected(monkeypatch):
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    pairs = robot.compile_collision_pairs(robot.collision_pairs('world')[:3])
    original = mujoco.mj_kinematics
    calls = []

    def kinematics(model, data):
        calls.append(data)
        return original(model, data)

    monkeypatch.setattr(mujoco, 'mj_kinematics', kinematics)
    prepared = robot.prepare_kinematics(q)
    robot.point_positions(q, ('quadrotor',), np.zeros((1, 3)), prepared=prepared)
    robot.exact_collision_distances(q, pairs=pairs, prepared=prepared)
    assert len(calls) == 1
    shifted = q.copy()
    shifted[0] += .1
    with pytest.raises(ValueError, match='configuration'):
        robot.exact_collision_distances(shifted, pairs=pairs, prepared=prepared)
    robot.point_positions(shifted, ('quadrotor',), np.zeros((1, 3)))
    with pytest.raises(ValueError, match='stale'):
        robot.exact_collision_distances(q, pairs=pairs, prepared=prepared)


def test_compiled_pairs_and_pose_context_cannot_cross_model_rebind():
    sim = MujocoSimulation(MODEL, record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    compiled = robot.compile_collision_pairs(robot.collision_pairs('world')[:1])
    prepared = robot.prepare_kinematics(q)
    robot.rebind(sim.model, lambda: sim.data)
    with pytest.raises(ValueError, match='model'):
        robot.exact_collision_distances(q, pairs=compiled)
    with pytest.raises(ValueError, match='model'):
        robot.exact_collision_distances(q, prepared=prepared)


def test_distance_only_query_omits_unused_witnesses_and_preserves_distances(monkeypatch):
    sim = MujocoSimulation('tests/fixtures/aerial_manipulator_gradient.xml',
                           record_actual_trajectory=False)
    robot = sim.robot
    q = robot.configuration.copy()
    compiled = robot.compile_collision_pairs(robot.collision_pairs('world')[:8])
    expected = robot.exact_collision_distances(q, pairs=compiled)
    original = mujoco.mj_geomDistance
    witnesses = []

    def distance(model, data, first, second, cutoff, witness):
        witnesses.append(witness)
        return original(model, data, first, second, cutoff, witness)

    monkeypatch.setattr(mujoco, 'mj_geomDistance', distance)
    actual = robot.exact_collision_distances(q, pairs=compiled, with_points=False)
    np.testing.assert_array_equal(actual['distances'], expected['distances'])
    assert actual['pairs'] == expected['pairs']
    assert 'points_ned' not in actual
    assert len(witnesses) == len(compiled.ids) and all(w is None for w in witnesses)


@pytest.mark.parametrize('payload_attached', [False, True])
def test_compiled_queries_match_native_distances_in_mixed_geometry_scene(tmp_path, payload_attached):
    from pathlib import Path

    source = Path('uav_ac/simulation/models/aerial_manipulator_workcell.xml')
    include = (source.parent.parent / 'model/aerial_manipulator.xml').resolve()
    xml = source.read_text().replace('../model/aerial_manipulator.xml', str(include))
    primitives = '''<geom name="obstacle_random_box" type="box" pos="2.4 0.4 1.2"
        quat="0.9393727 0 0 0.3428978" size="0.17 0.31 0.23"/>
        <geom name="obstacle_random_sphere" type="sphere" pos="3.1 -0.5 1.1" size="0.19"/>
        <geom name="obstacle_random_cylinder" type="cylinder" pos="4.0 0.7 1.3"
        quat="0.9800666 0.1986693 0 0" size="0.13 0.28"/>'''
    xml = xml.replace('</worldbody>', primitives+'</worldbody>', 1)
    path = tmp_path / 'mixed_geometry.xml'
    path.write_text(xml)
    sim = MujocoSimulation(path, record_actual_trajectory=False)
    robot = sim.robot
    compiled = robot.compile_collision_pairs(payload_attached=payload_attached)
    rng = np.random.default_rng(91)
    live_qpos, live_mocap = sim.data.qpos.copy(), sim.data.mocap_pos.copy()
    for _ in range(24):
        q = robot.configuration.copy()
        q[:3] = rng.uniform([0., -2., -1.8], [5., 2., -.6])
        q[7:11] = rng.uniform(robot.limits.joint_lower, robot.limits.joint_upper)
        q[3:7] = Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
        result = robot.exact_collision_distances(
            q, pairs=compiled, payload_attached=payload_attached, with_points=False)
        expected = [mujoco.mj_geomDistance(sim.model, robot._model._scratch,
                                         first, second, 1e6, None)
                    for first, second in compiled.ids]
        np.testing.assert_array_equal(result['distances'], expected)
    np.testing.assert_array_equal(sim.data.qpos, live_qpos)
    np.testing.assert_array_equal(sim.data.mocap_pos, live_mocap)


def test_compiled_pair_subset_preserves_empty_and_nonpayload_queries():
    sim = MujocoSimulation('tests/fixtures/aerial_manipulator_gradient.xml',
                           record_actual_trajectory=False)
    robot = sim.robot
    world = robot.collision_pairs('world')[:1]
    payload = (('payload_marker_geom', 'body'),)
    compiled = robot.compile_collision_pairs(world+payload, payload_attached=True)
    q = robot.configuration.copy()
    expected = robot.exact_collision_distances(q, pairs=world)
    subset = robot.exact_collision_distances(q, pairs=compiled, pair_indices=[0])
    np.testing.assert_array_equal(subset['distances'], expected['distances'])
    empty = robot.exact_collision_distances(q, pairs=compiled, pair_indices=[])
    assert empty['pairs'] == ()
    assert empty['distances'].shape == (0,)
    assert empty['points_ned'].shape == (0, 2, 3)
    with pytest.raises(ValueError, match='payload'):
        robot.exact_collision_distances(q, pairs=compiled, pair_indices=[1])
