import numpy as np
import pytest

from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import TaskWaypoint
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import (
    recover_state, stationary_waypoint)
from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import task_residual, TaskEventConstraint
from uav_ac.planning.trajectory.gcopter.optimizer import solve_equalities
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.trajectory.aerial_manipulator_minco.optimization import JointTaskObjective


@pytest.fixture
def robot():
    return MujocoSimulation('tests/fixtures/aerial_manipulator_gradient.xml').robot


def test_stationary_task_waypoint_position_is_exact_with_free_arm(robot):
    target = np.array([.2, -.1, -.8])
    waypoint = TaskWaypoint(target)
    target[:] = 42.
    angles = np.r_[.3, robot.configuration[7:11]]
    state, derivative = stationary_waypoint(robot, waypoint, angles, .04)
    q, v = recover_state(np.vstack((state, np.zeros((3, 8)))), .04)
    np.testing.assert_allclose(robot.forward_kinematics(q, frame='grasp')[0], waypoint.position, atol=1e-12)
    np.testing.assert_allclose(task_residual(robot, waypoint, np.vstack((state, np.zeros((3, 8)))), .04), 0., atol=1e-12)
    for k in range(5):
        delta = np.zeros(5); delta[k] = 1e-6
        numeric = (stationary_waypoint(robot, waypoint, angles+delta, .04)[0]
                   -stationary_waypoint(robot, waypoint, angles-delta, .04)[0])/2e-6
        np.testing.assert_allclose(derivative[:, k], numeric, atol=1e-7)
    assert not waypoint.position.flags.writeable


def test_flatness_twist_matches_forward_motion(robot):
    y = np.r_[.2, .1, -.8, .3, robot.configuration[7:11]]
    derivatives = np.stack((y, np.linspace(.01, .08, 8),
                            np.linspace(.02, .09, 8), np.linspace(.01, .04, 8)))
    q, v = recover_state(derivatives, .04)
    dt = 1e-6
    plus = derivatives.copy(); minus = derivatives.copy()
    plus[:3] += dt * derivatives[1:4]
    minus[:3] -= dt * derivatives[1:4]
    qp, _ = recover_state(plus, .04); qm, _ = recover_state(minus, .04)
    numeric = (robot.forward_kinematics(qp, frame='grasp')[0]
               -robot.forward_kinematics(qm, frame='grasp')[0])/(2*dt)
    np.testing.assert_allclose((robot.jacobian(q, frame='grasp') @ v)[:3], numeric, atol=1e-7)


def test_reference_uses_same_singularity_diagnostic():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
    coefficients = np.zeros((1, 6, 8)); coefficients[0, 2, 2] = 9.81 / 2
    trajectory = AerialManipulatorTrajectory(np.ones(1), coefficients, np.zeros((2, 8)), 0., 0, True, 'test')
    with pytest.raises(ValueError, match='singularity'):
        trajectory.reference(0., None, gripper_opening=.04)


def test_task_event_partial_pose_velocity_and_duration_jacobian(robot):
    coefficients = np.zeros((1, 6, 8))
    coefficients[0, 0] = np.r_[.2, .1, -.8, .3, robot.configuration[7:11]]
    coefficients[0, 1, :4] = [.1, -.05, .02, .04]
    coefficients[0, 2, :3] = [.02, -.01, .01]
    coefficients[0, 3, :3] = [.002, .001, 0.]
    times = np.array([.7])
    q, _ = recover_state(np.stack([coefficients[0, d] for d in range(4)]), .04)
    position, quaternion = robot.forward_kinematics(q, frame='grasp')
    waypoint = TaskWaypoint(position, -quaternion, linear_velocity=np.zeros(3),
                            position_mask=[True, False, True], orientation_mask=[False, False, True])
    event = TaskEventConstraint(robot, waypoint, knot=1, gripper_opening=.04)
    residual, gc, gt = event.linearize(times, coefficients)
    assert residual.size == 6
    dt = 1e-6
    numeric = (event.linearize(times+dt, coefficients)[0]-event.linearize(times-dt, coefficients)[0])/(2*dt)
    np.testing.assert_allclose(gt[:, 0], numeric, rtol=1e-5, atol=1e-6)
    plus, minus = coefficients.copy(), coefficients.copy()
    plus[0, 3, 0] += dt; minus[0, 3, 0] -= dt
    numeric = (event.linearize(times, plus)[0]-event.linearize(times, minus)[0])/(2*dt)
    np.testing.assert_allclose(gc[:, 0, 3, 0], numeric, rtol=1e-5, atol=1e-6)


def test_equality_solver_preserves_target_and_reports_infeasibility():
    objective = lambda x: (float(x@x), 2*x)
    residual = lambda x: (np.array([x[0]-2.]), np.ones((1, 1)))
    result = solve_equalities(objective, np.zeros(1), residual, tolerance=1e-6)
    assert result.feasible
    assert abs(result.x[0]-2.) < 1e-6
    bad = solve_equalities(objective, np.zeros(1),
                           lambda x: (np.ones(1), np.zeros((1, 1))), max_outer_iterations=3)
    assert not bad.feasible
    np.testing.assert_array_equal(bad.residual, np.ones(1))


def test_boundary_adjoint_includes_shared_event_state():
    head = np.zeros((3, 8)); tail = head.copy(); tail[0] = .4
    durations = np.array([.8, 1.2]); points = np.full((1, 8), .15)
    minco = MINCOQuintic(head, tail, 2)
    result = evaluate_minco_objective(minco, points, durations, boundary_gradients=True)
    for k in (0, 3, 7):
        delta = np.zeros_like(tail); delta[0, k] = 1e-6
        plus = evaluate_minco_objective(MINCOQuintic(head, tail+delta, 2), points, durations)[0]
        minus = evaluate_minco_objective(MINCOQuintic(head, tail-delta, 2), points, durations)[0]
        np.testing.assert_allclose(result[-1][0, k], (plus-minus)/2e-6, rtol=1e-6)


def test_joint_problem_shared_event_and_independent_time_gradients(robot):
    from types import SimpleNamespace
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
    targets = [TaskWaypoint([.3, 0., -.8], linear_velocity=[0., 1., 0.],
                            linear_velocity_mask=[True, False, True]), TaskWaypoint([.8, 0., -.8])]
    angles = np.r_[0., robot.configuration[7:11]]
    events = [stationary_waypoint(robot, target, angles, .04)[0] for target in targets]
    start = events[0].copy(); start[0] -= .3
    knots = [np.linspace(start, events[0], 4), np.linspace(events[0], events[1], 4)]
    def penalty(t, c):
        return 0., np.zeros_like(c), np.zeros_like(t)
    evaluators = [SimpleNamespace(integrated_penalty=penalty) for _ in range(2)]
    problem = JointTaskObjective(robot, AerialManipulatorMINCOConfig(), start, targets,
                                 knots, evaluators, [.04, .02])
    x = problem.initial
    for _, time_slice, _, _ in problem.layouts:
        np.testing.assert_array_equal(problem.scale[time_slice], 1.)
    cost, gradient = problem.objective(x)
    residual, jacobian = problem.equalities(x)
    np.testing.assert_allclose(residual, 0., atol=1e-9)
    for k in (0, 4, 5, 9, 10, len(x)-1, len(x)-3):
        delta = np.zeros_like(x); delta[k] = 1e-6
        numeric = (problem.objective(x+delta)[0]-problem.objective(x-delta)[0])/2e-6
        np.testing.assert_allclose(gradient[k], numeric, rtol=2e-5, atol=2e-5)
        numeric = (problem.equalities(x+delta)[0]-problem.equalities(x-delta)[0])/2e-6
        np.testing.assert_allclose(jacobian[:, k], numeric, rtol=2e-5, atol=2e-5)
    # Gripper dwell losses must participate in the same event-state pullback.
    for evaluator, gap in zip(evaluators, [.04, .02]):
        evaluator.gripper_opening = gap
        evaluator.last_violation, evaluator.minimum_clearance = -1., 1.
        def sample(y, v, a, j, evaluator=evaluator):
            g = np.zeros((4, 8)); g[0] = 2*y + np.arange(1., 9.)
            return float(y@y + y@np.arange(1., 9.) + evaluator.gripper_opening**2), g, -1., 1.
        evaluator.sample_cost_gradient = sample
    problem.dwell_time = .3
    _, gradient = problem.objective(x)
    for k in (0, 4, 5, 9):
        delta = np.zeros_like(x); delta[k] = 1e-6
        numeric = (problem.objective(x+delta)[0]-problem.objective(x-delta)[0])/2e-6
        np.testing.assert_allclose(gradient[k], numeric, rtol=2e-5, atol=2e-5)
    assert [e.gripper_opening for e in evaluators] == [.04, .02]
    result = SimpleNamespace(iterations=1, converged=True, message='test')
    plans = problem.trajectories(x, result)
    assert sum(plan.cost for plan in plans) == pytest.approx(problem.objective(x)[0])
    blocks = problem.decode(x)[0]
    np.testing.assert_array_equal(blocks[0][1], blocks[1][0])
    assert residual.shape == (8,)


def test_joint_timeline_has_explicit_smooth_grasp_and_release():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import JointTaskTrajectory, AerialManipulatorTrajectory
    coefficients = np.zeros((1, 6, 8))
    pick = AerialManipulatorTrajectory(np.array([2.]), coefficients, np.zeros((2, 8)), 0., 0, True, 'test')
    place = AerialManipulatorTrajectory(np.array([3.]), coefficients, np.zeros((2, 8)), 0., 0, True, 'test')
    task = JointTaskTrajectory(pick, place, .06, .02, .3)
    assert task.total_time == pytest.approx(5.6)
    np.testing.assert_allclose(task.gap_motion('grasp', 0.), [.06, 0., 0.])
    assert task.gap_motion('grasp', .15)[0] == pytest.approx(.04)
    np.testing.assert_allclose(task.gap_motion('grasp', .3), [.02, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(task.gap_motion('release', .3), [.06, 0., 0.], atol=1e-12)


def test_penetrating_capsule_box_uses_a_differentiable_escape_penalty():
    from uav_ac.planning.geometry.grid_map import GridMap
    from uav_ac.planning.geometry.esdf import ESDF
    from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import AerialManipulatorTrajectoryEvaluator
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
    sim = MujocoSimulation('uav_ac/simulation/models/aerial_manipulator_workcell.xml', record_actual_trajectory=False)
    cfg = AerialManipulatorMINCOConfig(esdf_resolution=.1, esdf_discretization_margin=.18)
    grid = GridMap.from_scene_geometries(sim.scene_geometries,
        sim.space_limits[0]-.8, sim.space_limits[1]+.8, .1)
    evaluator = AerialManipulatorTrajectoryEvaluator(sim.robot, ESDF.from_occupancy(grid),
        sim.quad, cfg, sim.space_limits, .025, True, occupancy=grid)
    state = np.array([4.029294202154599, .4123031683863034, -1.9053766999459676,
        1.5778458157563202, .2787360339266117, -.09931383869968922,
        .21620550616971212, -.3673103316216305])
    acceleration = np.array([3.6874073811360226, -3.5825827951730274, -.48476099572162046, 0., 0., 0., 0., 0.])
    cost, gradient = evaluator.collision_cost_gradient(state, acceleration)[:2]
    for index in (0, 3, 7):
        plus, minus = state.copy(), state.copy(); plus[index] += 1e-6; minus[index] -= 1e-6
        numeric = (evaluator.collision_cost_gradient(plus, acceleration)[0]
                   -evaluator.collision_cost_gradient(minus, acceleration)[0])/2e-6
        np.testing.assert_allclose(gradient[index], numeric, rtol=2e-4, atol=.02)


def test_dwell_validation_checks_intermediate_gap_and_restores_context():
    from types import SimpleNamespace
    from uav_ac.planning.trajectory.aerial_manipulator_minco.validation import validate_gripper_dwell
    from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
    trajectory = AerialManipulatorTrajectory(np.ones(1), np.zeros((1, 6, 8)), np.zeros((2, 8)), 0., 0, True, 'test')
    seen = []
    evaluator = SimpleNamespace(config=SimpleNamespace(validation_dt=.025), gripper_opening=.06)
    def validate(trajectory):
        seen.append(evaluator.gripper_opening)
        valid = abs(evaluator.gripper_opening-.04) > 1e-6
        return valid, .1 if valid else -.01, 0. if valid else .02, .025
    evaluator.dense_validate = validate
    valid, clearance, violation, dt = validate_gripper_dwell(evaluator, trajectory, .06, .02, .3)
    assert not valid and clearance == -.01 and violation == .02
    assert min(abs(gap-.04) for gap in seen) < 1e-6
    assert evaluator.gripper_opening == .06


def test_orientation_validation_uses_rotation_angle_norm(robot):
    from scipy.spatial.transform import Rotation
    from uav_ac.planning.trajectory.aerial_manipulator_minco.validation import validate_task_waypoint
    y = np.r_[.2, .1, -.8, .3, robot.configuration[7:11]]
    derivatives = np.vstack((y, np.zeros((3, 8))))
    q, _ = recover_state(derivatives, .04)
    position, quaternion = robot.forward_kinematics(q, frame='grasp')
    current = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
    desired = (current * Rotation.from_rotvec([-.0008, -.0008, 0.])).as_quat()
    target = TaskWaypoint(position, np.r_[desired[3], desired[:3]])
    valid, residual, errors = validate_task_waypoint(robot, target, derivatives, .04)
    assert not valid
    assert errors['orientation_rad'] == pytest.approx(np.sqrt(2)*.0008)


def test_task_returns_unconverged_result_without_restart(monkeypatch):
    from types import SimpleNamespace
    from uav_ac.planning.trajectory.aerial_manipulator_minco import optimization
    from uav_ac.planning.trajectory.aerial_manipulator_minco.planner import AerialManipulatorMINCO
    candidates = []
    def solve(planner, *args, **kwargs):
        converged = bool(candidates)
        plans = {name: SimpleNamespace(validation_passed=True, optimizer_converged=converged,
                                      maximum_violation=0.) for name in ('pick', 'place')}
        candidates.append(plans)
        planner.last_metrics = dict(joint_optimizer_calls=1, optimizer_seconds=1.)
        return plans, {}
    monkeypatch.setattr(optimization, 'plan_task', solve)
    planner = AerialManipulatorMINCO()
    plans, _ = planner.plan_task(np.zeros(8), [], [], robot=None, esdf=None, quad=None,
                                 workspace_bounds=None, gaps=[.06, .02])
    assert plans is candidates[0]
    assert len(candidates) == 1
    assert planner.last_metrics['joint_optimizer_calls'] == 1


def test_completed_joint_candidate_is_not_restarted(monkeypatch):
    from types import SimpleNamespace
    from uav_ac.planning.trajectory.aerial_manipulator_minco import optimization
    from uav_ac.planning.trajectory.aerial_manipulator_minco.planner import AerialManipulatorMINCO
    plans = {name: SimpleNamespace(validation_passed=True, optimizer_converged=False,
                                  maximum_violation=0.) for name in ('pick', 'place')}
    calls = []
    def solve(planner, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('retry initialization failed')
        planner.last_metrics = dict(joint_optimizer_calls=1, optimizer_seconds=1.)
        return plans, {}
    monkeypatch.setattr(optimization, 'plan_task', solve)
    planner = AerialManipulatorMINCO()
    actual, _ = planner.plan_task(np.zeros(8), [], [], robot=None, esdf=None, quad=None,
                                  workspace_bounds=None, gaps=[.06, .02])
    assert actual is plans
    assert len(calls) == 1
