"""Independent algebra, feasibility, failure, and published-duration BMTP checks."""

from dataclasses import replace
from math import comb, factorial

import cvxpy as cp
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from uav_ac.planning.geometry import ConvexPolytope
from uav_ac.planning.trajectory.bmtp import (
    BMTPConfig,
    BMTPLimits,
    BMTPPlanner,
    BMTPTrajectory,
    constraint_residuals,
)
from uav_ac.planning.trajectory.bmtp import planner as planner_module
from uav_ac.planning.trajectory.bmtp.bezier import (
    derivative_matrix,
    elevate_matrix,
    evaluate,
    product_weights,
    separation_coefficients,
    split,
)
from uav_ac.planning.trajectory.bmtp.collision import (
    box,
    collision_free,
    collisions,
    normalized,
    offset,
)


def _direct(points, parameters, order=0):
    """Power-basis oracle, independent of the production Bernstein operators."""
    points = np.asarray(points)
    degree = len(points) - 1
    u, one_minus_u = Polynomial([0, 1]), Polynomial([1, -1])
    basis = [comb(degree, i) * u**i * one_minus_u**(degree - i)
             for i in range(degree + 1)]
    return np.stack([
        sum((basis[i] * points[i, axis] for i in range(degree + 1)),
            Polynomial([0])).deriv(order)(parameters)
        for axis in range(points.shape[1])
    ], axis=-1)


@pytest.fixture
def config():
    return BMTPConfig(degree=6, continuity_order=2, terminal_order=2)


@pytest.fixture
def limits():
    return BMTPLimits(velocity=10, acceleration=30, jerk=None, snap=None)


@pytest.fixture
def detour():
    path = np.array([[-1, 0, 0], [-1, 1, 0], [1, 1, 0], [1, 0, 0]], float)
    obstacles = [box([-.2, -.2, -1], [.2, .2, 1])]
    domain = box([-2, -2, -1], [2, 2, 1])
    return path, obstacles, domain


def _assert_feasible(trajectory, path, obstacles, domain, limits, config):
    """Check the certificate and independently check every physical constraint."""
    residuals = constraint_residuals(
        trajectory, path[0], path[-1], domain, limits, config)
    assert set(residuals) == {"boundary", "workspace", "continuity", "terminal", "derivative"}
    assert all(np.isfinite(v) and 0 <= v <= config.feasibility_tolerance
               for v in residuals.values()), residuals
    points, h = trajectory.control_points, trajectory.segment_time
    tolerance = config.feasibility_tolerance
    assert np.all(np.isfinite(points)) and np.isfinite(h) and h > 0
    np.testing.assert_allclose(points[0, 0], path[0], atol=tolerance, rtol=0)
    np.testing.assert_allclose(points[-1, -1], path[-1], atol=tolerance, rtol=0)
    assert np.max(points @ domain.A.T - domain.b) <= tolerance
    degree = points.shape[1] - 1
    for order in range(max(len(limits.values), config.continuity_order,
                           config.terminal_order) + 1):
        controls = (np.diff(points, n=order, axis=1)
                    * factorial(degree) / factorial(degree - order) / h**order)
        scale = limits.values[order - 1] if 0 < order <= len(limits.values) else 1
        if order <= config.continuity_order and len(points) > 1:
            assert np.max(np.linalg.norm(controls[:-1, -1] - controls[1:, 0], axis=1)) <= tolerance * scale
        if 0 < order <= config.terminal_order:
            assert np.max(np.linalg.norm(controls[[0, -1], [0, -1]], axis=1)) <= tolerance * scale
        if 0 < order <= len(limits.values):
            assert np.max(np.linalg.norm(controls, axis=2)) <= scale * (1 + tolerance)
    assert not collisions(points, obstacles, config.collision_tolerance,
                          config.collision_max_depth)


@pytest.mark.parametrize("order", range(7))
def test_derivative_matrix_matches_direct_polynomial(order):
    points = np.random.default_rng(51).normal(size=(7, 3))
    parameters = np.linspace(0, 1, 29)
    matrix = derivative_matrix(6, order)
    assert matrix.shape == (7 - order, 7)
    np.testing.assert_allclose(evaluate(matrix @ points, parameters),
                               _direct(points, parameters, order), atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize("degree,target", [(0, 0), (0, 6), (2, 2), (2, 7), (6, 9)])
def test_elevation_preserves_direct_curve_and_convex_weights(degree, target):
    points = np.random.default_rng(4).normal(size=(degree + 1, 3))
    elevation = elevate_matrix(degree, target)
    parameters = np.linspace(0, 1, 31)
    assert elevation.shape == (target + 1, degree + 1)
    assert np.all(elevation >= 0)
    np.testing.assert_allclose(elevation.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(evaluate(elevation @ points, parameters),
                               _direct(points, parameters), atol=1e-11)
    np.testing.assert_allclose((elevation @ points)[[0, -1]], points[[0, -1]])


@pytest.mark.parametrize("degree,plane_degree", [(0, 0), (0, 3), (3, 0), (6, 1), (3, 4)])
def test_bernstein_dot_and_affine_plane_match_direct_evaluation(degree, plane_degree):
    rng = np.random.default_rng(17)
    points = rng.normal(size=(degree + 1, 3))
    normal = rng.normal(size=(plane_degree + 1, 3))
    bias = rng.normal(size=plane_degree + 1)
    parameters = np.linspace(0, 1, 37)
    expected_dot = np.sum(_direct(points, parameters) * _direct(normal, parameters), axis=1)
    weights = product_weights(degree, plane_degree)
    dot = np.einsum("kij,il,jl->k", weights, points, normal)
    np.testing.assert_allclose(evaluate(dot, parameters), expected_dot, atol=2e-11)
    np.testing.assert_allclose(weights.sum(axis=(1, 2)), 1, atol=1e-14)
    coefficients = separation_coefficients(points, normal, bias)
    expected = expected_dot + _direct(bias[:, None], parameters)[:, 0]
    np.testing.assert_allclose(evaluate(coefficients, parameters), expected, atol=2e-11)


@pytest.mark.parametrize("degree", [0, 1, 2, 6])
def test_split_matches_both_halves_of_direct_curve_without_mutation(degree):
    points = np.random.default_rng(29).normal(size=(degree + 1, 3))
    original = points.copy()
    left, right = split(points)
    parameters = np.linspace(0, 1, 23)
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(evaluate(left, parameters), _direct(points, parameters / 2), atol=1e-11)
    np.testing.assert_allclose(evaluate(right, parameters), _direct(points, .5 + parameters / 2), atol=1e-11)
    np.testing.assert_allclose(left[-1], right[0])
    np.testing.assert_array_equal(left[0], points[0])
    np.testing.assert_array_equal(right[-1], points[-1])


@pytest.mark.parametrize("order", [-1, 7])
def test_derivative_rejects_invalid_order(order):
    with pytest.raises(ValueError, match="derivative order"):
        derivative_matrix(6, order)


def test_elevation_rejects_degree_reduction():
    with pytest.raises(ValueError, match="cannot lower"):
        elevate_matrix(6, 5)


def test_curved_collision_is_detected_when_endpoints_and_chord_are_free():
    points = np.array([[-2, 0, 0], [0, 2, 0], [2, 0, 0]], float)
    obstacle = box([-.2, .8, -1], [.2, 1.2, 1])
    assert not np.any(obstacle.contains(points[[0, -1]]))
    assert collision_free(points[[0, -1]], obstacle)
    assert obstacle.contains(_direct(points, .5))
    assert not collision_free(points, obstacle)
    assert not collision_free(points[::-1], obstacle)


def test_tangent_contact_is_not_certified_free():
    # y=(u-1/3)^2 touches at a non-dyadic parameter; endpoints remain outside.
    points = np.array([[-1, 1/9, 0], [0, -2/9, 0], [1, 4/9, 0]])
    obstacle = box([-.5, -1, -1], [0, 0, 1])
    assert not np.any(obstacle.contains(points[[0, -1]]))
    assert _direct(points, 1/3)[1] == pytest.approx(0, abs=1e-15)
    assert not collision_free(points, obstacle, tolerance=1e-6)


@pytest.mark.parametrize("endpoint", [[0, 0, 0], [1, 0, 0]])
@pytest.mark.parametrize("reverse", [False, True])
def test_inside_or_boundary_endpoint_is_collision(endpoint, reverse):
    points = np.array([endpoint, [2, 2, 0]], float)
    assert not collision_free(points[::-1] if reverse else points, box([-1]*3, [1]*3))


def test_subdivision_certifies_clear_curve_but_exhaustion_is_conservative():
    points = np.array([[-2, 0, 0], [0, 2, 0], [2, 0, 0]], float)
    obstacle = box([-.2, -.2, -1], [.2, .2, 1])
    assert not collision_free(points, obstacle, max_depth=0)
    assert not collision_free(points, obstacle, tolerance=10)
    assert collision_free(points, obstacle, tolerance=1e-6)
    assert collision_free(points + [0, 3, 0], obstacle, max_depth=0)


def test_collision_tags_identify_segment_and_obstacle():
    segments = np.array([[[-2, 0, 0], [2, 0, 0]], [[-2, 3, 0], [2, 3, 0]]], float)
    first = box([-.1, -.1, -1], [.1, .1, 1])
    second = box([-.1, 2.9, -1], [.1, 3.1, 1])
    assert collisions(segments, [first, second]) == {(0, 0), (1, 1)}
    assert collisions(segments, []) == set()


def test_normalization_and_offset_use_physical_distance():
    obstacle = box([-1]*3, [1]*3)
    scales = np.arange(1, 7)
    scaled = ConvexPolytope(obstacle.A * scales[:, None], obstacle.b * scales)
    unit = normalized(scaled)
    np.testing.assert_allclose(unit.A, obstacle.A)
    np.testing.assert_allclose(unit.b, obstacle.b)
    np.testing.assert_allclose(offset(scaled, .25).b, 1.25)
    np.testing.assert_allclose(offset(scaled, -.25).b, .75)
    assert offset(scaled, .25).contains([1.2, 0, 0])
    assert not offset(scaled, -.25).contains([.8, 0, 0])


@pytest.mark.parametrize("bounds", [([0, 0], [1, 1]), ([1, 0, 0], [0, 1, 1])])
def test_box_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="box requires"):
        box(*bounds)


@pytest.mark.parametrize("order", range(7))
def test_physical_derivatives_and_time_dilation_match_direct_curve(order):
    points = np.random.default_rng(11).normal(size=(2, 7, 3))
    trajectory = BMTPTrajectory(points, .7)
    times = np.array([0, .14, .69, .7, .91, 1.4])
    indices = np.array([0, 0, 0, 1, 1, 1])
    parameters = np.array([0, .2, .69/.7, 0, .3, 1])
    expected = np.array([_direct(points[i], u, order) / .7**order
                         for i, u in zip(indices, parameters, strict=True)])
    np.testing.assert_allclose(trajectory.evaluate(times, order), expected, atol=2e-7, rtol=1e-9)
    assert trajectory.evaluate(.7, order).shape == (3,)
    np.testing.assert_allclose(trajectory.evaluate(.7, order), expected[3], atol=2e-7)
    # Scale h by the same operation as t so a join stays exactly at the join.
    slower = BMTPTrajectory(points, trajectory.segment_time * 3)
    assert slower.duration == pytest.approx(3 * trajectory.duration)
    np.testing.assert_allclose(slower.evaluate(times * 3, order), expected / 3**order,
                               atol=2e-7, rtol=1e-9)


@pytest.mark.parametrize("dt", [.25, .3, 2.0])
def test_pva_samples_preserve_exact_ticks_and_hold_only_fractional_final_tick(dt):
    points = np.array([[[0, 0, 0], [1, 2, 0], [3, 1, 2]]], float)
    trajectory = BMTPTrajectory(points, 1)
    samples = trajectory.sample(dt)
    ticks = np.arange(int(np.ceil(1 / dt)) + 1) * dt
    assert samples.shape == (len(ticks), 9)
    for row, tick in zip(samples, ticks, strict=True):
        expected = np.concatenate([_direct(points[0], min(tick, 1), order)
                                   for order in range(3)])
        if tick > 1:
            expected[3:] = 0
        np.testing.assert_allclose(row, expected, atol=1e-12)
    np.testing.assert_array_equal(samples[-1, :3], points[0, -1])
    if dt == .25:
        assert np.linalg.norm(samples[-1, 3:6]) > 0
        assert np.linalg.norm(samples[-1, 6:9]) > 0


def test_trajectory_owns_control_points_and_handles_empty_evaluation():
    points = np.zeros((1, 3, 3))
    trajectory = BMTPTrajectory(points, 1)
    points[:] = 9
    np.testing.assert_array_equal(trajectory.control_points, 0)
    assert trajectory.evaluate([]).shape == (0, 3)


@pytest.mark.parametrize("time", [-.01, 1.01, np.nan, np.inf])
def test_trajectory_rejects_outside_or_nonfinite_time(time):
    with pytest.raises(ValueError, match="evaluation time"):
        BMTPTrajectory(np.zeros((1, 3, 3)), 1).evaluate(time)


@pytest.mark.parametrize("dt", [0, -1, np.inf, np.nan])
def test_sampling_rejects_invalid_period(dt):
    with pytest.raises(ValueError, match="dt"):
        BMTPTrajectory(np.zeros((1, 3, 3)), 1).sample(dt)


@pytest.mark.parametrize("points,h", [
    (np.zeros((3, 3)), 1), (np.zeros((1, 3, 2)), 1),
    (np.zeros((0, 3, 3)), 1), (np.zeros((1, 0, 3)), 1),
    (np.full((1, 3, 3), np.nan), 1), (np.full((1, 3, 3), np.inf), 1),
    (np.zeros((1, 3, 3)), 0), (np.zeros((1, 3, 3)), -1),
    (np.zeros((1, 3, 3)), np.inf), (np.zeros((1, 3, 3)), np.nan),
])
def test_trajectory_rejects_invalid_storage_and_duration(points, h):
    with pytest.raises(ValueError):
        BMTPTrajectory(points, h)


def test_constraint_residuals_report_physical_bound_violations(config):
    points = np.zeros((1, 7, 3))
    points[0, :, 0] = np.linspace(0, 2, 7)
    trajectory = BMTPTrajectory(points, .5)
    residuals = constraint_residuals(trajectory, np.zeros(3), np.array([3, 0, 0]),
                                    box([-1]*3, [1]*3),
                                    BMTPLimits(1, 1, None, None), config)
    assert residuals == pytest.approx(dict(boundary=1, workspace=1, continuity=0,
                                          terminal=4, derivative=3), abs=1e-12)


def test_constraint_residuals_include_position_discontinuity(config, limits):
    points = np.zeros((2, 7, 3))
    points[1, :, 0] = 2
    residuals = constraint_residuals(BMTPTrajectory(points, 1), points[0, 0], points[-1, -1],
                                    box([-3]*3, [3]*3), limits, config)
    assert residuals == pytest.approx(dict(boundary=0, workspace=0, continuity=2,
                                          terminal=0, derivative=0), abs=1e-12)


@pytest.mark.parametrize("bounded_order", [2, 4])
def test_no_obstacle_solve_is_fully_feasible_and_reaches_relaxed_bound(config, bounded_order):
    path = np.array([[0, 0, 0], [1, 1, .5], [2, 0, 1]], float)
    domain = box([-1]*3, [3]*3)
    limits = BMTPLimits(3, 5, 15, 30) if bounded_order == 4 else BMTPLimits(3, 5, None, None)
    result = BMTPPlanner(config).plan(path, [], domain, limits)
    assert result.success and result.converged, (result.status, result.message)
    assert result.initialization is not None
    _assert_feasible(result.trajectory, path, [], domain, limits, config)
    assert result.trajectory.duration == pytest.approx(result.lower_bound_duration, rel=1e-8)
    assert result.trajectory.duration <= result.initialization.duration + 1e-6
    assert result.history and all(step.accepted for step in result.history)
    for step in result.history:
        assert not step.collisions and not step.new_tags
        assert max(step.residuals.values()) <= config.feasibility_tolerance
        _assert_feasible(step.trajectory, path, [], domain, limits, config)
    assert all(np.isfinite(value) and value >= 0 for value in result.timings.values())
    assert result.timings["total_seconds"] > 0
    np.testing.assert_array_equal(result.initial_path, path)


@pytest.mark.parametrize("path", [
    [], [0, 0, 0], [[0, 0, 0]], [[0, 0], [1, 1]],
    [[0, 0, 0], [np.nan, 1, 0]], [[0, 0, 0], [1, np.inf, 0]],
    [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1e-10, 0, 0]],
    [[0, 0, 0], [3, 0, 0]],
])
def test_planner_rejects_invalid_path_before_solving(config, limits, path, monkeypatch):
    def unexpected_solve(*args, **kwargs):
        pytest.fail("invalid input reached the optimizer")

    monkeypatch.setattr(planner_module, "TrajectoryProgram", unexpected_solve)
    with pytest.raises(ValueError):
        BMTPPlanner(config).plan(np.asarray(path), [], box([-2]*3, [2]*3), limits)


@pytest.mark.parametrize("path", [
    [[-1, 0, 0], [1, 0, 0]],  # Both endpoints free, chord crosses obstacle.
    [[-1, .2, 0], [1, .2, 0]],  # Boundary contact.
    [[0, 0, 0], [1, 0, 0]],  # Start inside.
    [[-1, 0, 0], [0, 0, 0]],  # Goal inside.
])
def test_planner_rejects_colliding_initial_path(config, limits, path):
    with pytest.raises(ValueError, match="initial path collides"):
        BMTPPlanner(config).plan(np.asarray(path), [box([-.2]*3, [.2]*3)],
                                 box([-2]*3, [2]*3), limits)


def test_planner_rejects_unbounded_continuity_order(limits):
    with pytest.raises(ValueError, match="bounded derivative order"):
        BMTPPlanner().plan(np.array([[0, 0, 0], [1, 0, 0]]), [], box([-2]*3, [2]*3), limits)


@pytest.mark.parametrize("as_obstacle", [False, True])
def test_planner_rejects_zero_polytope_normal(config, limits, as_obstacle):
    bad = ConvexPolytope(np.zeros((6, 3)), np.ones(6))
    domain = box([-2]*3, [2]*3)
    with pytest.raises(ValueError, match="normals must be nonzero"):
        BMTPPlanner(config).plan(np.array([[0, 0, 0], [1, 0, 0]]),
                                 [bad] if as_obstacle else [], domain if as_obstacle else bad, limits)


@pytest.mark.parametrize("kwargs", [
    {"velocity": 0}, {"acceleration": -1}, {"jerk": np.inf},
    {"snap": np.nan}, {"jerk": None, "snap": 1},
])
def test_limits_reject_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        BMTPLimits(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"degree": 4}, {"continuity_order": 8}, {"terminal_order": -1},
    {"plane_degree": -1}, {"max_iterations": 0}, {"collision_max_depth": 0},
    {"solver_max_iterations": 0}, {"relative_tolerance": 0},
    {"collision_tolerance": -1}, {"trajectory_margin": np.nan},
    {"obstacle_margin": np.inf}, {"feasibility_tolerance": 0}, {"solver_tolerance": 0},
])
def test_config_rejects_invalid_orders_budgets_and_tolerances(kwargs):
    with pytest.raises(ValueError):
        BMTPConfig(**kwargs)


def test_one_update_budget_returns_feasible_seed_and_records_new_collision_tags(config, limits, detour):
    config = replace(config, max_iterations=1)
    path, obstacles, domain = detour
    result = BMTPPlanner(config).plan(path, obstacles, domain, limits)
    assert result.status == "max_iterations", result.message
    assert result.success and not result.converged
    assert result.trajectory is result.initialization
    assert len(result.history) == 1
    step = result.history[0]
    assert not step.accepted and step.collisions
    assert step.new_tags == step.collisions
    assert result.lower_bound_duration == pytest.approx(step.trajectory.duration)
    _assert_feasible(result.trajectory, path, obstacles, domain, limits, config)


@pytest.mark.parametrize("error_type", [cp.error.SolverError, RuntimeError, FloatingPointError])
def test_initialization_failure_returns_no_trajectory(config, limits, detour, monkeypatch, error_type):
    def fail(*args, **kwargs):
        raise error_type("injected initialization failure")

    monkeypatch.setattr(planner_module.TrajectoryProgram, "solve", fail)
    result = BMTPPlanner(config).plan(*detour, limits)
    assert result.status == "solver_failure"
    assert "injected initialization failure" in result.message
    assert not result.success and not result.converged
    assert result.trajectory is None and result.initialization is None
    assert result.history == []
    assert result.timings["total_seconds"] >= 0


@pytest.mark.parametrize("failure", ["solver", "residual", "duration"])
def test_update_failure_preserves_last_feasible_trajectory(config, limits, monkeypatch, failure):
    path = np.array([[0, 0, 0], [1, 0, 0]], float)
    domain = box([-2]*3, [2]*3)
    original = planner_module.TrajectoryProgram.solve
    seed = None

    def solve(program, start, goal, planes):
        nonlocal seed
        if seed is None:
            seed, elapsed, solver_time = original(program, start, goal, planes)
            return seed, elapsed, solver_time
        if failure == "solver":
            raise cp.error.SolverError("injected update failure")
        points = seed.control_points.copy()
        if failure == "residual":
            points[0, 3, 1] = 20
        return BMTPTrajectory(points, seed.segment_time * 2), 0., 0.

    monkeypatch.setattr(planner_module.TrajectoryProgram, "solve", solve)
    result = BMTPPlanner(config).plan(path, [], domain, limits)
    assert result.status == "solver_failure"
    expected = {"solver": "injected update failure", "residual": "constraint residual",
                "duration": "increased feasible duration"}
    assert expected[failure] in result.message
    assert result.success and not result.converged
    assert result.trajectory is result.initialization
    assert result.history == []
    _assert_feasible(result.trajectory, path, [], domain, limits, config)


@pytest.mark.parametrize("fail_update", [False, True])
def test_infeasible_seed_is_never_exposed_on_budget_or_solver_failure(config, detour, monkeypatch, fail_update):
    # Chords stop at p/v/a boundaries, but need not agree in jerk at joins.
    config = replace(config, degree=8, continuity_order=3, max_iterations=1)
    limits = BMTPLimits(10, 30, 100, None)
    path, obstacles, domain = detour
    if fail_update:
        original = planner_module.TrajectoryProgram.solve
        calls = 0

        def solve(program, start, goal, planes):
            nonlocal calls
            calls += 1
            if calls >= len(path):
                raise cp.error.SolverError("injected update failure")
            return original(program, start, goal, planes)

        monkeypatch.setattr(planner_module.TrajectoryProgram, "solve", solve)
    result = BMTPPlanner(config).plan(path, obstacles, domain, limits)
    assert result.status == ("solver_failure" if fail_update else "max_iterations"), result.message
    assert result.initialization is not None
    residuals = constraint_residuals(result.initialization, path[0], path[-1], domain, limits, config)
    assert residuals["continuity"] > config.feasibility_tolerance
    assert not result.success and not result.converged and result.trajectory is None
    assert len(result.history) == (0 if fail_update else 1)
    assert all(not step.accepted for step in result.history)


def test_plane_failure_preserves_feasible_seed(config, limits, detour, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected plane certification failure")

    monkeypatch.setattr(planner_module.PlaneProgram, "solve", fail)
    result = BMTPPlanner(config).plan(*detour, limits)
    assert result.status == "solver_failure"
    assert "injected plane certification failure" in result.message
    assert result.success and result.trajectory is result.initialization
    assert len(result.history) == 1 and not result.history[0].accepted
    _assert_feasible(result.trajectory, *detour, limits, config)


def test_official_guiding_scene_is_fully_feasible_with_published_duration():
    """User-supplied upstream regression: T in [2.40, 2.52] s; no scene imports.

    Construct the fixed-z geometry independently of simulator/experimental data.
    The published interval is an acceptance target, never fitted to local output.
    """
    domain = box([0, 0, 0], [10, 10, 0])
    obstacles = []
    for x, y, width, height in [
        (5.25, 2, 7.5, 1), (3, 5, 5, 1), (7.6, 4.1, .8, 1.5),
        (8.5, 6, 1.2, 1.2), (7, 8, 4.8, .8),
    ]:
        obstacles.append(box([x - width/2, y - height/2, -1],
                             [x + width/2, y + height/2, 1]))
    path = np.array([[1, 1, 0], [1, 3, 0], [9.5, 3, 0], [9.5, 7, 0],
                     [3, 7, 0], [3, 9, 0], [9, 9, 0]], float)
    config = BMTPConfig(degree=6, continuity_order=2, terminal_order=2,
                        plane_degree=1, relative_tolerance=.01, max_iterations=120,
                        collision_tolerance=1e-3, trajectory_margin=1e-3,
                        obstacle_margin=1e-9)
    limits = BMTPLimits(velocity=10, acceleration=30, jerk=None, snap=None)
    result = BMTPPlanner(config).plan(path, obstacles, domain, limits)
    diagnostic = (f"status={result.status}; message={result.message}; "
                  f"duration={result.trajectory.duration if result.success else None}; "
                  f"iterations={len(result.history)}; "
                  f"accepted_durations={[s.trajectory.duration for s in result.history if s.accepted]}")
    assert result.success, diagnostic
    _assert_feasible(result.trajectory, path, obstacles, domain, limits, config)
    assert result.initialization is not None, diagnostic
    _assert_feasible(result.initialization, path, obstacles, domain, limits, config)
    np.testing.assert_allclose(result.trajectory.control_points[:, :, 2], 0,
                               atol=config.feasibility_tolerance, rtol=0)
    accepted = [step for step in result.history if step.accepted]
    assert accepted, diagnostic
    for step in accepted:
        assert not step.collisions and not step.new_tags
        assert max(step.residuals.values()) <= config.feasibility_tolerance
        _assert_feasible(step.trajectory, path, obstacles, domain, limits, config)
    durations = [result.initialization.duration, *[step.trajectory.duration for step in accepted]]
    assert np.all(np.diff(durations) <= 1e-6), diagnostic
    assert result.trajectory.duration == pytest.approx(durations[-1], abs=1e-8)
    assert 2.40 <= result.trajectory.duration <= 2.52, diagnostic
    assert result.converged, diagnostic
    print(diagnostic)
