import math

import numpy as np
import pytest

from uav_ac.planning.trajectory.gcopter import GCOPTER, GCOPTERConfig
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic as _MINCOQuintic
from uav_ac.planning.trajectory.gcopter.penalties import (
    stack_piece_halfspaces as _stack_piece_halfspaces,
)
from uav_ac.planning.trajectory.gcopter.mappings import (
    convex_weights,
    convex_weights_batch,
)


def _box(lower, upper):
    return np.vstack((np.eye(3), -np.eye(3))), np.concatenate((upper, -lower))


def test_convex_inverse_mapping_should_be_exact_and_nonnegative():
    vertices = np.array([
        [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
    ])
    point = np.array([0.73, -0.41, 0.22])

    weights = convex_weights(point, vertices)

    assert np.all(weights >= -1.0e-10)
    assert np.sum(weights) == pytest.approx(1.0, abs=1.0e-12)
    assert weights @ vertices == pytest.approx(point, abs=1.0e-10)


def test_batched_inverse_mapping_should_match_original_projected_iterations():
    rng = np.random.default_rng(3)
    vertices = rng.normal(size=(12, 3))
    target_weights = rng.dirichlet(np.ones(len(vertices)), size=5)
    points = target_weights @ vertices

    batched = convex_weights_batch(points, vertices)

    assert np.all(batched >= 0.0)
    assert np.sum(batched, axis=1) == pytest.approx(np.ones(len(points)))
    assert batched @ vertices == pytest.approx(points, abs=2.0e-3)


def test_gcopter_should_optimize_quintic_trajectory_in_one_polytope():
    corridor = [_box(np.array([-1.0, -1.0, -1.0]), np.array([4.0, 1.0, 1.0]))]

    trajectory = GCOPTER().plan(np.zeros(3), np.array([3.0, 0.0, 0.0]), corridor)
    samples = trajectory.sample(0.02)

    assert trajectory.converged
    assert trajectory.duration > 0.0
    assert trajectory.piece_count == 4
    assert samples.positions[0] == pytest.approx([0.0, 0.0, 0.0])
    assert samples.positions[-1] == pytest.approx([3.0, 0.0, 0.0])
    assert samples.velocities[[0, -1]] == pytest.approx(np.zeros((2, 3)), abs=1e-8)
    assert samples.accelerations[[0, -1]] == pytest.approx(np.zeros((2, 3)), abs=1e-8)
    assert trajectory.maximum_corridor_violation(corridor) <= 1e-6
    assert np.max(np.linalg.norm(samples.velocities, axis=1)) < 3.0
    assert np.max(np.linalg.norm(samples.accelerations, axis=1)) < 6.0


def test_gcopter_should_pass_between_overlapping_polytopes():
    corridor = [
        _box(np.array([-1.0, -1.0, -1.0]), np.array([2.0, 1.0, 1.0])),
        _box(np.array([1.0, -1.0, -1.0]), np.array([4.0, 1.0, 1.0])),
    ]

    trajectory = GCOPTER(GCOPTERConfig(integral_resolution=8)).plan(
        np.zeros(3), np.array([3.0, 0.0, 0.0]), corridor)

    assert trajectory.converged
    assert trajectory.corridor_indices.tolist() == [0, 0, 1, 1]
    assert trajectory.maximum_corridor_violation(corridor) <= 1e-6
    assert trajectory.evaluate(0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert trajectory.evaluate(trajectory.duration) == pytest.approx([3.0, 0.0, 0.0])


def test_gcopter_should_fix_mission_waypoint_at_corridor_boundary():
    corridor = [
        _box(np.array([-1.0, -1.0, -1.0]), np.array([2.0, 1.0, 1.0])),
        _box(np.array([1.0, -1.0, -1.0]), np.array([4.0, 1.0, 1.0])),
    ]
    waypoint = np.array([1.5, 0.25, 0.0])

    trajectory = GCOPTER().plan(
        np.zeros(3), np.array([3.0, 0.0, 0.0]), corridor,
        fixed_corridor_boundaries=[(0, waypoint)])
    boundary_piece_count = np.count_nonzero(trajectory.corridor_indices == 0)
    boundary_time = np.sum(trajectory.durations[:boundary_piece_count])

    assert trajectory.evaluate(boundary_time) == pytest.approx(waypoint, abs=1e-8)


def test_gcopter_flatness_constraints_should_respect_vehicle_limits():
    corridor = [_box(np.array([-1.0, -1.0, -1.0]), np.array([4.0, 1.0, 1.0]))]
    config = GCOPTERConfig(
        mass=0.5, gravity=9.81, min_thrust=0.4, max_thrust=18.0,
        max_tilt_angle=0.7, max_body_rate=2.1)

    trajectory = GCOPTER(config).plan(
        np.zeros(3), np.array([3.0, 0.0, 0.0]), corridor)
    samples = trajectory.sample(0.01)
    force = np.column_stack((
        -samples.accelerations[:, 0],
        -samples.accelerations[:, 1],
        config.gravity - samples.accelerations[:, 2],
    ))
    force_norm = np.linalg.norm(force, axis=1)
    body_z = force / force_norm[:, None]
    thrust = config.mass * force_norm
    tilt = np.arccos(np.clip(body_z[:, 2], -1.0, 1.0))
    projected_jerk = samples.jerks - body_z * np.sum(
        body_z * samples.jerks, axis=1)[:, None]
    body_rate = np.linalg.norm(projected_jerk, axis=1) / force_norm

    assert np.min(thrust) >= config.min_thrust
    assert np.max(thrust) <= config.max_thrust
    assert np.max(tilt) <= config.max_tilt_angle
    assert np.max(body_rate) <= config.max_body_rate


def test_gcopter_should_reject_invalid_corridor_and_queries():
    planner = GCOPTER()
    box = _box(-np.ones(3), np.ones(3))

    with pytest.raises(ValueError, match="first corridor"):
        planner.plan(np.array([2.0, 0.0, 0.0]), np.zeros(3), [box])
    with pytest.raises(ValueError, match="do not overlap"):
        planner.plan(
            np.zeros(3), np.array([4.0, 0.0, 0.0]),
            [box, _box(np.array([3.0, -1.0, -1.0]), np.array([5.0, 1.0, 1.0]))],
        )

    trajectory = planner.plan(np.zeros(3), np.array([0.5, 0.0, 0.0]), [box])
    with pytest.raises(ValueError, match="derivative"):
        trajectory.evaluate(0.0, 6)
    with pytest.raises(ValueError, match="dt"):
        trajectory.sample(0.0)


def test_gcopter_config_should_validate_numerical_settings():
    with pytest.raises(ValueError, match="continuous"):
        GCOPTERConfig(max_velocity=0.0)
    with pytest.raises(ValueError, match="iteration"):
        GCOPTERConfig(integral_resolution=1)
    with pytest.raises(ValueError, match="iteration"):
        GCOPTERConfig(feasible_iteration_patience=0)
    with pytest.raises(ValueError, match="iteration"):
        GCOPTERConfig(inverse_map_iterations=0)
    with pytest.raises(ValueError, match="thrust bounds"):
        GCOPTERConfig(min_thrust=2.0, max_thrust=1.0)


def test_minco_banded_plu_and_analytic_gradient_should_match_dense_reference():
    rng = np.random.default_rng(11)
    pieces = 5
    head = rng.normal(size=(3, 3))
    tail = rng.normal(size=(3, 3))
    points = rng.normal(size=(pieces - 1, 3))
    times = rng.uniform(0.5, 1.5, size=pieces)
    minco = _MINCOQuintic(head, tail, pieces)
    coefficients, system = minco.solve(points, times)

    rhs = np.zeros((6 * pieces, 3))
    rhs[:3] = head
    rhs[-3:] = tail
    rhs[np.arange(pieces - 1) * 6 + 5] = points
    assert coefficients == pytest.approx(
        np.linalg.solve(minco._matrix(times), rhs), abs=1.0e-10)

    grad_coefficients = rng.normal(size=coefficients.shape)
    direct_grad_times = rng.normal(size=pieces)
    _, grad_times = minco.propagate_gradient(
        system, coefficients, times, grad_coefficients, direct_grad_times)

    def functional(test_times):
        test_coefficients, _ = minco.solve(points, test_times)
        return (np.sum(test_coefficients * grad_coefficients)
                + np.dot(test_times, direct_grad_times))

    step = 1.0e-6
    finite_difference = np.empty_like(times)
    for index in range(pieces):
        plus, minus = times.copy(), times.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            functional(plus) - functional(minus)) / (2.0 * step)
    assert grad_times == pytest.approx(finite_difference, rel=1.0e-7, abs=2.0e-7)


def test_minco_rows_should_follow_original_constraint_order():
    head = np.array([
        [0.2, -0.3, 0.4],
        [0.5, 0.1, -0.2],
        [-0.1, 0.3, 0.2],
    ])
    tail = np.array([
        [2.1, 0.4, -0.6],
        [-0.2, 0.5, 0.1],
        [0.3, -0.4, 0.2],
    ])
    points = np.array([[0.8, -0.1, 0.2], [1.5, 0.3, -0.4]])
    times = np.array([0.7, 1.1, 0.9])
    coefficients, _ = _MINCOQuintic(head, tail, 3).solve(points, times)
    pieces = coefficients.reshape(3, 6, 3)

    def derivative(piece, time, order):
        powers = np.arange(order, 6)
        factors = np.array([
            math.factorial(power) / math.factorial(power - order)
            for power in powers
        ])
        basis = factors * time ** (powers - order)
        return basis @ piece[order:]

    assert np.stack([derivative(pieces[0], 0.0, order) for order in range(3)]) \
        == pytest.approx(head, abs=1.0e-10)
    for index in range(2):
        left = pieces[index]
        right = pieces[index + 1]
        duration = times[index]
        # MINCO's six interior rows are J, S, waypoint, P, V, A.
        assert derivative(left, duration, 3) == pytest.approx(
            derivative(right, 0.0, 3), abs=1.0e-9)
        assert derivative(left, duration, 4) == pytest.approx(
            derivative(right, 0.0, 4), abs=1.0e-9)
        assert derivative(left, duration, 0) == pytest.approx(
            points[index], abs=1.0e-10)
        for order in range(3):
            assert derivative(left, duration, order) == pytest.approx(
                derivative(right, 0.0, order), abs=1.0e-9)
    assert np.stack([
        derivative(pieces[-1], times[-1], order) for order in range(3)
    ]) == pytest.approx(tail, abs=1.0e-9)


def test_vectorized_penalty_gradient_should_match_finite_difference():
    rng = np.random.default_rng(19)
    times = np.array([0.8, 1.1])
    coefficients = rng.normal(scale=0.35, size=(12, 3))
    A, b = _box(-np.full(3, 0.4), np.full(3, 0.4))
    halfspaces = _stack_piece_halfspaces(np.zeros(2, dtype=int), [(A, b)])
    planner = GCOPTER(GCOPTERConfig(integral_resolution=6))

    def evaluate(test_times, test_coefficients):
        grad_coefficients = np.zeros_like(test_coefficients)
        grad_times = np.zeros_like(test_times)
        cost = planner._integrated_penalty(
            test_times, test_coefficients, halfspaces,
            grad_coefficients, grad_times)
        return cost, grad_coefficients, grad_times

    _, analytic_coefficients, analytic_times = evaluate(times, coefficients)
    step = 1.0e-6
    finite_coefficients = np.empty_like(coefficients)
    for row, axis in np.ndindex(coefficients.shape):
        plus, minus = coefficients.copy(), coefficients.copy()
        plus[row, axis] += step
        minus[row, axis] -= step
        finite_coefficients[row, axis] = (
            evaluate(times, plus)[0] - evaluate(times, minus)[0]) / (2.0 * step)
    finite_times = np.empty_like(times)
    for index in range(len(times)):
        plus, minus = times.copy(), times.copy()
        plus[index] += step
        minus[index] -= step
        finite_times[index] = (
            evaluate(plus, coefficients)[0] - evaluate(minus, coefficients)[0]
        ) / (2.0 * step)

    assert analytic_coefficients == pytest.approx(
        finite_coefficients, rel=2.0e-7, abs=2.0e-4)
    assert analytic_times == pytest.approx(finite_times, rel=2.0e-7, abs=4.0e-4)
