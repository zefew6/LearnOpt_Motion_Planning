import numpy as np
import pytest

from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic, BandedPLU
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_constraints
from uav_ac.planning.trajectory.gcopter.optimization import DerivativeWaypoint
from uav_ac.planning.trajectory.gcopter.optimization import integrate_samples
from uav_ac.planning.trajectory.gcopter.optimization import FixedWaypointMap
from uav_ac.planning.trajectory.gcopter.optimization import boundary_derivative_jacobian


def test_fixed_components_are_removed_from_decision_vector():
    points = np.arange(16., dtype=float).reshape(2, 8)
    fixed = np.zeros_like(points, dtype=bool)
    fixed[0, :3] = True
    mapping = FixedWaypointMap(points, fixed)
    x = mapping.encode(points)
    assert x.size == 13
    restored = mapping.decode(x + 1.)
    np.testing.assert_array_equal(restored[fixed], points[fixed])
    np.testing.assert_array_equal(restored[~fixed], points[~fixed] + 1.)
    np.testing.assert_array_equal(mapping.pullback(np.ones_like(points)), np.ones(13))
    with pytest.raises(ValueError):
        mapping.decode(np.zeros(12))


@pytest.mark.parametrize('dimensions', [3, 8])
def test_shared_objective_waypoint_and_duration_gradients(dimensions):
    rng = np.random.default_rng(43)
    head, tail = np.zeros((3, dimensions)), np.zeros((3, dimensions))
    tail[0] = .5
    minco = MINCOQuintic(head, tail, 2)
    points = rng.normal(size=(1, dimensions)) * .2
    durations = np.array([.8, 1.3])

    def sample(piece, alpha, values):
        grads = np.zeros((4, dimensions))
        grads[0] = 2 * values[0]
        grads[1] = .4 * values[1]
        return np.dot(values[0], values[0]) + .2 * np.dot(values[1], values[1]), grads

    def term(times, coefficients):
        return integrate_samples(times, coefficients, 7, sample)

    cost, gp, gt, coefficients = evaluate_minco_objective(
        minco, points, durations, terms=(term,), time_weight=.3)
    x = np.r_[points.ravel(), durations]
    grad = np.r_[gp.ravel(), gt]
    for i in range(len(x)):
        delta = np.zeros_like(x); delta[i] = 1e-6
        def f(z):
            return evaluate_minco_objective(minco, z[:-2].reshape(1, dimensions),
                z[-2:], terms=(term,), time_weight=.3)[0]
        np.testing.assert_allclose(grad[i], (f(x+delta)-f(x-delta))/2e-6,
                                   rtol=2e-5, atol=2e-5)
    assert np.isfinite(cost)
    assert coefficients.shape == (2, 6, dimensions)


def test_all_fixed_and_single_piece_maps():
    mapping = FixedWaypointMap(np.empty((0, 3)), np.empty((0, 3), dtype=bool))
    assert mapping.encode(np.empty((0, 3))).size == 0
    assert mapping.decode(np.empty(0)).shape == (0, 3)
    mapping = FixedWaypointMap(np.ones((1, 3)), np.ones((1, 3), dtype=bool))
    np.testing.assert_array_equal(mapping.decode(np.empty(0)), np.ones((1, 3)))


def test_fixed_boundary_jerk_duration_and_state_pullback():
    head = np.zeros((3, 3)); tail = head.copy(); tail[0] = 1.
    points = np.full((1, 3), .3); times = np.array([.7, 1.1])
    minco = MINCOQuintic(head, tail, 2)
    values, gp, gt, gh, ge = boundary_derivative_jacobian(minco, points, times, 3, [0, 1])
    for i in range(2):
        delta = np.zeros(2); delta[i] = 1e-6
        plus = boundary_derivative_jacobian(minco, points, times+delta, 3, [0, 1])[0]
        minus = boundary_derivative_jacobian(minco, points, times-delta, 3, [0, 1])[0]
        np.testing.assert_allclose(gt[:, i], (plus-minus)/2e-6, rtol=1e-6)
    assert gp.shape == (4, 1, 3)


def test_banded_solve_does_not_mutate_fortran_rhs():
    system = BandedPLU(np.diag([2., 3., 4.]), 1, 1)
    rhs = np.asfortranarray(np.arange(6., dtype=float).reshape(3, 2))
    before = rhs.copy()
    first = system.solve(rhs, transpose=True)
    np.testing.assert_array_equal(rhs, before)
    np.testing.assert_allclose(first, system.solve(rhs, transpose=True))


def test_shared_objective_rejects_nonfinite_data():
    minco = MINCOQuintic(np.zeros((3, 3)), np.zeros((3, 3)), 2)
    with pytest.raises(ValueError, match='finite'):
        evaluate_minco_objective(minco, np.full((1, 3), np.nan), np.ones(2))


def test_internal_velocity_constraint_has_shared_adjoint():
    head = np.zeros((3, 3)); tail = head.copy(); tail[0] = 1.
    minco = MINCOQuintic(head, tail, 2)
    points, times = np.full((1, 3), .4), np.array([.7, 1.2])
    constraint = DerivativeWaypoint(1, 1, np.array([.2, .3, .4]), np.array([True, False, True]))
    residual, gp, gt, gh, ge = evaluate_minco_constraints(minco, points, times, constraint)
    assert residual.shape == (2,)
    epsilon = 1e-6
    plus, minus = points.copy(), points.copy(); plus[0, 0] += epsilon; minus[0, 0] -= epsilon
    numeric = (evaluate_minco_constraints(minco, plus, times, constraint)[0]
               -evaluate_minco_constraints(minco, minus, times, constraint)[0])/(2*epsilon)
    np.testing.assert_allclose(gp[:, 0, 0], numeric, rtol=1e-6)
