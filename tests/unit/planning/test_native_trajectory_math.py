"""Native/reference parity gates; build native kernels explicitly before running."""
import numpy as np
import pytest

from uav_ac.planning.trajectory.gcopter import mappings, minco


def test_reference_backends_remain_available():
    assert callable(getattr(mappings, '_polynomial_basis_matrix_reference', None))
    assert callable(getattr(mappings, '_evaluate_piecewise_quintic_reference', None))
    assert callable(getattr(minco.MINCOQuintic, '_band_storage_reference', None))
    assert callable(getattr(minco.MINCOQuintic, '_jerk_energy_reference', None))
    assert callable(getattr(minco.MINCOQuintic, '_propagate_gradient_reference', None))


@pytest.fixture
def native():
    return pytest.importorskip('uav_ac.planning.native._trajectory_math')


@pytest.mark.parametrize('dimensions', [1, 3, 8])
@pytest.mark.parametrize('pieces', [1, 2, 11])
def test_native_minco_matches_reference(native, monkeypatch, dimensions, pieces):
    rng = np.random.default_rng(720 + pieces + dimensions)
    solver = minco.MINCOQuintic(rng.normal(size=(3, dimensions)),
                              rng.normal(size=(3, dimensions)), pieces)
    times = rng.uniform(.3, 1.4, pieces)
    points = rng.normal(size=(pieces-1, dimensions))
    monkeypatch.setattr(minco, '_native_math', None)
    coefficients_ref, system_ref = solver.solve(points, times)
    energy_ref = solver.jerk_energy(coefficients_ref, times, np.arange(dimensions)+.5)
    grad_c = rng.normal(size=coefficients_ref.shape)
    grad_t = rng.normal(size=pieces)
    adjoint_ref = solver.propagate_gradient(system_ref, coefficients_ref, times, grad_c.copy(), grad_t)
    monkeypatch.setattr(minco, '_native_math', native)
    np.testing.assert_allclose(solver._band_storage(times), solver._band_storage_reference(times), rtol=2e-15)
    coefficients, system = solver.solve(points, times)
    np.testing.assert_allclose(coefficients, coefficients_ref, rtol=1e-10, atol=1e-8)
    for actual, expected in zip(solver.jerk_energy(coefficients, times, np.arange(dimensions)+.5), energy_ref):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-7)
    for actual, expected in zip(solver.propagate_gradient(system, coefficients, times, grad_c.copy(), grad_t), adjoint_ref):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-7)


@pytest.mark.parametrize('derivative', range(6))
def test_polynomials_clipping_boundaries_empty_and_strides(native, monkeypatch, derivative):
    rng = np.random.default_rng(92)
    durations = np.array([.3, .7, 1.1])
    coefficients = rng.normal(size=(3, 6, 16))[:, :, ::2]
    queries = [np.array([[-1., 0., .3], [1., 2.1, 3.]]), np.array([]), .3]
    for query in queries:
        monkeypatch.setattr(mappings, '_native_math', None)
        expected = mappings.evaluate_piecewise_quintic(durations, coefficients, query, derivative)
        monkeypatch.setattr(mappings, '_native_math', native)
        actual = mappings.evaluate_piecewise_quintic(durations, coefficients, query, derivative)
        np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-13)
    times = np.linspace(-.4, 1.3, 23)[::2]
    np.testing.assert_allclose(mappings.polynomial_basis_matrix(times, derivative),
                               mappings._polynomial_basis_matrix_reference(times, derivative), rtol=3e-15)
    np.testing.assert_allclose(mappings.polynomial_bases(times),
                               mappings._polynomial_bases_reference(times), rtol=3e-15)


def test_native_adjoint_matches_finite_difference(native, monkeypatch):
    monkeypatch.setattr(minco, '_native_math', native)
    rng = np.random.default_rng(15)
    solver = minco.MINCOQuintic(np.zeros((3, 8)), rng.normal(size=(3, 8)), 3)
    points, times = rng.normal(size=(2, 8)), np.array([.6, .8, 1.1])
    weights = rng.uniform(.2, 2., 8)
    def objective(p, t):
        coefficients, system = solver.solve(p, t)
        cost, gc, gt = solver.jerk_energy(coefficients, t, weights)
        return cost, solver.propagate_gradient(system, coefficients, t, gc.reshape(-1, 8), gt)
    _, (gp, gt) = objective(points, times)
    eps = 1e-6
    for index in range(len(times)):
        plus, minus = times.copy(), times.copy()
        plus[index] += eps; minus[index] -= eps
        fd = (objective(points, plus)[0]-objective(points, minus)[0])/(2*eps)
        np.testing.assert_allclose(gt[index], fd, rtol=5e-6, atol=.01)
    for index in [(0, 0), (1, 7)]:
        plus, minus = points.copy(), points.copy()
        plus[index] += eps; minus[index] -= eps
        fd = (objective(plus, times)[0]-objective(minus, times)[0])/(2*eps)
        np.testing.assert_allclose(gp[index], fd, rtol=5e-6, atol=.01)


def test_native_facades_reject_unsafe_shapes(native, monkeypatch):
    monkeypatch.setattr(minco, '_native_math', native)
    monkeypatch.setattr(mappings, '_native_math', native)
    solver = minco.MINCOQuintic(np.zeros((3, 8)), np.ones((3, 8)), 2)
    with pytest.raises(ValueError):
        solver._band_storage(np.ones(1))
    with pytest.raises(ValueError):
        solver.jerk_energy(np.ones((2, 6, 8)), np.ones(2), np.ones(7))
    with pytest.raises(ValueError):
        mappings.polynomial_basis_matrix(np.ones(3), 6)
    with pytest.raises(ValueError):
        native.evaluate_quintic(np.ones((2, 6, 8)), np.array([2], dtype=np.intp), np.ones(1), 0)
    with pytest.raises(ValueError):
        native.adjoint_gradients(np.ones((2, 6, 8)), np.ones(2), np.ones((11, 8)), np.ones(2))


def test_float32_jerk_gradient_preserves_reference_dtype(native, monkeypatch):
    monkeypatch.setattr(minco, '_native_math', native)
    coefficients = np.ones((2, 6, 8), dtype=np.float32)
    times = np.array([.7, .9])
    actual = minco.MINCOQuintic.jerk_energy(coefficients, times)
    expected = minco.MINCOQuintic._jerk_energy_reference(coefficients, times)
    assert actual[1].dtype == np.float32
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_native_jerk_coefficient_and_time_gradients(native, monkeypatch):
    monkeypatch.setattr(minco, '_native_math', native)
    rng = np.random.default_rng(120)
    coefficients = rng.normal(size=(3, 6, 8))
    times = np.array([.3, .7, 1.1])
    weights = rng.uniform(0., 1.5, 8)
    weights[3] = 0.
    cost, gc, gt = minco.MINCOQuintic.jerk_energy(coefficients, times, weights)
    assert gc.shape == (48, 3)  # Historical public layout is deliberate in ND.
    eps = 1e-6
    for index in [(0, 0, 2), (0, 3, 1), (1, 4, 3), (2, 5, 7)]:
        plus, minus = coefficients.copy(), coefficients.copy()
        plus[index] += eps; minus[index] -= eps
        fd = (minco.MINCOQuintic.jerk_energy(plus, times, weights)[0]
              - minco.MINCOQuintic.jerk_energy(minus, times, weights)[0])/(2*eps)
        np.testing.assert_allclose(gc.reshape(coefficients.shape)[index], fd, rtol=1e-6, atol=1e-6)
    for index in range(3):
        plus, minus = times.copy(), times.copy()
        plus[index] += eps; minus[index] -= eps
        fd = (minco.MINCOQuintic.jerk_energy(coefficients, plus, weights)[0]
              - minco.MINCOQuintic.jerk_energy(coefficients, minus, weights)[0])/(2*eps)
        np.testing.assert_allclose(gt[index], fd, rtol=1e-6, atol=1e-6)
    assert np.isfinite(cost)
