import numpy as np

from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.planning.trajectory.gcopter.mappings import polynomial_basis


def test_minco_solves_eight_dimensions_and_preserves_boundary_conditions():
    head = np.zeros((3, 8))
    tail = np.zeros((3, 8)); tail[0, :3] = [1.0, -0.5, 0.2]; tail[0, 3] = 3.2
    minco = MINCOQuintic(head, tail, 3)
    points = np.array([[.2, .1, 0, 3.0, .2, -.1, .1, 0],
                       [.7, -.2, .1, 3.15, .1, .1, -.1, .2]])
    times = np.array([.7, .8, .9])
    coefficients, _ = minco.solve(points, times)
    blocks = coefficients.reshape(3, 6, 8)
    np.testing.assert_allclose(blocks[0, 0], head[0], atol=1e-8)
    np.testing.assert_allclose(np.array([1, .9, .9**2, .9**3, .9**4, .9**5]) @ blocks[-1],
                               tail[0], atol=1e-7)
    np.testing.assert_allclose(np.array([1, .7, .7**2, .7**3, .7**4, .7**5]) @ blocks[0],
                               points[0], atol=1e-7)


def test_weighted_jerk_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    times = np.array([.4, .8])
    coefficients = rng.normal(size=(2, 6, 8))
    weights = np.arange(1.0, 9.0)
    cost, gradient, grad_times = MINCOQuintic.jerk_energy(coefficients, times, weights)
    flat = coefficients.ravel()
    eps = 1e-6
    for index in (0, 5, 20, 47, 95):
        shifted = flat.copy(); shifted[index] += eps
        plus = MINCOQuintic.jerk_energy(shifted.reshape(2, 6, 8), times, weights)[0]
        shifted[index] -= 2*eps
        minus = MINCOQuintic.jerk_energy(shifted.reshape(2, 6, 8), times, weights)[0]
        np.testing.assert_allclose(gradient.ravel()[index], (plus-minus)/(2*eps), rtol=2e-6, atol=2e-6)
    assert np.isfinite(cost)
    assert grad_times.shape == (2,)


def test_minco_single_piece_supports_eight_dimensions():
    head = np.zeros((3, 8)); tail = np.zeros((3, 8)); tail[0] = np.arange(8)
    coefficients, _ = MINCOQuintic(head, tail, 1).solve(np.empty((0, 8)), np.array([2.0]))
    np.testing.assert_allclose(coefficients.reshape(1, 6, 8)[0, 0], head[0])


def test_minco_preserves_pva_boundaries_and_internal_c4_continuity():
    rng = np.random.default_rng(11)
    head = np.zeros((3, 8))
    tail = np.zeros((3, 8))
    head[:, :3] = rng.normal(size=(3, 3))
    tail[:, :3] = rng.normal(size=(3, 3))
    points = rng.normal(size=(2, 8))
    times = np.array([.45, .7, .9])
    coefficients, _ = MINCOQuintic(head, tail, 3).solve(points, times)
    blocks = coefficients.reshape(3, 6, 8)
    np.testing.assert_allclose(
        np.stack([polynomial_basis(0., order) @ blocks[0] for order in range(3)]),
        head, atol=1e-8)
    np.testing.assert_allclose(
        np.stack([polynomial_basis(times[-1], order) @ blocks[-1]
                  for order in range(3)]), tail, atol=1e-7)
    for index in range(2):
        left = np.stack([polynomial_basis(times[index], order) @ blocks[index]
                         for order in range(5)])
        right = np.stack([polynomial_basis(0., order) @ blocks[index+1]
                          for order in range(5)])
        np.testing.assert_allclose(left, right, atol=2e-7)
