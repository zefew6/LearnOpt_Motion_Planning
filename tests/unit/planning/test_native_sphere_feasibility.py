"""RRT boolean checks must preserve the signed-field reference semantics."""

import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF

native = pytest.importorskip('uav_ac.planning.native._collision_broadphase')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sphere_feasibility_matches_esdf_and_pair_reference(dtype):
    # Nonlinear grid samples exercise all eight interpolation corners.
    rng = np.random.default_rng(714)
    esdf = ESDF(rng.uniform(.05, 1.2, (9, 10, 11)).astype(dtype),
                np.array([-1., -1., -2.]), .3)
    points = rng.uniform(esdf.origin-.2, esdf.upper+.2, (300, 4, 3))
    radii = np.array([.05, .08, .12, .15])
    self_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    payload_pairs = np.array([[2, 3]], dtype=np.int64)
    clipped = np.clip(points, esdf.origin, esdf.upper)
    distances = esdf.distance(clipped.reshape(-1, 3)).reshape(300, 4)
    outside = np.linalg.norm(points-clipped, axis=2)
    valid = np.all(radii+.03+.02-distances+outside <= 1e-10, axis=1)
    valid &= np.all(points >= esdf.origin+radii[None, :, None]-1e-10, axis=(1, 2))
    valid &= np.all(points <= esdf.upper-radii[None, :, None]+1e-10, axis=(1, 2))
    for pairs in (self_pairs, payload_pairs):
        a, b = pairs.T
        valid &= np.all(radii[a]+radii[b]+.01-np.linalg.norm(
            points[:, a]-points[:, b], axis=2) <= 1e-10, axis=1)
    kernel = getattr(native, 'sphere_feasible', None)
    assert kernel is not None, 'native sphere/ESDF feasibility kernel is missing'
    actual = kernel(esdf.values, esdf.origin, esdf.resolution, points, radii,
                    self_pairs, payload_pairs, .03, .02, .01)
    np.testing.assert_array_equal(actual, valid)


def test_sphere_feasibility_contact_map_faces_and_empty_batch():
    kernel = getattr(native, 'sphere_feasible', None)
    assert kernel is not None, 'native sphere/ESDF feasibility kernel is missing'
    values = np.full((3, 3, 3), 10.)
    origin = np.zeros(3)
    radii = np.array([.25, .25])
    pairs = np.array([[0, 1]], dtype=np.int64)
    empty = np.empty((0, 2), dtype=np.int64)
    points = np.array([[[.25, 1., 1.], [.75, 1., 1.]],
                       [[.25, 1., 1.], [.749999, 1., 1.]],
                       [[.249999, 1., 1.], [1., 1., 1.]],
                       [[1.75, 1., 1.], [1., 1., 1.]],
                       [[1.750001, 1., 1.], [1., 1., 1.]]])
    np.testing.assert_array_equal(kernel(values, origin, 1., points, radii,
        pairs, empty, 0., 0., 0.), [True, False, False, True, False])
    assert kernel(values, origin, 1., points[:0], radii,
                  pairs, empty, 0., 0., 0.).shape == (0,)


def test_pair_tolerance_uses_original_distance_subtraction():
    radii = np.array([.15, .2])
    # Squaring a tolerance-shifted threshold can change a borderline decision.
    distance = .3599999999
    lengths = np.array([np.nextafter(distance, -np.inf), distance,
                        np.nextafter(distance, np.inf)])
    points = np.zeros((3, 2, 3))
    points[:, 1, 0] = lengths
    expected = radii.sum()+.01-lengths <= 1e-10
    actual = native.sphere_feasible(np.full((3, 3, 3), 10.), np.full(3, -1.), 1.,
        points, radii, np.array([[0, 1]], dtype=np.int64),
        np.empty((0, 2), dtype=np.int64), 0., 0., .01)
    np.testing.assert_array_equal(actual, expected)


def test_esdf_reports_tolerance_ambiguity_for_reference_confirmation():
    rng = np.random.default_rng(16)
    field = ESDF(rng.uniform(.2, .8, (12, 12, 12)), np.full(3, -5.), 1.)
    point = np.array([[[-.43112103, 1.86500622, -.71720249]]])
    distance = field.distance(point[0])[0]
    radius = np.array([distance-.05+1e-10])
    empty = np.empty((0, 2), dtype=np.int64)
    valid, ambiguous = native.sphere_feasible(field.values, field.origin, 1., point,
        radius, empty, empty, .03, .02, 0., with_uncertainty=True)
    np.testing.assert_array_equal(ambiguous, [True])
    assert valid.shape == (1,)
