import numpy as np
import pytest

from uav_ac.planning.geometry.grid_map import GridMap


def context(grid, backend='auto', empty_pairs=False):
    from uav_ac.planning.geometry.collision_broadphase import RRTBroadphase

    return RRTBroadphase(
        grid, radii=np.array([.05, .06, .04, .12]),
        self_pairs=np.empty((0, 2), dtype=int) if empty_pairs else np.array([[0, 1], [1, 2]]),
        payload_pairs=np.empty((0, 2), dtype=int) if empty_pairs else np.array([[1, 3]]),
        self_query_indices=np.empty(0, dtype=int) if empty_pairs else np.array([0, 1]),
        payload_query_indices=np.empty(0, dtype=int) if empty_pairs else np.array([2]),
        sphere_geom_indices=np.array([0, 0, 1, -1]),
        world_lower=np.array([[-.2, -.1, -.3], [.1, .2, .1], [-np.inf, -np.inf, 0.]]),
        world_upper=np.array([[.1, .3, .2], [.5, .4, .4], [np.inf, np.inf, 0.]]),
        world_query_table=np.array([[3, 4, 5], [6, -1, 7]]),
        query_count=8, self_clearance=.02, backend=backend)


@pytest.mark.parametrize('empty_pairs', [False, True])
@pytest.mark.parametrize('narrow_phase', [False, True])
def test_native_broadphase_matches_reference_at_random_and_boundary_points(empty_pairs, narrow_phase):
    pytest.importorskip('uav_ac.planning.geometry._collision_broadphase')
    occupied=np.zeros((6, 6, 6), dtype=bool)
    occupied[2:4, 2, 3]=True
    grid=GridMap(occupied, np.full(3, -.5), .2)
    reference=context(grid, 'numpy', empty_pairs)
    native=context(grid, 'native', empty_pairs)
    rng=np.random.default_rng(51)
    points=rng.uniform(-.8, .8, (80, 4, 3))
    # Exact half-cell ties on both sides of the map; includes index -0.5 and 5.5.
    ties=grid.origin+grid.resolution*np.array([-.5, .5, 1.5, 5.5])[:, None]
    points[:4]=ties[:, None, :]
    for margin in (0., .03, .4):
        expected=reference.query(points, margin, .05, narrow_phase=narrow_phase)
        actual=native.query(points, margin, .05, narrow_phase=narrow_phase)
        for index,(a,b) in enumerate(zip(actual, expected, strict=True)):
            if index in (1, 2):
                np.testing.assert_allclose(a,b,rtol=0.,atol=1e-15)
            else:
                np.testing.assert_array_equal(a,b)
        assert actual[3].dtype == bool
        for index, positions in enumerate(points):
            hits,outside=grid.collision_mask(positions, reference.radii, margin)
            assert actual[0][index] == (1. if np.any(hits | outside) else -1.)
            assert actual[5][index] == np.any(outside)


def test_broadphase_rejects_nonfinite_and_malformed_query_points():
    grid=GridMap(np.zeros((3, 3, 3), dtype=bool), np.zeros(3), .2)
    engine=context(grid, 'numpy')
    with pytest.raises(ValueError, match='points'):
        engine.query(np.zeros((3, 3)), .01, .01)
    with pytest.raises(ValueError, match='points'):
        engine.query(np.full((4, 3), np.nan), .01, .01)
    with pytest.raises(ValueError, match='margin'):
        engine.query(np.zeros((4, 3)), -.1, .01)
    empty=engine.query(np.empty((0, 4, 3)), .01, .01)
    assert empty[0].shape == (0,)
    assert empty[3].shape == (0, 8)


def test_broadphase_metadata_cannot_change_after_native_index_validation():
    grid=GridMap(np.zeros((3, 3, 3), dtype=bool), np.zeros(3), .2)
    engine=context(grid, 'numpy')
    with pytest.raises(AttributeError):
        engine.geom_indices=np.full(4, 999, dtype=int)
    with pytest.raises(ValueError):
        engine.world_table[0, 0]=999
