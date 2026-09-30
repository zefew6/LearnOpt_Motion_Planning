import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.geometry.grid_map import GridMap


def test_trilinear_esdf_recovers_linear_field_and_gradient():
    axes = np.arange(5, dtype=float)
    xx, yy, zz = np.meshgrid(axes, axes, axes, indexing="ij")
    field = 2*xx-3*yy+.5*zz+4
    esdf = ESDF(field, np.zeros(3), 1.0)
    query = np.array([[.31, .42, .68], [.75, .25, .5]])
    distance, gradient = esdf.distance_and_gradient(query)
    np.testing.assert_allclose(distance, 2*query[:, 0]-3*query[:, 1]+.5*query[:, 2]+4)
    np.testing.assert_allclose(gradient, np.tile([2., -3., .5], (2, 1)), atol=1e-12)


def test_box_esdf_and_outside_grid_behavior():
    esdf = ESDF.from_axis_aligned_boxes(
        np.array([[0., 0., 0., 1., 1., 1.]]),
        np.array([-1., -1., -1.]), np.array([2., 2., 2.]), .25)
    assert esdf.values.dtype == np.float32
    distance, _ = esdf.distance_and_gradient(np.array([[1.5, .5, .5]]))
    assert distance[0] == pytest.approx(.5)
    with pytest.raises(ValueError, match="outside"):
        esdf.distance(np.array([[3., 0., 0.]]))


def test_inflated_occupancy_grid_uses_batched_voxel_boxes():
    occupied = np.zeros((5, 5, 5), dtype=bool)
    occupied[2, 2, 2] = True
    grid = GridMap(occupied, np.zeros(3), 1.0)
    hits, outside = grid.collision_mask(
        np.array([[2., 2., 2.], [0., 0., 0.], [4., 4., 4.], [5., 2., 2.]]),
        np.zeros(4), 0.0)
    np.testing.assert_array_equal(hits, [True, False, False, False])
    np.testing.assert_array_equal(outside, [False, False, False, True])


def test_inflated_occupancy_grid_inflates_each_query_radius():
    occupied = np.zeros((7, 7, 7), dtype=bool)
    occupied[3, 3, 3] = True
    grid = GridMap(occupied, np.zeros(3), 1.0)
    hits, outside = grid.collision_mask(
        np.array([[1., 3., 3.], [1., 3., 3.]]),
        np.array([0.0, 1.0]), 0.0)
    np.testing.assert_array_equal(hits, [False, True])
    assert not np.any(outside)


def test_occupancy_grid_can_rasterize_boxes_without_distance_evaluation():
    boxes = np.array([[1., 1., 1., 2., 2., 2.]])
    grid = GridMap.from_axis_aligned_boxes(
        boxes, np.zeros(3), np.full(3, 3.), .5, ground_height=None)
    assert grid.occupied[2, 2, 2]
    assert not grid.occupied[0, 0, 0]
