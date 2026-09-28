import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF


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
    distance, _ = esdf.distance_and_gradient(np.array([[1.5, .5, .5]]))
    assert distance[0] == pytest.approx(.5)
    with pytest.raises(ValueError, match="outside"):
        esdf.distance(np.array([[3., 0., 0.]]))
