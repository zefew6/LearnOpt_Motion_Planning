import numpy as np

from uav_ac.planning.geometry.esdf import InflatedOccupancyGrid
from uav_ac.scenes.loader import SceneGeometry


def _geometry(kind, center, rotation, size, name):
    return SceneGeometry(name, kind, np.asarray(center, dtype=float),
                         np.asarray(rotation, dtype=float), np.asarray(size, dtype=float), 1, 1)


def test_scene_primitives_are_conservatively_voxelized_in_local_frames():
    angle = np.pi/4
    rotated = np.array([
        [np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.],
        [0., 0., 1.],
    ])
    geometries = (
        _geometry("box", [0., 0., 0.], rotated, [.08, .42, .12], "obstacle_rotated"),
        _geometry("sphere", [1., 0., 0.], np.eye(3), [.20], "obstacle_sphere"),
        _geometry("cylinder", [2., 0., 0.], np.eye(3), [.24, .35], "obstacle_cylinder"),
        _geometry("plane", [0., 0., -1.], np.eye(3), [], "ground"),
    )
    grid = InflatedOccupancyGrid.from_scene_geometries(
        geometries, [-1., -1., -2.], [3., 1., .5], .1)
    indices = np.rint((np.array([
        [0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 0., -1.0],
    ])-grid.origin)/grid.resolution).astype(int)
    assert np.all(grid.occupied[tuple(indices.T)])
    # This point lies along the rotated box's long axis, outside its tiny AABB
    # half-width in either horizontal coordinate.
    rotated_point = np.array([[.23, .23, 0.]])
    cell = np.rint((rotated_point[0]-grid.origin)/grid.resolution).astype(int)
    assert grid.occupied[tuple(cell)]
