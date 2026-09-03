"""Geometry primitives shared by search, corridor and trajectory planning."""

from uav_ac.planning.geometry.collision import segment_intersects_aabb
from uav_ac.planning.geometry.ellipsoid import Ellipsoid
from uav_ac.planning.geometry.polytope import ConvexPolytope, enumerate_vertices
from uav_ac.planning.geometry.sampling import sample_path, sample_path_preserving_vertices

__all__ = [
    "ConvexPolytope",
    "Ellipsoid",
    "enumerate_vertices",
    "sample_path",
    "sample_path_preserving_vertices",
    "segment_intersects_aabb",
]
