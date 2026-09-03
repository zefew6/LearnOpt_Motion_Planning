"""Reusable MuJoCo rendering adapter for translucent convex corridors."""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from uav_ac.planning.corridor.firi import FIRIRegion

REGION_COUNT = 40
TRIANGLE_COUNT = 128
COLORS = np.array([
    [0.10, 0.80, 1.00, 0.09], [0.35, 1.00, 0.35, 0.09],
    [1.00, 0.75, 0.10, 0.09], [0.90, 0.30, 1.00, 0.09],
])


def add_corridor_mesh_pool(specification: mujoco.MjSpec) -> None:
    """Preallocate fixed-topology triangle soups for runtime FIRI surfaces."""
    vertices = np.empty((TRIANGLE_COUNT, 3, 3))
    for index in range(TRIANGLE_COUNT):
        angle = 2.0 * np.pi * index / TRIANGLE_COUNT
        center = np.array([0.02*np.cos(angle), 0.02*np.sin(angle),
                           -20.0 + 0.001*(index % 3)])
        vertices[index] = center + np.array(
            [[0.0, 0.0, 0.0], [1.0e-3, 0.0, 0.0], [0.0, 1.0e-3, 0.0]])
    faces = np.arange(3 * TRIANGLE_COUNT).reshape(-1, 3)
    for index in range(REGION_COUNT):
        mesh_name = f"corridor_mesh_{index:03d}"
        specification.add_mesh(
            name=mesh_name, uservert=vertices.ravel(), userface=faces.ravel(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL, maxhullvert=4)
        specification.worldbody.add_geom(
            name=f"corridor_region_{index:03d}", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name, contype=0, conaffinity=0,
            rgba=[0.2, 0.8, 1.0, 0.0], group=2)


def triangulate_region(region: FIRIRegion) -> np.ndarray:
    """Return consistently outward-facing triangles for a convex region."""
    vertices = region.vertices()
    if len(vertices) < 4:
        raise ValueError("a displayed convex region must have at least four vertices")
    hull = ConvexHull(vertices)
    triangles = vertices[hull.simplices].copy()
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    inward = np.sum(normals * hull.equations[:, :3], axis=1) < 0.0
    triangles[inward, 1], triangles[inward, 2] = (
        triangles[inward, 2].copy(), triangles[inward, 1].copy())
    return triangles


class CorridorMeshVisualizer:
    """Own the compiled-model mesh slots used for convex corridor display."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.region_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"corridor_region_{index:03d}")
            for index in range(REGION_COUNT)])
        self.mesh_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH,
                              f"corridor_mesh_{index:03d}")
            for index in range(REGION_COUNT)])
        if np.any(self.region_ids < 0) or np.any(self.mesh_ids < 0):
            raise ValueError("compiled model is missing the corridor mesh pool")
        model.mesh_pos[self.mesh_ids] = 0.0
        model.mesh_quat[self.mesh_ids] = np.array([1.0, 0.0, 0.0, 0.0])

    def set_regions(self, regions: list[FIRIRegion], coordinate_map: np.ndarray) -> int:
        if len(regions) > REGION_COUNT:
            raise ValueError(f"at most {REGION_COUNT} convex regions can be displayed")
        self.model.geom_rgba[self.region_ids, 3] = 0.0
        for index, region in enumerate(regions):
            triangles = triangulate_region(region)
            if len(triangles) > TRIANGLE_COUNT:
                raise ValueError(f"region {index} needs {len(triangles)} triangles, "
                                 f"but only {TRIANGLE_COUNT} are available")
            mesh_id = self.mesh_ids[index]
            vertex_start = self.model.mesh_vertadr[mesh_id]
            normal_start = self.model.mesh_normaladr[mesh_id]
            vertex_count = self.model.mesh_vertnum[mesh_id]
            vertices = self.model.mesh_vert[vertex_start:vertex_start + vertex_count]
            normals = self.model.mesh_normal[normal_start:normal_start + vertex_count]
            vertices[:] = np.array([0.0, 0.0, -100.0])
            normals[:] = np.array([0.0, 0.0, 1.0])
            triangle_vertices = triangles.reshape(-1, 3) @ coordinate_map
            vertices[:len(triangle_vertices)] = triangle_vertices
            triangle_normals = np.cross(triangle_vertices[1::3] - triangle_vertices[0::3],
                                        triangle_vertices[2::3] - triangle_vertices[0::3])
            triangle_normals /= np.linalg.norm(triangle_normals, axis=1, keepdims=True)
            normals[:len(triangle_vertices)] = np.repeat(triangle_normals, 3, axis=0)
            geom_id = self.region_ids[index]
            self.model.geom_pos[geom_id] = 0.0
            self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0])
            self.model.geom_rbound[geom_id] = max(
                1.0, float(np.max(np.linalg.norm(triangle_vertices, axis=1))))
            self.model.geom_rgba[geom_id] = COLORS[index % len(COLORS)]
        return len(regions)


__all__ = ["CorridorMeshVisualizer", "add_corridor_mesh_pool", "triangulate_region"]
