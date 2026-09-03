"""High-level GCS relaxation, rounding and convex-restriction planner."""

import numpy as np

from ...geometry import ConvexPolytope
from .config import GCSConfig
from .formulation import solve_relaxation, solve_restriction
from .graph import GCSGraph
from .types import GCSTrajectory


class GCSPlanner:
    """Plan a smooth geometric trajectory through a union of convex sets."""

    def __init__(self, config: GCSConfig | None = None):
        self.config = GCSConfig() if config is None else config

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        regions: list[ConvexPolytope],
    ) -> GCSTrajectory:
        graph = GCSGraph.from_regions(
            regions, start, goal, tolerance=self.config.feasibility_tolerance)
        return self.plan_graph(graph, start, goal)

    def plan_graph(
        self,
        graph: GCSGraph,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> GCSTrajectory:
        start = _vector(start, "start")
        goal = _vector(goal, "goal")
        relaxation = solve_relaxation(graph, start, goal, self.config)
        edge_path = graph.round_flow(relaxation.flows, self.config.flow_tolerance)
        vertex_path = graph.vertex_path(edge_path)
        restriction = solve_restriction(
            graph, vertex_path, start, goal, self.config)
        physical = [
            index for index, vertex in enumerate(vertex_path)
            if vertex < graph.physical_region_count
        ]
        if not physical:
            raise RuntimeError("rounded GCS path visits no physical free-space region")
        control_points = restriction.control_points[physical]
        region_indices = np.asarray([vertex_path[index] for index in physical], dtype=int)
        return GCSTrajectory(
            control_points=control_points,
            region_indices=region_indices,
            vertex_path=vertex_path,
            edge_path=edge_path,
            relaxed_objective=relaxation.objective,
            objective=restriction.objective,
            relaxation_iterations=relaxation.iterations,
            restriction_iterations=restriction.iterations,
            status=restriction.status,
        )

def _vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return value


__all__ = ["GCSPlanner"]
