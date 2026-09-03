"""Convex-set graph construction and deterministic flow rounding."""

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.optimize import linprog

from ...geometry import ConvexPolytope


@dataclass(frozen=True)
class GCSGraph:
    """Directed graph whose vertices own bounded convex polytopes."""

    regions: tuple[ConvexPolytope, ...]
    edges: tuple[tuple[int, int], ...]
    source: int
    target: int
    physical_region_count: int

    def __post_init__(self) -> None:
        count = len(self.regions)
        if not 0 <= self.source < count or not 0 <= self.target < count:
            raise ValueError("source and target must be graph vertices")
        if self.source == self.target:
            raise ValueError("source and target must differ")
        if not 0 <= self.physical_region_count <= count:
            raise ValueError("invalid physical_region_count")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("duplicate GCS edges are not supported")
        for tail, head in self.edges:
            if not (0 <= tail < count and 0 <= head < count) or tail == head:
                raise ValueError(f"invalid GCS edge {(tail, head)}")

    @classmethod
    def from_regions(
        cls,
        regions: list[ConvexPolytope],
        start: np.ndarray,
        goal: np.ndarray,
        *,
        tolerance: float = 1.0e-7,
    ) -> "GCSGraph":
        """Build overlap edges and point vertices for a free-space cover."""
        if not regions:
            raise ValueError("at least one convex region is required")
        start = _vector(start, "start")
        goal = _vector(goal, "goal")
        physical = tuple(regions)
        source = len(physical)
        target = source + 1
        graph_regions = physical + (_point_polytope(start), _point_polytope(goal))
        edges: list[tuple[int, int]] = []
        for first in range(len(physical)):
            for second in range(first + 1, len(physical)):
                if polytopes_intersect(physical[first], physical[second], tolerance):
                    edges.extend(((first, second), (second, first)))
        start_regions = [i for i, region in enumerate(physical) if region.contains(start, tolerance)]
        goal_regions = [i for i, region in enumerate(physical) if region.contains(goal, tolerance)]
        if not start_regions:
            raise ValueError("start is not contained in any GCS region")
        if not goal_regions:
            raise ValueError("goal is not contained in any GCS region")
        edges.extend((source, index) for index in start_regions)
        edges.extend((index, target) for index in goal_regions)
        return cls(graph_regions, tuple(edges), source, target, len(physical)).pruned()

    def pruned(self) -> "GCSGraph":
        """Remove vertices and edges that cannot lie on a source-target path."""
        forward = _reachable(self.source, self.edges)
        reverse = _reachable(self.target, tuple((v, u) for u, v in self.edges))
        keep = forward & reverse
        if self.target not in keep:
            raise ValueError("the GCS region graph does not connect start to goal")
        edges = tuple((u, v) for u, v in self.edges if u in keep and v in keep)
        return GCSGraph(self.regions, edges, self.source, self.target,
                        self.physical_region_count)

    def round_flow(self, flows: np.ndarray, tolerance: float = 1.0e-9) -> tuple[int, ...]:
        """Return edge indices of the maximum-likelihood source-target path."""
        flows = np.asarray(flows, dtype=float)
        if flows.shape != (len(self.edges),):
            raise ValueError("flows must have one value per edge")
        outgoing: dict[int, list[tuple[float, int, int]]] = {}
        for edge_index, ((tail, head), flow) in enumerate(zip(self.edges, flows, strict=True)):
            if flow > tolerance:
                cost = -np.log(max(float(flow), tolerance))
                outgoing.setdefault(tail, []).append((cost, edge_index, head))
        queue = [(0.0, self.source, ())]
        best = {self.source: 0.0}
        while queue:
            cost, vertex, path = heapq.heappop(queue)
            if vertex == self.target:
                return path
            if cost > best.get(vertex, np.inf) + 1.0e-12:
                continue
            for edge_cost, edge_index, head in outgoing.get(vertex, []):
                candidate = cost + edge_cost
                if candidate + 1.0e-12 < best.get(head, np.inf):
                    best[head] = candidate
                    heapq.heappush(queue, (candidate, head, path + (edge_index,)))
        raise ValueError("relaxed GCS flow contains no source-target path")

    def vertex_path(self, edge_path: tuple[int, ...]) -> tuple[int, ...]:
        if not edge_path:
            raise ValueError("edge path cannot be empty")
        vertices = [self.source]
        for edge_index in edge_path:
            tail, head = self.edges[edge_index]
            if tail != vertices[-1]:
                raise ValueError("edge path is not contiguous")
            vertices.append(head)
        if vertices[-1] != self.target:
            raise ValueError("edge path does not terminate at target")
        return tuple(vertices)


def polytopes_intersect(
    first: ConvexPolytope, second: ConvexPolytope, tolerance: float = 1.0e-7,
) -> bool:
    first_lower, first_upper = _axis_aligned_bounds(first)
    second_lower, second_upper = _axis_aligned_bounds(second)
    if np.any(first_upper + tolerance < second_lower) or np.any(
            second_upper + tolerance < first_lower):
        return False
    A = np.vstack((first.A, second.A))
    b = np.concatenate((first.b, second.b)) + tolerance
    result = linprog(np.zeros(3), A_ub=A, b_ub=b,
                     bounds=[(None, None)] * 3, method="highs")
    return bool(result.success)


def _axis_aligned_bounds(polytope: ConvexPolytope) -> tuple[np.ndarray, np.ndarray]:
    """Extract cheap conservative bounds from any axis-aligned half-spaces."""
    lower = np.full(3, -np.inf)
    upper = np.full(3, np.inf)
    for normal, offset in zip(polytope.A, polytope.b, strict=True):
        axis = int(np.argmax(np.abs(normal)))
        if abs(normal[axis]) < 1.0e-12:
            continue
        other = np.delete(normal, axis)
        if np.max(np.abs(other), initial=0.0) > 1.0e-10:
            continue
        bound = offset / normal[axis]
        if normal[axis] > 0.0:
            upper[axis] = min(upper[axis], bound)
        else:
            lower[axis] = max(lower[axis], bound)
    return lower, upper


def _point_polytope(point: np.ndarray) -> ConvexPolytope:
    return ConvexPolytope(np.vstack((np.eye(3), -np.eye(3))),
                          np.concatenate((point, -point)))


def _reachable(start: int, edges: tuple[tuple[int, int], ...]) -> set[int]:
    outgoing: dict[int, list[int]] = {}
    for tail, head in edges:
        outgoing.setdefault(tail, []).append(head)
    reached = {start}
    stack = [start]
    while stack:
        for head in outgoing.get(stack.pop(), []):
            if head not in reached:
                reached.add(head)
                stack.append(head)
    return reached


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return value


__all__ = ["GCSGraph", "polytopes_intersect"]
