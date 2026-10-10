"""GCOPTER safe-corridor trajectory planner orchestration.

This module ports the central ideas in ZJU FAST Lab's MIT-licensed GCOPTER:

* non-uniform, minimum-control-effort (MINCO) quintic splines;
* unconstrained positive segment-time mapping;
* unconstrained spatial variables mapped into convex corridor polytopes; and
* joint limited-memory BFGS optimization with integrated soft constraints.

The original ROS front end, map implementation, nonlinear-drag flatness map and
visualization are deliberately outside this module.  The point-mass flatness
penalties here cover velocity, acceleration, thrust, tilt and body rate.
Corridor geometry follows this project's convention ``A @ position <= b``.

The GCOPTER source used as the reference is Copyright (c) 2021 Zhepei Wang and
is distributed under the MIT License.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...geometry import enumerate_vertices as _enumerate_vertices
from .config import GCOPTERConfig
from .mappings import (
    backward_polytope_point as _backward_polytope_point,
    backward_time_gradient as _backward_time_gradient,
    convex_weights_batch as _convex_weights_batch,
    forward_polytope_point as _forward_polytope_point,
    forward_time as _forward_time,
    inverse_time as _inverse_time,
    polynomial_bases as _polynomial_bases,
    smoothed_l1_array as _smoothed_l1_array,
)
from .optimization import evaluate_minco_objective
from .minco import MINCOQuintic as _MINCOQuintic
from .optimizer import scipy_lbfgs as _scipy_lbfgs
from .penalties import (stack_piece_halfspaces as _stack_piece_halfspaces,
                        integrated_penalty as _integrated_penalty)
from .types import (
    GCOPTERTrajectory,
    HalfSpaceRegion,
    TrajectorySamples,
    coerce_region as _coerce_region,
)


class GCOPTER:
    """Optimize jerk energy plus weighted flight time inside a FIRI corridor."""

    def __init__(self, config: GCOPTERConfig | None = None):
        self.config = GCOPTERConfig() if config is None else config
        self._last_penalty_cost = np.inf

    def plan(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            corridor: Sequence[HalfSpaceRegion | tuple[np.ndarray, np.ndarray]],
            *,
            start_velocity: np.ndarray | None = None,
            start_acceleration: np.ndarray | None = None,
            goal_velocity: np.ndarray | None = None,
            goal_acceleration: np.ndarray | None = None,
            fixed_corridor_boundaries: Sequence[tuple[int, np.ndarray]] | None = None,
    ) -> GCOPTERTrajectory:
        """Jointly optimize segment durations and corridor-constrained waypoints."""
        self._last_penalty_cost = np.inf
        start = _vector3(start, "start")
        goal = _vector3(goal, "goal")
        if len(corridor) == 0:
            raise ValueError("corridor must contain at least one convex region")
        h_polytopes = [self._normalized_region(region) for region in corridor]
        if np.any(h_polytopes[0][0] @ start > h_polytopes[0][1] + 1.0e-7):
            raise ValueError("start must lie in the first corridor region")
        if np.any(h_polytopes[-1][0] @ goal > h_polytopes[-1][1] + 1.0e-7):
            raise ValueError("goal must lie in the last corridor region")

        v_polytopes = self._process_corridor(h_polytopes)
        short_path = self._shortest_path(start, goal, v_polytopes)
        lengths = np.linalg.norm(np.diff(short_path, axis=0), axis=1)
        piece_counts = np.floor(lengths / self.config.length_per_piece).astype(int) + 1
        piece_count = int(np.sum(piece_counts))
        desired_points, initial_times = self._set_initial(short_path, piece_counts)

        point_poly_indices, corridor_indices = self._piece_indices(piece_counts, v_polytopes)
        fixed_points: dict[int, np.ndarray] = {}
        if fixed_corridor_boundaries is not None:
            cumulative_pieces = np.cumsum(piece_counts)
            for boundary_index, waypoint in fixed_corridor_boundaries:
                if not 0 <= boundary_index < len(h_polytopes) - 1:
                    raise ValueError("fixed corridor boundary index is out of range")
                waypoint = _vector3(waypoint, "fixed corridor waypoint")
                left_A, left_b = h_polytopes[boundary_index]
                right_A, right_b = h_polytopes[boundary_index + 1]
                if (np.any(left_A @ waypoint > left_b + 1.0e-7)
                        or np.any(right_A @ waypoint > right_b + 1.0e-7)):
                    raise ValueError("fixed waypoint must lie in both adjacent corridor regions")
                point_index = int(cumulative_pieces[boundary_index] - 1)
                fixed_points[point_index] = waypoint
        variable_point_indices = np.asarray([
            index for index in range(piece_count - 1) if index not in fixed_points
        ], dtype=int)
        variable_poly_indices = point_poly_indices[variable_point_indices]
        xi, xi_slices = self._encode_points(
            desired_points[variable_point_indices], variable_poly_indices, v_polytopes)
        tau = _inverse_time(initial_times)
        x0 = np.concatenate((tau, xi))

        head_pva = np.stack((
            start,
            _optional_vector3(start_velocity, "start_velocity"),
            _optional_vector3(start_acceleration, "start_acceleration"),
        ))
        tail_pva = np.stack((
            goal,
            _optional_vector3(goal_velocity, "goal_velocity"),
            _optional_vector3(goal_acceleration, "goal_acceleration"),
        ))
        minco = _MINCOQuintic(head_pva, tail_pva, piece_count)
        piece_halfspaces = _stack_piece_halfspaces(corridor_indices, h_polytopes)

        def objective(variables: np.ndarray) -> tuple[float, np.ndarray]:
            return self._objective(
                variables, minco, piece_count, xi_slices,
                variable_point_indices, variable_poly_indices, fixed_points,
                v_polytopes, piece_halfspaces,
            )

        result = _scipy_lbfgs(
            objective,
            x0,
            max_iterations=self.config.max_iterations,
            memory=self.config.lbfgs_memory,
            gradient_tolerance=self.config.gradient_tolerance,
            relative_cost_tolerance=self.config.relative_cost_tolerance,
            is_feasible=lambda: self._last_penalty_cost <= 1.0e-10,
            feasible_iteration_patience=self.config.feasible_iteration_patience,
        )
        durations = _forward_time(result.x[:piece_count])
        points = self._decode_points(
            result.x[piece_count:], xi_slices, variable_point_indices,
            variable_poly_indices, fixed_points, piece_count - 1, v_polytopes)
        flat_coefficients, _ = minco.solve(points, durations)
        coefficients = flat_coefficients.reshape(piece_count, 6, 3)
        return GCOPTERTrajectory(
            durations=durations,
            coefficients=coefficients,
            corridor_indices=corridor_indices,
            cost=result.cost,
            iterations=result.iterations,
            converged=result.converged,
            message=result.message,
        )

    def _objective(
            self,
            variables: np.ndarray,
            minco: _MINCOQuintic,
            piece_count: int,
            xi_slices: list[slice],
            variable_point_indices: np.ndarray,
            variable_poly_indices: np.ndarray,
            fixed_points: dict[int, np.ndarray],
            v_polytopes: list[np.ndarray],
            piece_halfspaces: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[float, np.ndarray]:
        tau = variables[:piece_count]
        xi = variables[piece_count:]
        times = _forward_time(tau)
        if np.min(times) < self.config.minimum_piece_time:
            return 1.0e30, np.zeros_like(variables)
        points = self._decode_points(
            xi, xi_slices, variable_point_indices, variable_poly_indices,
            fixed_points, piece_count - 1, v_polytopes)
        def term(durations, blocks):
            gc = np.zeros_like(blocks).reshape(-1, 3)
            gt = np.zeros(len(durations))
            value = self._integrated_penalty(durations, blocks, piece_halfspaces, gc, gt)
            self._last_penalty_cost = value
            return value, gc.reshape(blocks.shape), gt

        try:
            cost, grad_points, grad_times, _ = evaluate_minco_objective(
                minco, points, times, terms=(term,), time_weight=self.config.time_weight)
        except (np.linalg.LinAlgError, ValueError):
            return 1.0e30, np.zeros_like(variables)

        grad_xi = np.zeros_like(xi)
        for segment, point_index, poly_index in zip(
                xi_slices, variable_point_indices, variable_poly_indices, strict=True):
            q = xi[segment]
            grad_xi[segment] = _backward_polytope_point(
                q, v_polytopes[int(poly_index)], grad_points[point_index])
            norm_violation = float(np.dot(q, q) - 1.0)
            if norm_violation > 0.0:
                restriction = norm_violation**3
                cost += restriction
                grad_xi[segment] += 6.0 * norm_violation**2 * q

        gradient = np.concatenate((_backward_time_gradient(tau, grad_times), grad_xi))
        if not np.isfinite(cost) or not np.all(np.isfinite(gradient)):
            return 1.0e30, np.zeros_like(variables)
        return float(cost), gradient

    def _integrated_penalty(self, times, coefficients, piece_halfspaces, grad_coefficients, grad_times):
        return _integrated_penalty(times, coefficients, piece_halfspaces,
                                   grad_coefficients, grad_times, self.config)

    @staticmethod
    def _normalized_region(
            region: HalfSpaceRegion | tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        A, b = _coerce_region(region)
        norms = np.linalg.norm(A, axis=1)
        if np.any(norms <= 1.0e-12):
            raise ValueError("corridor contains a zero half-space normal")
        return A / norms[:, None], b / norms

    @staticmethod
    def _process_corridor(
            h_polytopes: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray]:
        v_polytopes: list[np.ndarray] = []
        for index, (A, b) in enumerate(h_polytopes):
            vertices = _enumerate_vertices(A, b)
            if len(vertices) < 4:
                raise ValueError(f"corridor region {index} is empty or unbounded")
            v_polytopes.append(vertices)
            if index < len(h_polytopes) - 1:
                next_A, next_b = h_polytopes[index + 1]
                overlap = _enumerate_vertices(
                    np.vstack((A, next_A)), np.concatenate((b, next_b)))
                if len(overlap) < 4:
                    raise ValueError(f"corridor regions {index} and {index + 1} do not overlap")
                v_polytopes.append(overlap)
        return v_polytopes

    def _shortest_path(
            self, start: np.ndarray, goal: np.ndarray, v_polytopes: list[np.ndarray],
    ) -> np.ndarray:
        overlap_polytopes = v_polytopes[1::2]
        if not overlap_polytopes:
            return np.stack((start, goal))
        slices: list[slice] = []
        offset = 0
        values = []
        for polytope in overlap_polytopes:
            size = len(polytope)
            slices.append(slice(offset, offset + size))
            values.append(np.full(size, np.sqrt(1.0 / size)))
            offset += size
        x0 = np.concatenate(values)

        def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
            inner = np.stack([
                _forward_polytope_point(x[segment], polytope)
                for segment, polytope in zip(slices, overlap_polytopes, strict=True)
            ])
            path = np.vstack((start, inner, goal))
            deltas = np.diff(path, axis=0)
            lengths = np.sqrt(np.sum(deltas * deltas, axis=1) + self.config.smoothing_epsilon)
            cost = float(np.sum(lengths))
            grad_points = np.zeros_like(inner)
            for index in range(len(inner)):
                grad_points[index] = deltas[index] / lengths[index] - deltas[index + 1] / lengths[index + 1]
            gradient = np.zeros_like(x)
            for index, (segment, polytope) in enumerate(
                    zip(slices, overlap_polytopes, strict=True)):
                gradient[segment] = _backward_polytope_point(
                    x[segment], polytope, grad_points[index])
            return cost, gradient

        result = _scipy_lbfgs(
            objective, x0, max_iterations=min(80, self.config.max_iterations),
            memory=min(8, self.config.lbfgs_memory), gradient_tolerance=1.0e-6,
            relative_cost_tolerance=1.0e-5,
        )
        inner = np.stack([
            _forward_polytope_point(result.x[segment], polytope)
            for segment, polytope in zip(slices, overlap_polytopes, strict=True)
        ])
        return np.vstack((start, inner, goal))

    def _set_initial(
            self, short_path: np.ndarray, piece_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = []
        times = []
        # The original full-flatness optimizer starts at 3 * v_max and quickly
        # expands infeasible times.  This point-mass port starts from a feasible
        # speed scale because its softer dynamics model otherwise exhausts the
        # iteration budget before satisfying the velocity bound.
        allocation_speed = self.config.max_velocity
        for index, count in enumerate(piece_counts):
            start, goal = short_path[index:index + 2]
            delta = (goal - start) / count
            duration = max(np.linalg.norm(delta) / allocation_speed, self.config.minimum_piece_time * 10.0)
            times.extend([duration] * int(count))
            for subdivision in range(int(count)):
                if index > 0 or subdivision > 0:
                    points.append(start + delta * subdivision)
        return np.asarray(points, dtype=float).reshape(-1, 3), np.asarray(times)

    @staticmethod
    def _piece_indices(
            piece_counts: np.ndarray, v_polytopes: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        point_indices = []
        corridor_indices = []
        region_count = len(piece_counts)
        for region, count in enumerate(piece_counts):
            for local_piece in range(int(count)):
                corridor_indices.append(region)
                if local_piece < count - 1:
                    point_indices.append(2 * region)
                elif region < region_count - 1:
                    point_indices.append(2 * region + 1)
        if len(v_polytopes) != 2 * region_count - 1:
            raise RuntimeError("internal corridor indexing mismatch")
        return np.asarray(point_indices, dtype=int), np.asarray(corridor_indices, dtype=int)

    def _encode_points(
            self,
            points: np.ndarray,
            poly_indices: np.ndarray,
            v_polytopes: list[np.ndarray],
    ) -> tuple[np.ndarray, list[slice]]:
        encoded: list[np.ndarray | None] = [None] * len(points)
        slices = []
        offset = 0
        for poly_index in poly_indices:
            vertices = v_polytopes[int(poly_index)]
            slices.append(slice(offset, offset + len(vertices)))
            offset += len(vertices)
        for poly_index in np.unique(poly_indices):
            point_indices = np.flatnonzero(poly_indices == poly_index)
            vertices = v_polytopes[int(poly_index)]
            weights = _convex_weights_batch(
                points[point_indices], vertices, self.config.inverse_map_iterations)
            for point_index, point_weights in zip(point_indices, weights, strict=True):
                encoded[point_index] = np.concatenate(
                    (np.sqrt(point_weights[1:]), np.sqrt(point_weights[:1])))
        return (
            np.concatenate([value for value in encoded if value is not None])
            if encoded else np.zeros(0),
            slices,
        )

    @staticmethod
    def _decode_points(
            xi: np.ndarray,
            slices: list[slice],
            point_indices: np.ndarray,
            poly_indices: np.ndarray,
            fixed_points: dict[int, np.ndarray],
            point_count: int,
            v_polytopes: list[np.ndarray],
    ) -> np.ndarray:
        points = np.empty((point_count, 3), dtype=float)
        for point_index, point in fixed_points.items():
            points[point_index] = point
        for segment, point_index, poly_index in zip(
                slices, point_indices, poly_indices, strict=True):
            points[point_index] = _forward_polytope_point(
                xi[segment], v_polytopes[int(poly_index)])
        return points


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return vector


def _optional_vector3(value: np.ndarray | None, name: str) -> np.ndarray:
    return np.zeros(3) if value is None else _vector3(value, name)


__all__ = ["GCOPTER", "GCOPTERConfig", "GCOPTERTrajectory", "TrajectorySamples"]
