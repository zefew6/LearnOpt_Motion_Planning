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
from .minco import MINCOQuintic as _MINCOQuintic
from .optimizer import scipy_lbfgs as _scipy_lbfgs
from .penalties import stack_piece_halfspaces as _stack_piece_halfspaces
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
        try:
            coefficients, system = minco.solve(points, times)
        except np.linalg.LinAlgError:
            return 1.0e30, np.zeros_like(variables)
        if not np.all(np.isfinite(coefficients)):
            return 1.0e30, np.zeros_like(variables)

        cost, grad_coefficients, direct_grad_times = minco.jerk_energy(coefficients, times)
        penalty_cost = self._integrated_penalty(
            times, coefficients, piece_halfspaces,
            grad_coefficients, direct_grad_times,
        )
        self._last_penalty_cost = penalty_cost
        cost += penalty_cost + self.config.time_weight * float(np.sum(times))
        direct_grad_times += self.config.time_weight
        grad_points, grad_times = minco.propagate_gradient(
            system, coefficients, times, grad_coefficients, direct_grad_times)

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

    def _integrated_penalty(
            self,
            times: np.ndarray,
            coefficients: np.ndarray,
            piece_halfspaces: tuple[np.ndarray, np.ndarray, np.ndarray],
            grad_coefficients: np.ndarray,
            grad_times: np.ndarray,
    ) -> float:
        resolution = self.config.integral_resolution
        fraction = 1.0 / resolution
        piece_count = len(times)
        sample_count = resolution + 1
        blocks = coefficients.reshape(piece_count, 6, 3)
        alpha = np.arange(sample_count, dtype=float) * fraction
        local_times = (times[:, None] * alpha[None, :]).reshape(-1)
        bases = _polynomial_bases(local_times).reshape(5, piece_count, sample_count, 6)
        position, velocity, acceleration, jerk, snap = np.einsum(
            "dpqk,pkc->dpqc", bases, blocks)

        halfspace_A, halfspace_b, halfspace_mask = piece_halfspaces
        position_violation = (
            np.einsum("pqd,pfd->pqf", position, halfspace_A)
            - halfspace_b[:, None, :]
        )
        position_values, position_derivatives = _smoothed_l1_array(
            position_violation, self.config.smoothing_epsilon)
        position_values *= halfspace_mask[:, None, :]
        position_derivatives *= halfspace_mask[:, None, :]
        penalty = self.config.position_weight * np.sum(position_values, axis=2)
        grad_position = self.config.position_weight * np.einsum(
            "pqf,pfd->pqd", position_derivatives, halfspace_A)

        vmax2 = self.config.max_velocity**2
        amax2 = self.config.max_acceleration**2
        velocity_values, velocity_derivatives = _smoothed_l1_array(
            np.sum(velocity * velocity, axis=2) - vmax2,
            self.config.smoothing_epsilon)
        penalty += self.config.velocity_weight * velocity_values
        grad_velocity = (
            2.0 * self.config.velocity_weight
            * velocity_derivatives[:, :, None] * velocity)

        acceleration_values, acceleration_derivatives = _smoothed_l1_array(
            np.sum(acceleration * acceleration, axis=2) - amax2,
            self.config.smoothing_epsilon)
        penalty += self.config.acceleration_weight * acceleration_values
        grad_acceleration = (
            2.0 * self.config.acceleration_weight
            * acceleration_derivatives[:, :, None] * acceleration)

        # NED point-mass differential flatness: a = g*e_z - thrust/mass * body_z.
        force_direction = np.stack((
            -acceleration[:, :, 0],
            -acceleration[:, :, 1],
            self.config.gravity - acceleration[:, :, 2],
        ), axis=2)
        force_norm = np.maximum(np.linalg.norm(force_direction, axis=2), 1.0e-8)
        body_z = force_direction / force_norm[:, :, None]
        thrust = self.config.mass * force_norm

        thrust_mean = 0.5 * (self.config.min_thrust + self.config.max_thrust)
        thrust_radius = 0.5 * (self.config.max_thrust - self.config.min_thrust)
        thrust_violation = (thrust - thrust_mean)**2 - thrust_radius**2
        thrust_values, thrust_derivatives = _smoothed_l1_array(
            thrust_violation, self.config.smoothing_epsilon)
        penalty += self.config.thrust_weight * thrust_values
        thrust_violation_gradient = (
            -2.0 * self.config.mass * (thrust - thrust_mean)[:, :, None] * body_z)
        grad_acceleration += (
            self.config.thrust_weight * thrust_derivatives[:, :, None]
            * thrust_violation_gradient)

        cos_tilt = np.clip(body_z[:, :, 2], -1.0, 1.0)
        tilt = np.arccos(cos_tilt)
        tilt_values, tilt_derivatives = _smoothed_l1_array(
            tilt - self.config.max_tilt_angle, self.config.smoothing_epsilon)
        penalty += self.config.tilt_weight * tilt_values
        sin_tilt = np.maximum(
            np.sqrt(np.maximum(1.0 - cos_tilt**2, 0.0)), 1.0e-8)
        vertical = np.array([0.0, 0.0, 1.0])
        tilt_gradient_acceleration = (
            vertical - cos_tilt[:, :, None] * body_z
        ) / (force_norm * sin_tilt)[:, :, None]
        grad_acceleration += (
            self.config.tilt_weight * tilt_derivatives[:, :, None]
            * tilt_gradient_acceleration)

        body_z_dot_jerk = np.sum(body_z * jerk, axis=2)
        projected_jerk = jerk - body_z_dot_jerk[:, :, None] * body_z
        projected_jerk_norm2 = np.sum(projected_jerk * projected_jerk, axis=2)
        body_rate_squared = projected_jerk_norm2 / force_norm**2
        body_rate_values, body_rate_derivatives = _smoothed_l1_array(
            body_rate_squared - self.config.max_body_rate**2,
            self.config.smoothing_epsilon)
        penalty += self.config.body_rate_weight * body_rate_values
        body_rate_gradient_jerk = (
            2.0 * projected_jerk / force_norm[:, :, None]**2)
        body_rate_gradient_acceleration = 2.0 * (
            body_z_dot_jerk[:, :, None] * projected_jerk
            + projected_jerk_norm2[:, :, None] * body_z
        ) / force_norm[:, :, None]**3
        grad_acceleration += (
            self.config.body_rate_weight * body_rate_derivatives[:, :, None]
            * body_rate_gradient_acceleration)
        grad_jerk = (
            self.config.body_rate_weight * body_rate_derivatives[:, :, None]
            * body_rate_gradient_jerk)

        quadrature_weights = np.ones(sample_count)
        quadrature_weights[[0, -1]] = 0.5
        scales = times[:, None] * fraction * quadrature_weights[None, :]
        grad_blocks = (
            np.einsum("pqk,pqc,pq->pkc", bases[0], grad_position, scales)
            + np.einsum("pqk,pqc,pq->pkc", bases[1], grad_velocity, scales)
            + np.einsum("pqk,pqc,pq->pkc", bases[2], grad_acceleration, scales)
            + np.einsum("pqk,pqc,pq->pkc", bases[3], grad_jerk, scales)
        )
        grad_coefficients += grad_blocks.reshape(-1, 3)
        state_gradient = (
            np.sum(grad_position * velocity, axis=2)
            + np.sum(grad_velocity * acceleration, axis=2)
            + np.sum(grad_acceleration * jerk, axis=2)
            + np.sum(grad_jerk * snap, axis=2)
        )
        grad_times += np.sum(
            alpha[None, :] * scales * state_gradient
            + quadrature_weights[None, :] * fraction * penalty,
            axis=1,
        )
        return float(np.sum(scales * penalty))

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
