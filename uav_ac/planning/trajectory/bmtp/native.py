"""Reusable sparse Clarabel trajectory-update SOCP for BMTP."""

from time import perf_counter

import clarabel
import numpy as np
from scipy import sparse

from ...geometry.polytope import ConvexPolytope
from .bezier import derivative_matrix, elevate_matrix, product_weights
from .config import BMTPConfig, BMTPLimits
from .types import BMTPTrajectory


class NativeTrajectoryProgram:
    """One fixed-size conic program with parameterized active plane slots.

    Keeping a fixed number of plane slots *per spline segment* lets Clarabel
    update numerical values without repeating symbolic factorization. Empty
    slots are relaxed with a large right-hand side.
    """

    _INACTIVE_BOUND = 1.0e6

    def __init__(self, segments: int, domain: ConvexPolytope,
                 limits: BMTPLimits, config: BMTPConfig):
        self.segments, self.domain, self.limits, self.config = segments, domain, limits, config
        self.n, self.order = config.degree, len(limits.values)
        self.points = segments * (self.n + 1)
        self.power_offset = 3 * self.points
        self.variables = self.power_offset + self.order
        self._entries: list[tuple[int, int, float]] = []
        self._rhs: list[float] = []
        self._plane_rows: list[list[int]] = []
        self._plane_entry_indices: np.ndarray | None = None
        self._start_rows: list[int] = []
        self._goal_rows: list[int] = []
        self._cones = []
        self._build()
        self._solver = None

    def _point(self, point: int, axis: int) -> int:
        return 3 * point + axis

    def _power(self, order: int) -> int:
        return self.power_offset + order - 1

    def _rows(self, count: int, cone) -> list[int]:
        start = len(self._rhs)
        self._rhs.extend([0.0] * count)
        self._cones.append(cone(count))
        return list(range(start, start + count))

    def _add(self, row: int, column: int, value: float) -> None:
        self._entries.append((row, column, value))

    def _build(self) -> None:
        # Equalities: endpoints, spline continuity, and rest conditions.
        equality: list[tuple[list[tuple[int, float]], float, str | None]] = []
        for axis in range(3):
            equality.append(([(self._point(0, axis), 1.0)], 0.0, "start"))
            equality.append(([(self._point(self.points - 1, axis), 1.0)], 0.0, "goal"))
        for segment in range(1, self.segments):
            previous = (segment - 1) * (self.n + 1)
            current = segment * (self.n + 1)
            for order in range(self.config.continuity_order + 1):
                matrix = derivative_matrix(self.n, order)
                for axis in range(3):
                    terms = [(self._point(current + i, axis), matrix[0, i])
                             for i in range(self.n + 1)]
                    terms += [(self._point(previous + i, axis), -matrix[-1, i])
                              for i in range(self.n + 1)]
                    equality.append((terms, 0.0, None))
        for order in range(1, self.config.terminal_order + 1):
            matrix = derivative_matrix(self.n, order)
            for axis in range(3):
                equality.append(([(self._point(i, axis), matrix[0, i])
                                  for i in range(self.n + 1)], 0.0, None))
                tail = (self.segments - 1) * (self.n + 1)
                equality.append(([(self._point(tail + i, axis), matrix[-1, i])
                                  for i in range(self.n + 1)], 0.0, None))
        rows = self._rows(len(equality), clarabel.ZeroConeT)
        for row, (terms, rhs, kind) in zip(rows, equality):
            self._rhs[row] = rhs
            for column, value in terms:
                self._add(row, column, value)
            if kind == "start": self._start_rows.append(row)
            if kind == "goal": self._goal_rows.append(row)

        # Linear inequalities: workspace, nonnegative powers, and fixed plane slots.
        inequalities: list[tuple[list[tuple[int, float]], float]] = []
        for point in range(self.points):
            for normal, bound in zip(self.domain.A, self.domain.b):
                inequalities.append(([(self._point(point, axis), normal[axis]) for axis in range(3)], bound))
        for order in range(1, self.order + 1):
            inequalities.append(([(self._power(order), -1.0)], 0.0))
        inequalities.append(([(self._power(self.order), -1.0)], -1.0e-12))
        plane_count = self.n + self.config.plane_degree + 1
        inequalities.extend([([], self._INACTIVE_BOUND)] * (
            self.segments * self.config.active_plane_slots * plane_count))
        rows = self._rows(len(inequalities), clarabel.NonnegativeConeT)
        fixed = len(inequalities) - self.segments * self.config.active_plane_slots * plane_count
        for index, (row, (terms, rhs)) in enumerate(zip(rows, inequalities)):
            self._rhs[row] = rhs
            for column, value in terms: self._add(row, column, value)
            if index >= fixed:
                self._plane_rows.append(row)
        # Every plane slot owns coefficients for every trajectory segment.
        # They begin as explicit *zeros*, rather than absent entries, so the
        # CSC pattern is identical when a slot becomes active later.
        self._plane_rows = np.asarray(self._plane_rows, dtype=int).reshape(
            self.segments, self.config.active_plane_slots, plane_count)
        self._plane_entry_indices = np.empty(
            (self.segments, self.config.active_plane_slots, plane_count, self.n + 1, 3), dtype=int)
        for segment in range(self.segments):
            base = segment * (self.n + 1)
            for slot in range(self.config.active_plane_slots):
                for output in range(plane_count):
                    row = self._plane_rows[segment, slot, output]
                    for index in range(self.n + 1):
                        for axis in range(3):
                            self._plane_entry_indices[segment, slot, output, index, axis] = len(self._entries)
                            self._add(row, self._point(base + index, axis), 0.0)

        # Norm derivative bounds and geometric-mean power constraints.
        for segment in range(self.segments):
            base = segment * (self.n + 1)
            for order, limit in enumerate(self.limits.values, 1):
                matrix = derivative_matrix(self.n, order)
                for output in range(self.n - order + 1):
                    rows = self._rows(4, clarabel.SecondOrderConeT)
                    self._add(rows[0], self._power(order), -limit)
                    for axis in range(3):
                        for index, value in enumerate(matrix[output]):
                            self._add(rows[axis + 1], self._point(base + index, axis), -value)
        powers = [None] + [self._power(k) for k in range(1, self.order + 1)]
        for order in range(2, self.order + 1):
            rows = self._rows(3, clarabel.SecondOrderConeT)
            if order == 2:
                # z_0 is the fixed scalar one: [1 + z_2, 2 z_1, 1 - z_2]
                # belongs to the SOC.
                self._rhs[rows[0]] = self._rhs[rows[2]] = 1.0
            else:
                self._add(rows[0], powers[order - 2], -1.0)
                self._add(rows[2], powers[order - 2], -1.0)
            self._add(rows[0], powers[order], -1.0)
            self._add(rows[1], powers[order - 1], -2.0)
            self._add(rows[2], powers[order], 1.0)

    def _matrix(self, planes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]) -> tuple[sparse.csc_matrix, np.ndarray]:
        rows, columns, values = zip(*self._entries) if self._entries else ([], [], [])
        rows, columns, values = list(rows), list(columns), np.asarray(values, dtype=float)
        rhs = np.asarray(self._rhs, float).copy()
        plane_count = self.n + self.config.plane_degree + 1
        weights = product_weights(self.n, self.config.plane_degree)
        elevation = elevate_matrix(self.config.plane_degree, self.n + self.config.plane_degree)
        tags_by_segment = [[] for _ in range(self.segments)]
        for tag in sorted(planes):
            tags_by_segment[tag[0]].append(tag)
        for segment, tags in enumerate(tags_by_segment):
            if len(tags) > self.config.active_plane_slots:
                raise RuntimeError(
                    f"segment {segment} exceeded its {self.config.active_plane_slots} active-plane slots")
            for slot in range(self.config.active_plane_slots):
                active = slot < len(tags)
                if active:
                    a, b = planes[tags[slot]]
                    coefficients = np.einsum("kij,jl->kil", weights, a)
                    bias = elevation @ b
                else:
                    coefficients = None
                    bias = np.full(plane_count, -self._INACTIVE_BOUND)
                for output in range(plane_count):
                    row = self._plane_rows[segment, slot, output]
                    rhs[row] = -self.config.trajectory_margin - bias[output]
                    if active:
                        indices = self._plane_entry_indices[segment, slot, output]
                        values[indices] = coefficients[output]
        matrix = sparse.coo_matrix((values, (rows, columns)), shape=(len(rhs), self.variables)).tocsc()
        return matrix, rhs

    def solve(self, start: np.ndarray, goal: np.ndarray,
              planes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]) -> tuple[BMTPTrajectory, float, float]:
        matrix, rhs = self._matrix(planes)
        for row, value in zip(self._start_rows, start): rhs[row] = value
        for row, value in zip(self._goal_rows, goal): rhs[row] = value
        objective = np.zeros(self.variables); objective[-1] = 1.0
        settings = clarabel.DefaultSettings(); settings.verbose = False
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = self.config.solver_tolerance
        settings.max_iter = self.config.solver_max_iterations
        started = perf_counter()
        if self._solver is None:
            self._solver = clarabel.DefaultSolver(sparse.csc_matrix((self.variables, self.variables)), objective, matrix, rhs, self._cones, settings)
        else:
            self._solver.update(A=matrix, b=rhs)
        self._solver.solve()
        elapsed = perf_counter() - started
        solution, info = self._solver.get_solution(), self._solver.get_info()
        if str(solution.status) not in {"Solved", "AlmostSolved"}:
            raise RuntimeError(f"Clarabel returned {solution.status}")
        values = np.asarray(solution.x)
        h = float(values[-1]) ** (1 / self.order)
        points = values[:self.power_offset].reshape(self.points, 3).reshape(self.segments, self.n + 1, 3)
        return BMTPTrajectory(points, h), elapsed, float(info.solve_time)
