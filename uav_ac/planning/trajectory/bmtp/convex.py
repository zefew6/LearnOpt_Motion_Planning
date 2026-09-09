"""Independent SOCP formulations of BMTP trajectory and plane updates.

Each segment uses local parameter u in [0,1] and physical duration h. Auxiliary
powers z[0]=1, z[k-1]^2 <= z[k-2] z[k] imply z[k] <= z[J]**(k/J).
Recover h from z[J], NOT the potentially slack z[1].
"""

from time import perf_counter

import cvxpy as cp
import numpy as np

from ...geometry.polytope import ConvexPolytope
from .bezier import derivative_matrix, elevate_matrix, product_weights
from .config import BMTPConfig, BMTPLimits
from .types import BMTPTrajectory


def solve_problem(problem: cp.Problem, config: BMTPConfig) -> tuple[float, float]:
    start = perf_counter()
    problem.solve(solver="CLARABEL", warm_start=True,
                  tol_gap_abs=config.solver_tolerance,
                  tol_gap_rel=config.solver_tolerance,
                  tol_feas=config.solver_tolerance,
                  max_iter=config.solver_max_iterations)
    elapsed = perf_counter()-start
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Clarabel returned {problem.status}")
    return elapsed, float(problem.solver_stats.solve_time or 0)


class TrajectoryProgram:
    def __init__(self, segments: int, domain: ConvexPolytope, limits: BMTPLimits,
                 config: BMTPConfig, tags: tuple[tuple[int, int], ...] = (),
                 on_segment: bool = False):
        self.config = config
        self.segments = segments
        self.n = config.degree
        self.order = len(limits.values)
        self.points = cp.Variable((segments*(self.n+1), 3))
        self.powers = cp.Variable(self.order, nonneg=True)
        self.start = cp.Parameter(3)
        self.goal = cp.Parameter(3)
        constraints = [self.points[0] == self.start, self.points[-1] == self.goal,
                       self.points @ domain.A.T <= domain.b[None, :],
                       self.powers[-1] >= 1e-12]
        powers = [cp.Constant(1.0), *[self.powers[k] for k in range(self.order)]]
        for k in range(2, self.order+1):
            lo, mid, hi = powers[k-2:k+1]
            constraints.append(cp.SOC(lo+hi, cp.hstack((2*mid, lo-hi))))
        if on_segment:
            if segments != 1:
                raise ValueError("chord initialization requires one segment")
            alpha = cp.Variable((self.n+1, 1))
            constraints.extend([alpha >= 0, alpha <= 1,
                                self.points == self.start[None, :] + cp.multiply(alpha, (self.goal-self.start)[None, :])])
        for i in range(segments):
            points = self.points[i*(self.n+1):(i+1)*(self.n+1)]
            for k, limit in enumerate(limits.values, 1):
                # Constrain ALL control points, including the boundaries.
                constraints.append(cp.norm(derivative_matrix(self.n, k) @ points, axis=1) <= limit*powers[k])
            if i:
                previous = self.points[(i-1)*(self.n+1):i*(self.n+1)]
                for k in range(config.continuity_order+1):
                    diff = derivative_matrix(self.n, k)
                    constraints.append(diff[0] @ points == diff[-1] @ previous)
        for k in range(1, config.terminal_order+1):
            diff = derivative_matrix(self.n, k)
            constraints.extend([diff[0] @ self.points[:self.n+1] == 0,
                                diff[-1] @ self.points[-self.n-1:] == 0])
        self.plane_parameters = {}
        count = self.n + config.plane_degree + 1
        for tag in tags:
            segment, _ = tag
            maps = [cp.Parameter((count, self.n+1)) for _ in range(3)]
            bias = cp.Parameter(count)
            points = self.points[segment*(self.n+1):(segment+1)*(self.n+1)]
            coefficients = sum(maps[k] @ points[:, k] for k in range(3)) + bias
            constraints.append(coefficients <= -config.trajectory_margin)
            self.plane_parameters[tag] = (maps, bias)
        self.problem = cp.Problem(cp.Minimize(self.powers[-1]), constraints)

    def solve(self, start, goal, planes) -> tuple[BMTPTrajectory, float, float]:
        self.start.value, self.goal.value = start, goal
        weights = product_weights(self.n, self.config.plane_degree)
        elevation = elevate_matrix(self.config.plane_degree, self.n+self.config.plane_degree)
        for tag, (maps, bias) in self.plane_parameters.items():
            a, b = planes[tag]
            matrix = np.einsum("kij,jl->kil", weights, a)
            for k in range(3):
                maps[k].value = matrix[:, :, k]
            bias.value = elevation @ b
        elapsed, solver_seconds = solve_problem(self.problem, self.config)
        h = float(self.powers.value[-1])**(1/self.order)
        points = self.points.value.reshape(self.segments, self.n+1, 3)
        return BMTPTrajectory(points, h), elapsed, solver_seconds


class PlaneProgram:
    def __init__(self, obstacle: ConvexPolytope, config: BMTPConfig):
        self.config = config
        n, d = config.degree, config.plane_degree
        self.points = cp.Parameter((n+1, 3))
        self.a = cp.Variable((d+1, 3))
        self.b = cp.Variable(d+1)
        dual = cp.Variable((len(obstacle.b), d+1), nonneg=True)
        margin = cp.Variable()
        coefficients = elevate_matrix(d, n+d) @ self.b
        weights = product_weights(n, d)
        for j in range(d+1):
            coefficients += cp.sum(cp.multiply(weights[:, :, j] @ self.points, self.a[j]), axis=1)
        self.problem = cp.Problem(cp.Minimize(margin), [
            self.a == -dual.T @ obstacle.A,
            self.b >= obstacle.b @ dual + config.obstacle_margin,
            cp.norm(self.a, axis=1) <= 1,
            coefficients <= margin,
        ])

    def solve(self, points) -> tuple[tuple[np.ndarray, np.ndarray], float, float]:
        self.points.value = points
        elapsed, solver_seconds = solve_problem(self.problem, self.config)
        if not np.isfinite(self.problem.value) or self.problem.value > -self.config.trajectory_margin:
            raise RuntimeError("finite-degree plane cannot certify required separation margin")
        return (self.a.value.copy(), self.b.value.copy()), elapsed, solver_seconds
