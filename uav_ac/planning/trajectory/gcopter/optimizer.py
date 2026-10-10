"""Unconstrained SciPy L-BFGS adapter with GCOPTER stopping rules."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EqualityResult:
    x: np.ndarray
    residual: np.ndarray
    feasible: bool
    iterations: int
    message: str


def solve_equalities(objective, initial, equality, *, tolerance=1e-4,
                     max_outer_iterations=12, max_iterations=100):
    """Augmented-Lagrangian equality driver; equality returns residual/Jacobian.

    Residuals should be normalized by their physical scales before this call.
    Infeasible targets are returned with their residuals, never altered.
    """
    x = np.array(initial, dtype=float)
    residual, _ = equality(x)
    multiplier = np.zeros_like(residual)
    penalty, iterations = 10., 0
    for _ in range(max_outer_iterations):
        def augmented(z):
            cost, gradient = objective(z)
            residual, jacobian = equality(z)
            return (cost + multiplier @ residual + .5 * penalty * (residual @ residual),
                    gradient + jacobian.T @ (multiplier + penalty * residual))

        inner = scipy_lbfgs(augmented, x, max_iterations=max_iterations,
            memory=12, gradient_tolerance=1e-9, relative_cost_tolerance=1e-12)
        x = inner.x
        iterations += inner.iterations
        residual, _ = equality(x)
        if np.all(np.isfinite(residual)) and np.max(np.abs(residual), initial=0.) <= tolerance:
            return EqualityResult(x, residual, True, iterations, 'task equalities satisfied')
        multiplier += penalty * residual
        penalty *= 10.
    return EqualityResult(x, residual, False, iterations, 'task equality tolerance not reached')


@dataclass(frozen=True)
class LBFGSResult:
    x: np.ndarray
    cost: float
    gradient: np.ndarray
    iterations: int
    converged: bool
    message: str
    relative_cost_change: float


def scipy_lbfgs(
    objective: Callable,
    initial: np.ndarray,
    *,
    max_iterations: int,
    memory: int,
    gradient_tolerance: float,
    relative_cost_tolerance: float,
    is_feasible: Callable | None = None,
    feasible_iteration_patience: int = 1,
    require_convergence: bool = False,
) -> LBFGSResult:
    """Run unconstrained limited-memory BFGS through SciPy.

    SciPy exposes this algorithm under the ``L-BFGS-B`` method name. With no
    bounds supplied it is the unconstrained L-BFGS needed after GCOPTER's
    diffeomorphic time and waypoint mappings.
    """
    latest_x: np.ndarray | None = None
    latest_cost = np.inf
    latest_gradient = np.zeros_like(initial, dtype=float)
    recent_costs: list[float] = []
    feasible_iterations = 0
    stop_message: str | None = None
    relative_cost_change = np.inf

    def scipy_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal latest_x, latest_cost, latest_gradient
        cost, gradient = objective(x)
        latest_x = np.asarray(x, dtype=float).copy()
        latest_cost = float(cost)
        latest_gradient = np.asarray(gradient, dtype=float).copy()
        return latest_cost, latest_gradient

    def callback(intermediate_result) -> None:
        nonlocal feasible_iterations, stop_message, relative_cost_change
        accepted_x = np.asarray(intermediate_result.x, dtype=float)
        if latest_x is None or not np.array_equal(accepted_x, latest_x):
            scipy_objective(accepted_x)
        feasible = is_feasible is None or is_feasible()
        if is_feasible is not None and feasible and not require_convergence:
            feasible_iterations += 1
            if feasible_iterations >= feasible_iteration_patience:
                stop_message = "constraint-feasible solution reached"
                raise StopIteration
        else:
            feasible_iterations = 0
        if np.linalg.norm(latest_gradient, ord=np.inf) <= gradient_tolerance and feasible:
            stop_message = "gradient tolerance reached"
            raise StopIteration
        recent_costs.append(latest_cost)
        if len(recent_costs) > 4:
            old_cost = recent_costs.pop(0)
            relative_cost_change = (abs(old_cost-latest_cost)
                                    / max(1.0, abs(latest_cost)))
            if (relative_cost_change <= relative_cost_tolerance
                    and (is_feasible is None or require_convergence and feasible)):
                stop_message = "relative cost tolerance reached"
                raise StopIteration

    result = minimize(
        scipy_objective,
        np.asarray(initial, dtype=float),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": max_iterations, "maxcor": memory, "ftol": 0.0,
                 "gtol": 0.0, "maxls": 40},
    )
    scipy_objective(result.x)
    feasible = is_feasible is None or is_feasible()
    converged = stop_message is not None or (bool(result.success) and feasible)
    message = stop_message or str(result.message)
    if result.success and not feasible:
        message = f"SciPy stopped before constraints became feasible: {result.message}"
    return LBFGSResult(np.asarray(result.x, dtype=float), latest_cost, latest_gradient,
                       int(result.nit), converged, message,
                       float(relative_cost_change))


__all__ = ["LBFGSResult", "scipy_lbfgs"]
