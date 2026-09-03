"""Unconstrained SciPy L-BFGS adapter with GCOPTER stopping rules."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class LBFGSResult:
    x: np.ndarray
    cost: float
    gradient: np.ndarray
    iterations: int
    converged: bool
    message: str


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

    def scipy_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal latest_x, latest_cost, latest_gradient
        cost, gradient = objective(x)
        latest_x = np.asarray(x, dtype=float).copy()
        latest_cost = float(cost)
        latest_gradient = np.asarray(gradient, dtype=float).copy()
        return latest_cost, latest_gradient

    def callback(intermediate_result) -> None:
        nonlocal feasible_iterations, stop_message
        accepted_x = np.asarray(intermediate_result.x, dtype=float)
        if latest_x is None or not np.array_equal(accepted_x, latest_x):
            scipy_objective(accepted_x)
        feasible = is_feasible is None or is_feasible()
        if is_feasible is not None and feasible:
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
            relative_change = abs(old_cost - latest_cost) / max(1.0, abs(latest_cost))
            if relative_change <= relative_cost_tolerance and is_feasible is None:
                stop_message = "relative cost tolerance reached"
                raise StopIteration

    result = minimize(
        scipy_objective,
        np.asarray(initial, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=None,
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
                       int(result.nit), converged, message)


__all__ = ["LBFGSResult", "scipy_lbfgs"]
