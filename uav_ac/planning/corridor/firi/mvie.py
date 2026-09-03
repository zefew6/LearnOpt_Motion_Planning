"""Maximum-volume inscribed ellipsoid optimization for FIRI."""

import numpy as np
from scipy.optimize import linprog, minimize

from uav_ac.planning.corridor.firi.config import FIRIConfig


def ellipsoid_volume(shape: np.ndarray) -> float:
    return 4.0 * np.pi * abs(float(np.linalg.det(shape))) / 3.0


def maximum_volume_inscribed_ellipsoid(
        A: np.ndarray,
        b: np.ndarray,
        shape_hint: np.ndarray,
        center_hint: np.ndarray,
        config: FIRIConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize GCOPTER's Cholesky-parameterized MVIE objective."""
    norms = np.linalg.norm(A, axis=1)
    normalized_A = A / norms[:, None]
    normalized_b = b / norms
    interior, depth = deepest_interior(normalized_A, normalized_b)
    slack = normalized_b - normalized_A @ interior
    scaled_A = normalized_A / slack[:, None]
    try:
        lower = np.linalg.cholesky(shape_hint @ shape_hint.T)
    except np.linalg.LinAlgError:
        lower = max(0.5 * depth, 1.0e-4) * np.eye(3)
    variables = np.array([
        *(center_hint - interior),
        np.sqrt(max(lower[0, 0], 1.0e-12)),
        np.sqrt(max(lower[1, 1], 1.0e-12)),
        np.sqrt(max(lower[2, 2], 1.0e-12)),
        lower[1, 0], lower[2, 1], lower[2, 0],
    ])
    optimized = minimize(
        mvie_cost_gradient,
        variables,
        args=(scaled_A, config.mvie_smoothing, config.mvie_penalty),
        method="L-BFGS-B",
        jac=True,
        bounds=None,
        options={
            "maxiter": config.mvie_max_iterations,
            "maxcor": 18,
            "ftol": 1.0e-12,
            "gtol": 0.0,
            "maxls": 40,
        },
    )
    center_offset, lower = decode_mvie(optimized.x)
    center = center_offset + interior
    available = b - A @ center
    radii = np.linalg.norm(A @ lower, axis=1)
    scale = float(np.min(available / np.maximum(radii, 1.0e-15)))
    if scale <= 0.0:
        raise ValueError("MVIE optimization left the polytope")
    lower *= min(1.0, scale * (1.0 - 1.0e-10))
    return lower, center


def deepest_interior(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the Chebyshev center from GCOPTER's four-variable LP."""
    result = linprog(
        np.array([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))),
        b_ub=b,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if not result.success or result.x[3] <= 1.0e-10:
        raise ValueError("FIRI polytope has no strictly interior point")
    return result.x[:3], float(result.x[3])


def decode_mvie(variables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros((3, 3))
    lower[0, 0] = variables[3] ** 2 + np.finfo(float).eps
    lower[1, 1] = variables[4] ** 2 + np.finfo(float).eps
    lower[2, 2] = variables[5] ** 2 + np.finfo(float).eps
    lower[1, 0], lower[2, 1], lower[2, 0] = variables[6:9]
    return variables[:3], lower


def mvie_cost_gradient(
        variables: np.ndarray,
        A: np.ndarray,
        smoothing: float,
        penalty: float,
) -> tuple[float, np.ndarray]:
    """Return the smooth-penalty MVIE cost and its analytic gradient."""
    center, lower = decode_mvie(variables)
    transformed = A @ lower
    transformed_norm = np.maximum(np.linalg.norm(transformed, axis=1), 1.0e-15)
    violation = transformed_norm + A @ center - 1.0
    active = violation >= 0.0
    cost_terms = np.zeros(len(A))
    derivatives = np.zeros(len(A))
    linear = violation > smoothing
    cost_terms[linear] = violation[linear] - 0.5 * smoothing
    derivatives[linear] = 1.0
    smooth = active & ~linear
    ratio = violation[smooth] / smoothing
    cost_terms[smooth] = (smoothing - 0.5 * violation[smooth]) * ratio**3
    derivatives[smooth] = ratio**2 * (
        -0.5 * ratio
        + 3.0 * (smoothing - 0.5 * violation[smooth]) / smoothing)
    weighted = derivatives[:, None] * transformed / transformed_norm[:, None]
    grad_lower = penalty * A.T @ weighted
    gradient = np.zeros(9)
    gradient[:3] = penalty * derivatives @ A
    diagonal = np.diag(lower)
    diagonal_gradient = np.diag(grad_lower) - 1.0 / diagonal
    gradient[3:6] = 2.0 * variables[3:6] * diagonal_gradient
    gradient[6] = grad_lower[1, 0]
    gradient[7] = grad_lower[2, 1]
    gradient[8] = grad_lower[2, 0]
    cost = penalty * np.sum(cost_terms) - np.log(diagonal).sum()
    return float(cost), gradient
