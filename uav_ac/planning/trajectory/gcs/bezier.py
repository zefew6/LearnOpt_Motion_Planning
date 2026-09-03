"""Bezier basis and endpoint derivative operators used by GCS."""

from math import comb, factorial

import numpy as np


def endpoint_derivative_coefficients(
    degree: int, derivative: int, *, at_end: bool,
) -> np.ndarray:
    """Linear coefficients for a normalized-time endpoint derivative."""
    if not 0 <= derivative <= degree:
        raise ValueError("derivative must lie in [0, degree]")
    coefficients = np.zeros(degree + 1)
    scale = factorial(degree) / factorial(degree - derivative)
    for index in range(derivative + 1):
        sign = (-1.0) ** (derivative - index)
        control_index = degree - derivative + index if at_end else index
        coefficients[control_index] = scale * sign * comb(derivative, index)
    return coefficients


def evaluate_bezier(control_points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Evaluate one Bezier segment for normalized times in ``[0, 1]``."""
    control_points = np.asarray(control_points, dtype=float)
    times = np.asarray(times, dtype=float).reshape(-1)
    degree = len(control_points) - 1
    basis = np.stack([
        comb(degree, index) * (1.0 - times) ** (degree - index) * times**index
        for index in range(degree + 1)
    ], axis=1)
    return basis @ control_points


def evaluate_bezier_derivative(
    control_points: np.ndarray, times: np.ndarray, derivative: int,
) -> np.ndarray:
    """Evaluate a normalized-parameter derivative of one Bézier segment."""
    control_points = np.asarray(control_points, dtype=float)
    degree = len(control_points) - 1
    if not 0 <= derivative <= degree:
        raise ValueError("derivative must lie in [0, degree]")
    if derivative == 0:
        return evaluate_bezier(control_points, times)
    scale = factorial(degree) / factorial(degree - derivative)
    derivative_points = scale * np.diff(control_points, n=derivative, axis=0)
    return evaluate_bezier(derivative_points, times)


def derivative_energy_matrix(degree: int, derivative: int) -> np.ndarray:
    """Return ``Q`` with ``trace(P.T @ Q @ P) = integral ||r^(k)||^2``.

    The Gram matrix integrates the Bernstein basis exactly; the finite-
    difference map converts the original control points to derivative control
    points.  This avoids quadrature and is positive semidefinite by construction.
    """
    if not 1 <= derivative <= degree:
        raise ValueError("derivative must lie in [1, degree]")
    reduced_degree = degree - derivative
    gram = np.empty((reduced_degree + 1, reduced_degree + 1))
    for row in range(reduced_degree + 1):
        for column in range(reduced_degree + 1):
            gram[row, column] = (
                comb(reduced_degree, row) * comb(reduced_degree, column)
                / ((2 * reduced_degree + 1)
                   * comb(2 * reduced_degree, row + column)))
    difference = np.zeros((reduced_degree + 1, degree + 1))
    scale = factorial(degree) / factorial(degree - derivative)
    for row in range(reduced_degree + 1):
        for offset in range(derivative + 1):
            difference[row, row + offset] = (
                scale * (-1.0) ** (derivative - offset)
                * comb(derivative, offset))
    matrix = difference.T @ gram @ difference
    return 0.5 * (matrix + matrix.T)


def derivative_energy_factor(degree: int, derivative: int) -> np.ndarray:
    """Return a stable factor ``L`` satisfying ``L.T @ L == Q``."""
    values, vectors = np.linalg.eigh(derivative_energy_matrix(degree, derivative))
    threshold = max(1.0, float(np.max(values))) * 1.0e-11
    keep = values > threshold
    return np.sqrt(values[keep])[:, None] * vectors[:, keep].T


__all__ = [
    "derivative_energy_factor", "derivative_energy_matrix",
    "endpoint_derivative_coefficients", "evaluate_bezier",
    "evaluate_bezier_derivative",
]
