"""Dimension-independent polynomial evaluation with optional native acceleration."""

import numpy as np

try:
    from uav_ac.planning.native import _trajectory_math as _native_math
except ImportError:
    _native_math = None


def polynomial_basis(time: float, derivative: int) -> np.ndarray:
    return polynomial_basis_matrix(np.array([time]), derivative)[0]


def _polynomial_basis_matrix_reference(times: np.ndarray, derivative: int) -> np.ndarray:
    """Evaluate one polynomial derivative for an arbitrary batch of times."""
    if not 0 <= derivative <= 5:
        raise ValueError("derivative must lie in [0, 5]")
    times = np.asarray(times, dtype=float).reshape(-1)
    powers = np.arange(6)
    factors = np.ones(6)
    for offset in range(derivative):
        factors *= np.maximum(powers - offset, 0)
    basis = np.zeros((len(times), 6))
    valid = powers >= derivative
    basis[:, valid] = (
        factors[valid][None, :] * times[:, None] ** (powers[valid] - derivative))
    return basis


def _polynomial_bases_reference(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    powers = np.stack(
        [np.ones_like(times), times, times**2, times**3, times**4, times**5], axis=1)
    position = powers
    velocity = np.stack([
        np.zeros_like(times), np.ones_like(times), 2.0 * times,
        3.0 * times**2, 4.0 * times**3, 5.0 * times**4], axis=1)
    acceleration = np.stack([
        np.zeros_like(times), np.zeros_like(times), 2.0 * np.ones_like(times),
        6.0 * times, 12.0 * times**2, 20.0 * times**3], axis=1)
    jerk = np.stack([
        np.zeros_like(times), np.zeros_like(times), np.zeros_like(times),
        6.0 * np.ones_like(times), 24.0 * times, 60.0 * times**2], axis=1)
    snap = np.stack([
        np.zeros_like(times), np.zeros_like(times), np.zeros_like(times),
        np.zeros_like(times), 24.0 * np.ones_like(times), 120.0 * times], axis=1)
    return np.stack((position, velocity, acceleration, jerk, snap), axis=0)


def _evaluate_piecewise_quintic_reference(
        durations: np.ndarray, coefficients: np.ndarray,
        times: float | np.ndarray, derivative: int) -> np.ndarray:
    """Evaluate ascending-power quintic pieces at clipped trajectory times."""
    durations = np.asarray(durations, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    query = np.asarray(times, dtype=float)
    scalar = query.ndim == 0
    flat = np.clip(query.reshape(-1), 0.0, float(np.sum(durations)))
    boundaries = np.cumsum(durations)
    pieces = np.minimum(np.searchsorted(boundaries, flat, side="right"), len(durations)-1)
    starts = np.r_[0.0, boundaries[:-1]]
    local = flat-starts[pieces]
    basis = _polynomial_basis_matrix_reference(local, derivative)
    result = np.einsum("nk,nkc->nc", basis, coefficients[pieces])
    return result[0] if scalar else result


def polynomial_basis_matrix(times: np.ndarray, derivative: int) -> np.ndarray:
    """Evaluate a derivative batch using the optional native arithmetic kernel."""
    if _native_math is None or not isinstance(derivative, (int, np.integer)):
        return _polynomial_basis_matrix_reference(times, derivative)
    if not 0 <= derivative <= 5:
        raise ValueError("derivative must lie in [0, 5]")
    return _native_math.polynomial_basis_matrix(
        np.ascontiguousarray(np.asarray(times, dtype=float).reshape(-1)), int(derivative))


def polynomial_bases(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if _native_math is None or times.ndim != 1:
        return _polynomial_bases_reference(times)
    return _native_math.polynomial_bases(np.ascontiguousarray(times))


def evaluate_piecewise_quintic(durations: np.ndarray, coefficients: np.ndarray,
                               times: float | np.ndarray, derivative: int) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if (_native_math is None or durations.ndim != 1 or len(durations) == 0
            or coefficients.ndim != 3 or coefficients.shape[:2] != (len(durations), 6)
            or not isinstance(derivative, (int, np.integer))):
        return _evaluate_piecewise_quintic_reference(durations, coefficients, times, derivative)
    if not 0 <= derivative <= 5:
        raise ValueError("derivative must lie in [0, 5]")
    query = np.asarray(times, dtype=float)
    flat = np.clip(query.reshape(-1), 0.0, float(np.sum(durations)))
    boundaries = np.cumsum(durations)
    pieces = np.minimum(np.searchsorted(boundaries, flat, side="right"), len(durations)-1)
    local = flat - np.r_[0.0, boundaries[:-1]][pieces]
    result = _native_math.evaluate_quintic(np.ascontiguousarray(coefficients),
        np.ascontiguousarray(pieces, dtype=np.intp), np.ascontiguousarray(local), int(derivative))
    return result[0] if query.ndim == 0 else result


