"""GCOPTER time, polynomial and convex-polytope coordinate mappings."""

import numpy as np


def forward_polytope_point(q: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-12:
        return np.mean(vertices, axis=0)
    unit = q / norm
    return vertices[0] + unit[:-1]**2 @ (vertices[1:] - vertices[0])


def backward_polytope_point(
        q: np.ndarray, vertices: np.ndarray, grad_point: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-12:
        return np.zeros_like(q)
    unit = q / norm
    grad_unit = np.zeros_like(unit)
    grad_unit[:-1] = 2.0 * unit[:-1] * ((vertices[1:] - vertices[0]) @ grad_point)
    return (grad_unit - unit * np.dot(unit, grad_unit)) / norm


def project_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = sorted_values - cumulative / indices > 0.0
    threshold = cumulative[np.flatnonzero(valid)[-1]] / np.count_nonzero(valid)
    return np.maximum(values - threshold, 0.0)


def convex_weights(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return the closest-to-uniform exact barycentric coordinates.

    Projecting the uniform weights onto ``[vertices.T; 1] @ weights =
    [point; 1]`` gives the same well-spread inverse-map initialization sought
    by the former projected-gradient loop.  A small active-set loop enforces
    non-negativity and avoids hundreds of simplex projections per waypoint.
    """
    count = len(vertices)
    uniform = np.full(count, 1.0 / count)
    constraints = np.vstack((np.asarray(vertices, dtype=float).T, np.ones(count)))
    target = np.concatenate((np.asarray(point, dtype=float), [1.0]))
    active = np.ones(count, dtype=bool)

    for _ in range(count):
        active_constraints = constraints[:, active]
        active_uniform = uniform[active]
        multiplier = np.linalg.lstsq(
            active_constraints @ active_constraints.T,
            target - active_constraints @ active_uniform,
            rcond=None,
        )[0]
        active_weights = active_uniform + active_constraints.T @ multiplier
        if np.min(active_weights) >= -1.0e-10:
            weights = np.zeros(count)
            weights[active] = np.maximum(active_weights, 0.0)
            weights /= np.sum(weights)
            if np.linalg.norm(weights @ vertices - point) <= 1.0e-7:
                return weights
            break
        active_indices = np.flatnonzero(active)
        active[active_indices[int(np.argmin(active_weights))]] = False

    # Numerical fallback for a point just outside a degenerate vertex hull.
    weights = uniform.copy()
    gram_scale = float(np.linalg.norm(vertices, ord=2)**2)
    step = 1.0 / max(gram_scale, 1.0e-9)
    for _ in range(200):
        residual = weights @ vertices - point
        candidate = project_simplex(weights - step * (vertices @ residual))
        if np.linalg.norm(candidate - weights) < 1.0e-10:
            return candidate
        weights = candidate
    return weights


def convex_weights_batch(
    points: np.ndarray,
    vertices: np.ndarray,
    max_iterations: int = 40,
) -> np.ndarray:
    """Run the original well-spread inverse map for many points at once.

    The projected-gradient trajectory is intentionally retained because its
    approximate, interior initialization gives GCOPTER a better optimization
    basin than sparse exact barycentric coordinates.  Batching removes the
    per-waypoint Python loop without changing those iterations.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    count = len(vertices)
    weights = np.full((len(points), count), 1.0 / count)
    extrapolated = weights.copy()
    momentum = 1.0
    step = 1.0 / max(float(np.linalg.norm(vertices, ord=2)**2), 1.0e-9)
    for _ in range(max_iterations):
        residual = extrapolated @ vertices - points
        candidate = project_simplex_rows(
            extrapolated - step * (residual @ vertices.T))
        if np.max(np.linalg.norm(candidate - weights, axis=1)) < 1.0e-10:
            return candidate
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = candidate + (
            (momentum - 1.0) / next_momentum) * (candidate - weights)
        weights = candidate
        momentum = next_momentum
    return weights


def project_simplex_rows(values: np.ndarray) -> np.ndarray:
    """Project every row onto the probability simplex in one NumPy batch."""
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1)
    valid = ordered - cumulative / indices > 0.0
    active_count = np.count_nonzero(valid, axis=1)
    threshold = cumulative[np.arange(len(values)), active_count - 1] / active_count
    return np.maximum(values - threshold[:, None], 0.0)


def forward_time(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    positive = tau > 0.0
    times = np.empty_like(tau)
    times[positive] = 0.5 * tau[positive] ** 2 + tau[positive] + 1.0
    times[~positive] = 2.0 / (tau[~positive] ** 2 - 2.0 * tau[~positive] + 2.0)
    return times


def inverse_time(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    large = times > 1.0
    tau = np.empty_like(times)
    tau[large] = np.sqrt(2.0 * times[large] - 1.0) - 1.0
    tau[~large] = 1.0 - np.sqrt(2.0 / times[~large] - 1.0)
    return tau


def backward_time_gradient(tau: np.ndarray, grad_times: np.ndarray) -> np.ndarray:
    positive = tau > 0.0
    gradient = np.empty_like(tau)
    gradient[positive] = grad_times[positive] * (tau[positive] + 1.0)
    denominator = tau[~positive] ** 2 - 2.0 * tau[~positive] + 2.0
    gradient[~positive] = (
        grad_times[~positive] * 4.0 * (1.0 - tau[~positive]) / denominator**2)
    return gradient


def polynomial_basis(time: float, derivative: int) -> np.ndarray:
    return polynomial_basis_matrix(np.array([time]), derivative)[0]


def polynomial_basis_matrix(times: np.ndarray, derivative: int) -> np.ndarray:
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


def polynomial_bases(times: np.ndarray) -> np.ndarray:
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


def smoothed_l1_array(
        values: np.ndarray, epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    costs = np.zeros_like(values)
    derivatives = np.zeros_like(values)
    linear = values > epsilon
    costs[linear] = values[linear] - 0.5 * epsilon
    derivatives[linear] = 1.0
    smooth = (values > 0.0) & ~linear
    ratio = values[smooth] / epsilon
    costs[smooth] = (epsilon - 0.5 * values[smooth]) * ratio**3
    derivatives[smooth] = ratio**2 * (
        -0.5 * ratio + 3.0 * (epsilon - 0.5 * values[smooth]) / epsilon)
    return costs, derivatives
