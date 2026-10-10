"""Vectorized helpers used by GCOPTER's integrated constraints."""

import numpy as np

from .mappings import smoothed_l1_array


def stack_piece_halfspaces(
    corridor_indices: np.ndarray,
    h_polytopes: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad per-piece half-spaces once so quadrature nodes can be batched."""
    max_faces = max(len(A) for A, _ in h_polytopes)
    piece_count = len(corridor_indices)
    stacked_A = np.zeros((piece_count, max_faces, 3), dtype=float)
    stacked_b = np.zeros((piece_count, max_faces), dtype=float)
    active = np.zeros((piece_count, max_faces), dtype=bool)
    for piece, region_index in enumerate(corridor_indices):
        A, b = h_polytopes[int(region_index)]
        face_count = len(A)
        stacked_A[piece, :face_count] = A
        stacked_b[piece, :face_count] = b
        active[piece, :face_count] = True
    return stacked_A, stacked_b, active


__all__ = ["smoothed_l1_array", "stack_piece_halfspaces"]


from .trajectory import polynomial_bases as _polynomial_bases
from .mappings import smoothed_l1_array as _smoothed_l1_array
from .optimization import pullback_samples


def integrated_penalty(times, coefficients, piece_halfspaces, grad_coefficients, grad_times, config):
    resolution = config.integral_resolution
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
        position_violation, config.smoothing_epsilon)
    position_values *= halfspace_mask[:, None, :]
    position_derivatives *= halfspace_mask[:, None, :]
    penalty = config.position_weight * np.sum(position_values, axis=2)
    grad_position = config.position_weight * np.einsum(
        "pqf,pfd->pqd", position_derivatives, halfspace_A)

    vmax2 = config.max_velocity**2
    amax2 = config.max_acceleration**2
    velocity_values, velocity_derivatives = _smoothed_l1_array(
        np.sum(velocity * velocity, axis=2) - vmax2,
        config.smoothing_epsilon)
    penalty += config.velocity_weight * velocity_values
    grad_velocity = (
        2.0 * config.velocity_weight
        * velocity_derivatives[:, :, None] * velocity)

    acceleration_values, acceleration_derivatives = _smoothed_l1_array(
        np.sum(acceleration * acceleration, axis=2) - amax2,
        config.smoothing_epsilon)
    penalty += config.acceleration_weight * acceleration_values
    grad_acceleration = (
        2.0 * config.acceleration_weight
        * acceleration_derivatives[:, :, None] * acceleration)

    # NED point-mass differential flatness: a = g*e_z - thrust/mass * body_z.
    force_direction = np.stack((
        -acceleration[:, :, 0],
        -acceleration[:, :, 1],
        config.gravity - acceleration[:, :, 2],
    ), axis=2)
    force_norm = np.maximum(np.linalg.norm(force_direction, axis=2), 1.0e-8)
    body_z = force_direction / force_norm[:, :, None]
    thrust = config.mass * force_norm

    thrust_mean = 0.5 * (config.min_thrust + config.max_thrust)
    thrust_radius = 0.5 * (config.max_thrust - config.min_thrust)
    thrust_violation = (thrust - thrust_mean)**2 - thrust_radius**2
    thrust_values, thrust_derivatives = _smoothed_l1_array(
        thrust_violation, config.smoothing_epsilon)
    penalty += config.thrust_weight * thrust_values
    thrust_violation_gradient = (
        -2.0 * config.mass * (thrust - thrust_mean)[:, :, None] * body_z)
    grad_acceleration += (
        config.thrust_weight * thrust_derivatives[:, :, None]
        * thrust_violation_gradient)

    cos_tilt = np.clip(body_z[:, :, 2], -1.0, 1.0)
    tilt = np.arccos(cos_tilt)
    tilt_values, tilt_derivatives = _smoothed_l1_array(
        tilt - config.max_tilt_angle, config.smoothing_epsilon)
    penalty += config.tilt_weight * tilt_values
    sin_tilt = np.maximum(
        np.sqrt(np.maximum(1.0 - cos_tilt**2, 0.0)), 1.0e-8)
    vertical = np.array([0.0, 0.0, 1.0])
    tilt_gradient_acceleration = (
        vertical - cos_tilt[:, :, None] * body_z
    ) / (force_norm * sin_tilt)[:, :, None]
    grad_acceleration += (
        config.tilt_weight * tilt_derivatives[:, :, None]
        * tilt_gradient_acceleration)

    body_z_dot_jerk = np.sum(body_z * jerk, axis=2)
    projected_jerk = jerk - body_z_dot_jerk[:, :, None] * body_z
    projected_jerk_norm2 = np.sum(projected_jerk * projected_jerk, axis=2)
    body_rate_squared = projected_jerk_norm2 / force_norm**2
    body_rate_values, body_rate_derivatives = _smoothed_l1_array(
        body_rate_squared - config.max_body_rate**2,
        config.smoothing_epsilon)
    penalty += config.body_rate_weight * body_rate_values
    body_rate_gradient_jerk = (
        2.0 * projected_jerk / force_norm[:, :, None]**2)
    body_rate_gradient_acceleration = 2.0 * (
        body_z_dot_jerk[:, :, None] * projected_jerk
        + projected_jerk_norm2[:, :, None] * body_z
    ) / force_norm[:, :, None]**3
    grad_acceleration += (
        config.body_rate_weight * body_rate_derivatives[:, :, None]
        * body_rate_gradient_acceleration)
    grad_jerk = (
        config.body_rate_weight * body_rate_derivatives[:, :, None]
        * body_rate_gradient_jerk)

    gradients = np.stack((grad_position, grad_velocity, grad_acceleration, grad_jerk))
    scales, grad_blocks, alpha, quadrature_weights = pullback_samples(
        times, bases, penalty, gradients)
    grad_coefficients += grad_blocks.reshape(-1, 3)
    state_gradient = (
        np.sum(grad_position * velocity, axis=2)
        + np.sum(grad_velocity * acceleration, axis=2)
        + np.sum(grad_acceleration * jerk, axis=2)
        + np.sum(grad_jerk * snap, axis=2)
    )
    grad_times += np.sum(
        alpha[None, :] * scales * state_gradient
        + quadrature_weights[None, :] * penalty,
        axis=1,
    )
    return float(np.sum(scales * penalty))
