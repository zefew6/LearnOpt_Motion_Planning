"""Shared spline optimization: variables, constraints, sampling and adjoints."""

import numpy as np
from dataclasses import dataclass
from .trajectory import polynomial_basis_matrix

# Waypoint specifications

@dataclass(frozen=True)
class DerivativeWaypoint:
    knot: int
    derivative: int
    target: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        target, mask = np.array(self.target, dtype=float), np.array(self.mask, dtype=bool)
        if self.knot < 0 or not 0 <= self.derivative <= 3:
            raise ValueError('invalid knot or derivative')
        if target.ndim != 1 or mask.shape != target.shape or not np.all(np.isfinite(target)):
            raise ValueError('invalid waypoint target or mask')
        target.setflags(write=False); mask.setflags(write=False)
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, 'mask', mask)

    def linearize(self, durations, coefficients):
        if self.knot > len(durations) or self.target.size != coefficients.shape[-1]:
            raise ValueError('waypoint knot or dimension does not match trajectory')
        piece, local = (0, 0.) if self.knot == 0 else (self.knot-1, durations[self.knot-1])
        basis = polynomial_basis_matrix(np.array([local]), self.derivative)[0]
        next_basis = polynomial_basis_matrix(np.array([local]), self.derivative+1)[0]
        dimensions = np.flatnonzero(self.mask)
        residual = (basis@coefficients[piece] - self.target)[dimensions]
        gc = np.zeros((len(dimensions), *coefficients.shape))
        gt = np.zeros((len(dimensions), len(durations)))
        for row, dimension in enumerate(dimensions):
            gc[row, piece, :, dimension] = basis
            if self.knot:
                gt[row, piece] = next_basis@coefficients[piece, :, dimension]
        return residual, gc, gt


# Decision-variable maps

class FixedWaypointMap:
    def __init__(self, points, fixed):
        self.points = np.asarray(points, dtype=float).copy()
        self.fixed = np.asarray(fixed, dtype=bool).copy()
        if self.points.ndim != 2 or self.fixed.shape != self.points.shape:
            raise ValueError('waypoint values and mask must have matching 2-D shapes')
        if not np.all(np.isfinite(self.points)):
            raise ValueError('waypoints must be finite')
        self.points.setflags(write=False)
        self.fixed.setflags(write=False)
        self.dimension = int(np.count_nonzero(~self.fixed))

    def encode(self, points):
        points = np.asarray(points)
        if points.shape != self.points.shape:
            raise ValueError('waypoint shape mismatch')
        return points[~self.fixed].copy()

    def decode(self, variables):
        variables = np.asarray(variables, dtype=float)
        if variables.shape != (self.dimension,) or not np.all(np.isfinite(variables)):
            raise ValueError('waypoint decision dimension or values invalid')
        points = self.points.copy()
        points[~self.fixed] = variables
        return points

    def pullback(self, gradient):
        return self.encode(gradient)


class FixedBoundaryDerivativeMap:
    """Eliminate selected waypoint coordinates for fixed boundary derivatives.

    The derivative is linear in waypoints for fixed times/boundaries. The
    implicit pullback accounts for changes of times and both boundary states.
    """

    def __init__(self, dependent, derivative=3, components=(0, 1)):
        self.dependent = np.asarray(dependent, dtype=bool)
        self.derivative, self.components = derivative, components

    def project(self, minco, points, durations):
        residual, gp, *_ = boundary_derivative_jacobian(
            minco, points, durations, self.derivative, self.components)
        matrix = gp[:, self.dependent]
        corrected = points.copy()
        corrected[self.dependent] += np.linalg.solve(matrix, -residual)
        linearization = boundary_derivative_jacobian(
            minco, corrected, durations, self.derivative, self.components)
        return corrected, linearization

    def pullback(self, gp, gt, gh, ge, linearization):
        _, jp, jt, jh, je = linearization
        multipliers = np.linalg.solve(jp[:, self.dependent].T, gp[self.dependent])
        return (gp - np.einsum('r,rij->ij', multipliers, jp),
                gt - multipliers @ jt,
                gh - np.einsum('r,rij->ij', multipliers, jh),
                ge - np.einsum('r,rij->ij', multipliers, je))


# Sampling and derivative pullbacks

def boundary_derivative_jacobian(minco, points, durations, derivative, components):
    """Values and physical-variable Jacobians at the two trajectory boundaries."""
    flat, system = minco.solve(points, durations)
    blocks = flat.reshape(len(durations), 6, minco.dimensions)
    values, gps, gts, ghs, ges = [], [], [], [], []
    for end in (False, True):
        piece, local = (-1, durations[-1]) if end else (0, 0.)
        basis = polynomial_basis_matrix(np.array([local]), derivative)[0]
        next_basis = polynomial_basis_matrix(np.array([local]), derivative + 1)[0]
        for component in components:
            values.append(float(basis @ blocks[piece, :, component]))
            gc, gt = np.zeros_like(blocks), np.zeros(len(durations))
            gc[piece, :, component] = basis
            if end:
                gt[-1] = next_basis @ blocks[-1, :, component]
            gp, gt = minco.propagate_gradient(system, flat, durations, gc.reshape(flat.shape), gt)
            adjoint = system.solve(gc.reshape(flat.shape), transpose=True)
            gps.append(gp); gts.append(gt); ghs.append(adjoint[:3]); ges.append(adjoint[-3:])
    return np.array(values), np.array(gps), np.array(gts), np.array(ghs), np.array(ges)


def integrate_samples(durations, coefficients, resolution, sample):
    """Integrate a local cost with derivatives w.r.t. p/v/a/j.

    sample(piece, alpha, values[0:5]) returns (cost, gradients[0:4]).
    Explicit time dependence belongs in a coefficient term instead.
    """
    if resolution < 1:
        raise ValueError('quadrature resolution must be positive')
    alpha = np.linspace(0., 1., resolution + 1)
    weights = np.ones(resolution + 1); weights[[0, -1]] = .5
    gc, gt = np.zeros_like(coefficients), np.zeros(len(durations))
    total = 0.
    for piece, duration in enumerate(durations):
        bases = np.stack([polynomial_basis_matrix(alpha * duration, d) for d in range(5)])
        values = bases @ coefficients[piece]
        for k, fraction in enumerate(alpha):
            cost, grads = sample(piece, fraction, values[:, k])
            scale = duration * weights[k] / resolution
            total += scale * cost
            gc[piece] += scale * np.einsum('dk,dc->kc', bases[:4, k], grads)
            gt[piece] += scale * fraction * np.sum(grads * values[1:5, k])
            gt[piece] += weights[k] / resolution * cost
    return float(total), gc, gt


def pullback_samples(durations, bases, penalties, gradients):
    """Vectorized trapezoidal pullback for a batch of piece-local costs."""
    resolution = bases.shape[2] - 1
    alpha = np.linspace(0., 1., resolution + 1)
    weights = np.ones(resolution + 1); weights[[0, -1]] = .5
    scales = np.asarray(durations)[:, None] * weights / resolution
    gc = np.einsum('dpqk,dpqc,pq->pkc', bases[:4], gradients, scales)
    return scales, gc, alpha, weights / resolution


# Objective and constraint assembly

def evaluate_minco_objective(minco, points, durations, *, terms=(),
                             time_weight=0., energy_weights=None, boundary_gradients=False):
    """Return cost, waypoint/time gradients and piecewise coefficients.

    Each term accepts (durations, coefficients) and returns cost, coefficient
    partials and independent duration partials. No term runs its own adjoint.
    """
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(durations)):
        raise ValueError('MINCO points and durations must be finite')
    flat, system = minco.solve(points, durations)
    if not np.all(np.isfinite(flat)):
        raise ValueError('MINCO coefficients must be finite')
    coefficients = flat.reshape(len(durations), 6, minco.dimensions)
    cost, gc, gt = minco.jerk_energy(coefficients, durations, energy_weights)
    gc = gc.reshape(coefficients.shape)
    for term in terms:
        value, partial_c, partial_t = term(durations, coefficients)
        partial_c = np.asarray(partial_c)
        partial_t = np.asarray(partial_t)
        if partial_c.shape != coefficients.shape or partial_t.shape != gt.shape:
            raise ValueError('objective component gradient shape mismatch')
        cost += value
        gc += partial_c
        gt += partial_t
    cost += time_weight * np.sum(durations)
    gp, gt = minco.propagate_gradient(system, flat, durations,
                                    gc.reshape(flat.shape), gt + time_weight)
    result = float(cost), gp, gt, coefficients
    if boundary_gradients:
        adjoint = system.solve(gc.reshape(flat.shape), transpose=True)
        return (*result, adjoint[:3].copy(), adjoint[-3:].copy())
    return result


def evaluate_minco_constraints(minco, points, durations, constraint):
    """Propagate any coefficient-domain constraint to physical spline variables."""
    coefficients, system = minco.solve(points, durations)
    residual, coefficient_rows, time_rows = constraint.linearize(
        durations, coefficients.reshape(len(durations), 6, minco.dimensions))
    gps, gts, ghs, ges = [], [], [], []
    for gc, gt in zip(coefficient_rows, time_rows, strict=True):
        gc = gc.reshape(coefficients.shape)
        gp, gt = minco.propagate_gradient(system, coefficients, durations, gc, gt)
        adjoint = system.solve(gc, transpose=True)
        gps.append(gp); gts.append(gt); ghs.append(adjoint[:3]); ges.append(adjoint[-3:])
    return residual, np.array(gps), np.array(gts), np.array(ghs), np.array(ges)
