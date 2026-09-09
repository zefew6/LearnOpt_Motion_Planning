"""Independent Bernstein algebra; control points are rows, coordinates columns."""

from functools import lru_cache
from math import comb, factorial

import numpy as np


def evaluate(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    s = np.asarray(parameters, dtype=float).reshape(-1)
    n = len(points) - 1
    basis = np.stack([comb(n, i) * s**i * (1-s)**(n-i) for i in range(n+1)], axis=1)
    return basis @ points


@lru_cache(maxsize=64)
def derivative_matrix(degree: int, order: int) -> np.ndarray:
    if not 0 <= order <= degree:
        raise ValueError("derivative order must lie between zero and degree")
    return np.diff(np.eye(degree+1), n=order, axis=0) * factorial(degree) / factorial(degree-order)


@lru_cache(maxsize=64)
def product_weights(n: int, d: int) -> np.ndarray:
    weights = np.zeros((n+d+1, n+1, d+1))
    for i in range(n+1):
        for j in range(d+1):
            weights[i+j, i, j] = comb(n, i) * comb(d, j) / comb(n+d, i+j)
    return weights


def elevate_matrix(degree: int, target: int) -> np.ndarray:
    if target < degree:
        raise ValueError("cannot lower Bernstein degree")
    return product_weights(degree, target-degree).sum(axis=2)


def separation_coefficients(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, d = len(points)-1, len(a)-1
    return (np.einsum("kij,il,jl->k", product_weights(n, d), points, a)
            + elevate_matrix(d, n+d) @ b)


def split(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    work = np.array(points, dtype=float, copy=True)
    left, right = [work[0]], [work[-1]]
    while len(work) > 1:
        work = (work[:-1] + work[1:]) * 0.5
        left.append(work[0])
        right.append(work[-1])
    return np.asarray(left), np.asarray(right[::-1])
