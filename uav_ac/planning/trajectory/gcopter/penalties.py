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
