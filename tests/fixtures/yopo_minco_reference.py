"""YOPO-MINCO inference matrix reference (test-only).

Copyright (c) 2024 TJU-Aerial-Robotics, Tianjin University, China.
MIT license: see YOPO_REFERENCE_LICENSE.
Source: policy/poly_solver.py, commit d5e1e60f83cc991602f142d9b6b96204a24d6b9f.
Only the official matrix construction is retained; no upstream dependencies.
"""
import numpy as np

def build_A(T0, T1):
    """Batched 12×12 constraint matrices for 2-piece MincoS3NU. T0/T1: (B,) piece durations."""
    B = T0.shape[0]
    A = np.zeros((B, 12, 12))
    one = np.ones(B)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 2, 2] = 2.0
    t1, t2, t3, t4, t5 = (T0, T0 ** 2, T0 ** 3, T0 ** 4, T0 ** 5)
    A[:, 3, 0:6] = np.stack([one, t1, t2, t3, t4, t5], axis=-1)
    A[:, 4, 0:6] = A[:, 3, 0:6]
    A[:, 4, 6] = -1.0
    A[:, 5, 1:6] = np.stack([one, 2 * t1, 3 * t2, 4 * t3, 5 * t4], axis=-1)
    A[:, 5, 7] = -1.0
    A[:, 6, 2:6] = np.stack([2 * one, 6 * t1, 12 * t2, 20 * t3], axis=-1)
    A[:, 6, 8] = -2.0
    A[:, 7, 3:6] = np.stack([6 * one, 24 * t1, 60 * t2], axis=-1)
    A[:, 7, 9] = -6.0
    A[:, 8, 4:6] = np.stack([24 * one, 120 * t1], axis=-1)
    A[:, 8, 10] = -24.0
    t1, t2, t3, t4, t5 = (T1, T1 ** 2, T1 ** 3, T1 ** 4, T1 ** 5)
    A[:, 9, 6:12] = np.stack([one, t1, t2, t3, t4, t5], axis=-1)
    A[:, 10, 7:12] = np.stack([one, 2 * t1, 3 * t2, 4 * t3, 5 * t4], axis=-1)
    A[:, 11, 8:12] = np.stack([2 * one, 6 * t1, 12 * t2, 20 * t3], axis=-1)
    return A
