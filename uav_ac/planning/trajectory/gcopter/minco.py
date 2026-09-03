"""MINCO quintic coefficient solve and analytic adjoint gradients."""

import numpy as np
from scipy.linalg.lapack import dgbtrf, dgbtrs


class BandedPLU:
    """Pivoted LAPACK factorization reusable for a banded system and transpose."""

    def __init__(self, matrix: np.ndarray, lower_bandwidth: int, upper_bandwidth: int):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("banded system matrix must be square")
        self.lower_bandwidth = lower_bandwidth
        self.upper_bandwidth = upper_bandwidth
        size = len(matrix)
        storage = np.zeros((2 * lower_bandwidth + upper_bandwidth + 1, size),
                           dtype=float, order="F")
        for offset in range(-lower_bandwidth, upper_bandwidth + 1):
            diagonal = np.diagonal(matrix, offset=offset)
            column = max(offset, 0)
            storage[lower_bandwidth + upper_bandwidth - offset,
                    column:column + len(diagonal)] = diagonal
        self._factorize(storage)

    @classmethod
    def from_storage(cls, storage: np.ndarray, lower_bandwidth: int,
                     upper_bandwidth: int) -> "BandedPLU":
        system = cls.__new__(cls)
        system.lower_bandwidth = lower_bandwidth
        system.upper_bandwidth = upper_bandwidth
        system._factorize(np.array(storage, dtype=float, order="F", copy=True))
        return system

    def _factorize(self, storage: np.ndarray) -> None:
        self.lu, self.pivots, info = dgbtrf(
            storage, self.lower_bandwidth, self.upper_bandwidth, overwrite_ab=True)
        if info != 0:
            raise np.linalg.LinAlgError(f"banded PLU factorization failed with info={info}")

    def solve(self, rhs: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        values = np.asarray(rhs, dtype=float)
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        solution, info = dgbtrs(
            self.lu, self.lower_bandwidth, self.upper_bandwidth,
            np.asfortranarray(values), self.pivots,
            trans=1 if transpose else 0, overwrite_b=True)
        if info != 0:
            raise np.linalg.LinAlgError(f"banded PLU solve failed with info={info}")
        return solution[:, 0] if vector else solution


class MINCOQuintic:
    """Banded-LAPACK port of GCOPTER's non-uniform ``MINCO_S3NU``."""

    def __init__(self, head_pva: np.ndarray, tail_pva: np.ndarray, pieces: int):
        self.head_pva = np.asarray(head_pva, dtype=float)
        self.tail_pva = np.asarray(tail_pva, dtype=float)
        self.pieces = pieces

    def solve(self, points: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, BandedPLU]:
        rhs = np.zeros((6 * self.pieces, 3), dtype=float)
        rhs[0:3] = self.head_pva
        # MINCO_S3NU puts the waypoint equation on row 5 of each interior
        # block to preserve bandwidth; rows are deliberately not derivative-sorted.
        rhs[6 * np.arange(self.pieces - 1) + 5] = points
        rhs[-3:] = self.tail_pva
        system = BandedPLU.from_storage(self._band_storage(times), 6, 6)
        return system.solve(rhs), system

    def _matrix(self, times: np.ndarray) -> np.ndarray:
        storage = self._band_storage(times)
        size = 6 * self.pieces
        matrix = np.zeros((size, size), dtype=float)
        for offset in range(-6, 7):
            column = max(offset, 0)
            length = size - abs(offset)
            rows = np.arange(length) + max(-offset, 0)
            columns = np.arange(length) + column
            matrix[rows, columns] = storage[12 - offset, column:column + length]
        return matrix

    def _band_storage(self, times: np.ndarray) -> np.ndarray:
        count = self.pieces
        storage = np.zeros((19, 6 * count), dtype=float, order="F")

        def put(row: int, column: int, value: float) -> None:
            storage[12 + row - column, column] = value

        put(0, 0, 1.0)
        put(1, 1, 1.0)
        put(2, 2, 2.0)
        for index in range(count - 1):
            time = times[index]
            t2, t3, t4, t5 = time**2, time**3, time**4, time**5
            row = 6 * index
            jerk_row, snap_row, waypoint_row = row + 3, row + 4, row + 5
            position_row, velocity_row, acceleration_row = row + 6, row + 7, row + 8
            put(jerk_row, row + 3, 6.0)
            put(jerk_row, row + 4, 24.0 * time)
            put(jerk_row, row + 5, 60.0 * t2)
            put(jerk_row, row + 9, -6.0)
            put(snap_row, row + 4, 24.0)
            put(snap_row, row + 5, 120.0 * time)
            put(snap_row, row + 10, -24.0)
            for column, value in enumerate((1.0, time, t2, t3, t4, t5), row):
                put(waypoint_row, column, value)
                put(position_row, column, value)
            put(position_row, row + 6, -1.0)
            for column, value in enumerate(
                    (1.0, 2.0 * time, 3.0 * t2, 4.0 * t3, 5.0 * t4), row + 1):
                put(velocity_row, column, value)
            put(velocity_row, row + 7, -1.0)
            for column, value in enumerate(
                    (2.0, 6.0 * time, 12.0 * t2, 20.0 * t3), row + 2):
                put(acceleration_row, column, value)
            put(acceleration_row, row + 8, -2.0)

        time = times[-1]
        t2, t3, t4, t5 = time**2, time**3, time**4, time**5
        row = 6 * count - 6
        for column, value in enumerate((1.0, time, t2, t3, t4, t5), row):
            put(6 * count - 3, column, value)
        for column, value in enumerate(
                (1.0, 2.0 * time, 3.0 * t2, 4.0 * t3, 5.0 * t4), row + 1):
            put(6 * count - 2, column, value)
        for column, value in enumerate(
                (2.0, 6.0 * time, 12.0 * t2, 20.0 * t3), row + 2):
            put(6 * count - 1, column, value)
        return storage

    @staticmethod
    def jerk_energy(coefficients: np.ndarray, times: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        blocks = coefficients.reshape(len(times), 6, 3)
        c3, c4, c5 = blocks[:, 3], blocks[:, 4], blocks[:, 5]
        t1 = times
        t2, t3 = t1 * t1, t1**3
        t4, t5 = t2 * t2, t2 * t3
        dot33 = np.sum(c3 * c3, axis=1)
        dot43 = np.sum(c4 * c3, axis=1)
        dot44 = np.sum(c4 * c4, axis=1)
        dot53 = np.sum(c5 * c3, axis=1)
        dot54 = np.sum(c5 * c4, axis=1)
        dot55 = np.sum(c5 * c5, axis=1)
        energy = np.sum(36 * dot33 * t1 + 144 * dot43 * t2 + 192 * dot44 * t3
                        + 240 * dot53 * t3 + 720 * dot54 * t4 + 720 * dot55 * t5)
        gradient = np.zeros_like(blocks)
        gradient[:, 3] = 72*c3*t1[:, None] + 144*c4*t2[:, None] + 240*c5*t3[:, None]
        gradient[:, 4] = 144*c3*t2[:, None] + 384*c4*t3[:, None] + 720*c5*t4[:, None]
        gradient[:, 5] = 240*c3*t3[:, None] + 720*c4*t4[:, None] + 1440*c5*t5[:, None]
        grad_times = (36*dot33 + 288*dot43*t1 + 576*dot44*t2 + 720*dot53*t2
                      + 2880*dot54*t3 + 3600*dot55*t4)
        return float(energy), gradient.reshape(-1, 3), grad_times

    def propagate_gradient(self, system: BandedPLU, coefficients: np.ndarray,
                           times: np.ndarray, grad_coefficients: np.ndarray,
                           direct_grad_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        adjoint = system.solve(grad_coefficients, transpose=True)
        grad_points = np.stack([adjoint[6*i + 5] for i in range(self.pieces - 1)], axis=0) \
            if self.pieces > 1 else np.zeros((0, 3))
        grad_times = direct_grad_times.copy()
        blocks = coefficients.reshape(self.pieces, 6, 3)
        duration = times[:, None]
        t2, t3, t4 = duration**2, duration**3, duration**4
        velocity = blocks[:, 1] + 2*duration*blocks[:, 2] + 3*t2*blocks[:, 3] \
            + 4*t3*blocks[:, 4] + 5*t4*blocks[:, 5]
        acceleration = 2*blocks[:, 2] + 6*duration*blocks[:, 3] \
            + 12*t2*blocks[:, 4] + 20*t3*blocks[:, 5]
        jerk = 6*blocks[:, 3] + 24*duration*blocks[:, 4] + 60*t2*blocks[:, 5]
        snap = 24*blocks[:, 4] + 120*duration*blocks[:, 5]
        crackle = 120*blocks[:, 5]
        if self.pieces > 1:
            indices = 6 * np.arange(self.pieces - 1)
            grad_times[:-1] += np.sum(
                -snap[:-1]*adjoint[indices + 3] - crackle[:-1]*adjoint[indices + 4]
                - velocity[:-1]*(adjoint[indices + 5] + adjoint[indices + 6])
                - acceleration[:-1]*adjoint[indices + 7]
                - jerk[:-1]*adjoint[indices + 8], axis=1)
        grad_times[-1] += np.sum(-velocity[-1]*adjoint[-3]
                                 - acceleration[-1]*adjoint[-2] - jerk[-1]*adjoint[-1])
        return grad_points, grad_times


__all__ = ["BandedPLU", "MINCOQuintic"]
