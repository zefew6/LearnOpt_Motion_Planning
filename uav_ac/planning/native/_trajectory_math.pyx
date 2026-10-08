# cython: language_level=3
"""Pure trajectory arithmetic; Python facades own validation and backend choice."""
import numpy as np
cimport cython

cdef double basis_value(double time, int power, int derivative) noexcept:
    cdef double value = 1.0
    cdef int i
    if power < derivative:
        return 0.0
    for i in range(derivative):
        value *= power-i
    for i in range(power-derivative):
        value *= time
    return value


def polynomial_basis_matrix(const double[::1] times, int derivative):
    if derivative < 0 or derivative > 5:
        raise ValueError("derivative must lie in [0, 5]")
    result = np.empty((times.shape[0], 6), dtype=np.float64)
    cdef double[:, ::1] out = result
    cdef Py_ssize_t i
    cdef int k
    for i in range(times.shape[0]):
        for k in range(6):
            out[i, k] = basis_value(times[i], k, derivative)
    return result


def polynomial_bases(const double[::1] times):
    result = np.empty((5, times.shape[0], 6), dtype=np.float64)
    cdef double[:, :, ::1] out = result
    cdef Py_ssize_t i
    cdef int k, derivative
    for derivative in range(5):
        for i in range(times.shape[0]):
            for k in range(6):
                out[derivative, i, k] = basis_value(times[i], k, derivative)
    return result


def evaluate_quintic(const double[:, :, ::1] coefficients,
                     const Py_ssize_t[::1] pieces, const double[::1] local,
                     int derivative):
    if coefficients.shape[1] != 6 or pieces.shape[0] != local.shape[0]:
        raise ValueError("invalid polynomial batch shape")
    if derivative < 0 or derivative > 5:
        raise ValueError("derivative must lie in [0, 5]")
    result = np.zeros((local.shape[0], coefficients.shape[2]), dtype=np.float64)
    cdef double[:, ::1] out = result
    cdef Py_ssize_t i, d, piece
    cdef int k
    cdef double basis
    for i in range(local.shape[0]):
        piece = pieces[i]
        if piece < 0 or piece >= coefficients.shape[0]:
            raise ValueError("piece index out of bounds")
        for k in range(derivative, 6):
            basis = basis_value(local[i], k, derivative)
            for d in range(coefficients.shape[2]):
                out[i, d] += basis * coefficients[piece, k, d]
    return result


cdef void put(double[:, :] storage, Py_ssize_t row, Py_ssize_t column, double value):
    storage[12 + row-column, column] = value


def band_storage(const double[::1] times):
    cdef Py_ssize_t count = times.shape[0], i, row, k
    if count < 1:
        raise ValueError("times must contain at least one duration")
    result = np.zeros((19, 6*count), dtype=np.float64, order='F')
    cdef double[:, :] storage = result
    cdef double t
    put(storage, 0, 0, 1.0)
    put(storage, 1, 1, 1.0)
    put(storage, 2, 2, 2.0)
    for i in range(count):
        t = times[i]
        row = 6*i
        if i < count-1:
            put(storage, row+3, row+3, 6.0)
            put(storage, row+3, row+4, 24.0*t)
            put(storage, row+3, row+5, 60.0*t*t)
            put(storage, row+3, row+9, -6.0)
            put(storage, row+4, row+4, 24.0)
            put(storage, row+4, row+5, 120.0*t)
            put(storage, row+4, row+10, -24.0)
            for k in range(6):
                put(storage, row+5, row+k, basis_value(t, k, 0))
                put(storage, row+6, row+k, basis_value(t, k, 0))
            put(storage, row+6, row+6, -1.0)
            for k in range(1, 6):
                put(storage, row+7, row+k, basis_value(t, k, 1))
            put(storage, row+7, row+7, -1.0)
            for k in range(2, 6):
                put(storage, row+8, row+k, basis_value(t, k, 2))
            put(storage, row+8, row+8, -2.0)
        else:
            for k in range(6):
                put(storage, row+3, row+k, basis_value(t, k, 0))
            for k in range(1, 6):
                put(storage, row+4, row+k, basis_value(t, k, 1))
            for k in range(2, 6):
                put(storage, row+5, row+k, basis_value(t, k, 2))
    return result


def jerk_energy(const double[:, :, ::1] blocks, const double[::1] times,
                const double[::1] weights):
    if blocks.shape[0] != times.shape[0] or blocks.shape[1] != 6 or blocks.shape[2] != weights.shape[0]:
        raise ValueError("invalid jerk energy input shapes")
    gradient = np.zeros((blocks.shape[0], 6, blocks.shape[2]), dtype=np.float64)
    grad_times = np.zeros(times.shape[0], dtype=np.float64)
    cdef double[:, :, ::1] gc = gradient
    cdef double[::1] gt = grad_times
    cdef Py_ssize_t i, d
    cdef double energy=0.0, t, t2, t3, t4, t5, c3, c4, c5, w
    cdef double dot33, dot43, dot44, dot53, dot54, dot55
    for i in range(times.shape[0]):
        t=times[i]; t2=t*t; t3=t2*t; t4=t2*t2; t5=t2*t3
        dot33=0.; dot43=0.; dot44=0.; dot53=0.; dot54=0.; dot55=0.
        for d in range(weights.shape[0]):
            c3=blocks[i,3,d]; c4=blocks[i,4,d]; c5=blocks[i,5,d]; w=weights[d]
            dot33+=c3*c3*w; dot43+=c4*c3*w; dot44+=c4*c4*w
            dot53+=c5*c3*w; dot54+=c5*c4*w; dot55+=c5*c5*w
            gc[i,3,d]=w*(72*c3*t+144*c4*t2+240*c5*t3)
            gc[i,4,d]=w*(144*c3*t2+384*c4*t3+720*c5*t4)
            gc[i,5,d]=w*(240*c3*t3+720*c4*t4+1440*c5*t5)
        energy+=36*dot33*t+144*dot43*t2+192*dot44*t3+240*dot53*t3+720*dot54*t4+720*dot55*t5
        gt[i]=36*dot33+288*dot43*t+576*dot44*t2+720*dot53*t2+2880*dot54*t3+3600*dot55*t4
    return energy, gradient.reshape(-1, 3), grad_times


def adjoint_gradients(const double[:, :, ::1] blocks, const double[::1] times,
                      const double[:, ::1] adjoint, const double[::1] direct):
    cdef Py_ssize_t n=times.shape[0], dimensions=blocks.shape[2], i, d, row
    if n < 1 or blocks.shape[0] != n or blocks.shape[1] != 6 or adjoint.shape[0] != 6*n or adjoint.shape[1] != dimensions or direct.shape[0] != n:
        raise ValueError("invalid adjoint input shapes")
    points = np.empty((n-1, dimensions), dtype=np.float64) if n > 1 else np.zeros((0, 3))
    result = np.array(direct, dtype=np.float64, copy=True)
    cdef double[:, ::1] gp = points
    cdef double[::1] gt = result
    cdef double t, t2, t3, t4, velocity, acceleration, jerk, snap, crackle
    for i in range(n):
        row=6*i; t=times[i]; t2=t*t; t3=t2*t; t4=t2*t2
        for d in range(dimensions):
            velocity=blocks[i,1,d]+2*t*blocks[i,2,d]+3*t2*blocks[i,3,d]+4*t3*blocks[i,4,d]+5*t4*blocks[i,5,d]
            acceleration=2*blocks[i,2,d]+6*t*blocks[i,3,d]+12*t2*blocks[i,4,d]+20*t3*blocks[i,5,d]
            jerk=6*blocks[i,3,d]+24*t*blocks[i,4,d]+60*t2*blocks[i,5,d]
            if i < n-1:
                snap=24*blocks[i,4,d]+120*t*blocks[i,5,d]
                crackle=120*blocks[i,5,d]
                gp[i,d]=adjoint[row+5,d]
                gt[i] += -snap*adjoint[row+3,d]-crackle*adjoint[row+4,d]-velocity*(adjoint[row+5,d]+adjoint[row+6,d])-acceleration*adjoint[row+7,d]-jerk*adjoint[row+8,d]
            else:
                gt[i] += -velocity*adjoint[row+3,d]-acceleration*adjoint[row+4,d]-jerk*adjoint[row+5,d]
    return points, result
