"""Fused numeric RRT broad phase; no model state or geometric approximation."""

from libc.math cimport ceil, nearbyint, sqrt, fabs, INFINITY
from libc.stdint cimport int64_t
import numpy as np


def query(
        const int64_t[:, :, ::1] prefix, const double[::1] origin,
        double resolution, const double[:, :, ::1] points,
        const double[::1] radii, double margin,
        const int64_t[:, ::1] self_pairs, const int64_t[:, ::1] payload_pairs,
        const int64_t[::1] self_indices, const int64_t[::1] payload_indices,
        const int64_t[::1] geom_indices, const double[:, ::1] world_lower,
        const double[:, ::1] world_upper, const int64_t[:, ::1] world_table,
        Py_ssize_t query_count, double self_clearance, double world_clearance,
        bint narrow_phase):
    cdef Py_ssize_t batches = points.shape[0], spheres = points.shape[1]
    cdef Py_ssize_t nx = prefix.shape[0]-2, ny = prefix.shape[1]-2, nz = prefix.shape[2]-2
    env_array = np.full(batches, -np.inf if spheres == 0 else -1., dtype=np.float64)
    self_array = np.full(batches, -np.inf, dtype=np.float64)
    payload_array = np.full(batches, -np.inf, dtype=np.float64)
    selected_array = np.zeros((batches, query_count), dtype=np.uint8)
    worlds_array = np.zeros(batches, dtype=np.int64)
    outside_array = np.zeros(batches, dtype=np.uint8)
    world_seen_array = np.zeros((batches, query_count), dtype=np.uint8)
    cdef double[::1] env = env_array, self_violation = self_array, payload_violation = payload_array
    cdef unsigned char[:, ::1] selected = selected_array, seen = world_seen_array
    cdef int64_t[::1] worlds = worlds_array
    cdef unsigned char[::1] outside = outside_array
    cdef Py_ssize_t batch, sphere, pair, first, second, world, geometry, query_index
    cdef Py_ssize_t x, y, z, x0, y0, z0, x1, y1, z1
    cdef double rx, ry, rz, cells, dx, dy, dz, value, limit
    cdef int64_t occupied
    cdef bint hit, out
    with nogil:
        for batch in range(batches):
            for pair in range(self_pairs.shape[0]):
                first, second = self_pairs[pair, 0], self_pairs[pair, 1]
                dx = points[batch, first, 0]-points[batch, second, 0]
                dy = points[batch, first, 1]-points[batch, second, 1]
                dz = points[batch, first, 2]-points[batch, second, 2]
                value = radii[first]+radii[second]+self_clearance-sqrt(dx*dx+dy*dy+dz*dz)
                if value > self_violation[batch]:
                    self_violation[batch] = value
                if value > 0.:
                    selected[batch, self_indices[pair]] = 1
            for pair in range(payload_pairs.shape[0]):
                first, second = payload_pairs[pair, 0], payload_pairs[pair, 1]
                dx = points[batch, first, 0]-points[batch, second, 0]
                dy = points[batch, first, 1]-points[batch, second, 1]
                dz = points[batch, first, 2]-points[batch, second, 2]
                value = radii[first]+radii[second]+self_clearance-sqrt(dx*dx+dy*dy+dz*dz)
                if value > payload_violation[batch]:
                    payload_violation[batch] = value
                if value > 0.:
                    selected[batch, payload_indices[pair]] = 1
            for sphere in range(spheres):
                rx = nearbyint((points[batch, sphere, 0]-origin[0])/resolution)
                ry = nearbyint((points[batch, sphere, 1]-origin[1])/resolution)
                rz = nearbyint((points[batch, sphere, 2]-origin[2])/resolution)
                out = rx < 0 or rx >= nx or ry < 0 or ry >= ny or rz < 0 or rz >= nz
                hit = False
                if out:
                    outside[batch] = 1
                else:
                    x, y, z = <Py_ssize_t>rx, <Py_ssize_t>ry, <Py_ssize_t>rz
                    cells = ceil((radii[sphere]+margin)/resolution)+1.
                    x0 = 0 if cells > x else x-<Py_ssize_t>cells
                    y0 = 0 if cells > y else y-<Py_ssize_t>cells
                    z0 = 0 if cells > z else z-<Py_ssize_t>cells
                    x1 = nx if cells >= nx-1-x else x+<Py_ssize_t>cells+1
                    y1 = ny if cells >= ny-1-y else y+<Py_ssize_t>cells+1
                    z1 = nz if cells >= nz-1-z else z+<Py_ssize_t>cells+1
                    occupied = (prefix[x1,y1,z1]-prefix[x0,y1,z1]-prefix[x1,y0,z1]
                                -prefix[x1,y1,z0]+prefix[x0,y0,z1]+prefix[x0,y1,z0]
                                +prefix[x1,y0,z0]-prefix[x0,y0,z0])
                    hit = occupied > 0
                if not (hit or out):
                    continue
                env[batch] = 1.
                geometry = geom_indices[sphere]
                if not narrow_phase or geometry < 0:
                    continue
                limit = radii[sphere]+world_clearance
                for world in range(world_lower.shape[0]):
                    query_index = world_table[geometry, world]
                    if query_index < 0:
                        continue
                    dx = dy = dz = 0.
                    if points[batch,sphere,0] < world_lower[world,0]:
                        dx = points[batch,sphere,0]-world_lower[world,0]
                    elif points[batch,sphere,0] > world_upper[world,0]:
                        dx = points[batch,sphere,0]-world_upper[world,0]
                    if points[batch,sphere,1] < world_lower[world,1]:
                        dy = points[batch,sphere,1]-world_lower[world,1]
                    elif points[batch,sphere,1] > world_upper[world,1]:
                        dy = points[batch,sphere,1]-world_upper[world,1]
                    if points[batch,sphere,2] < world_lower[world,2]:
                        dz = points[batch,sphere,2]-world_lower[world,2]
                    elif points[batch,sphere,2] > world_upper[world,2]:
                        dz = points[batch,sphere,2]-world_upper[world,2]
                    if dx*dx+dy*dy+dz*dz <= limit*limit+1e-12:
                        selected[batch, query_index] = 1
                        if not seen[batch, query_index]:
                            seen[batch, query_index] = 1
                            worlds[batch] += 1
    return (env_array, self_array, payload_array, selected_array.view(np.bool_),
            worlds_array, outside_array.view(np.bool_))


ctypedef fused field_t:
    float
    double


cdef bint _pairs_collide(const double[:, :, ::1] points, Py_ssize_t batch,
                        const double[::1] radii,
                        const int64_t[:, ::1] pairs, double clearance) noexcept nogil:
    cdef Py_ssize_t pair, a, b
    cdef double dx, dy, dz, limit, distance2
    for pair in range(pairs.shape[0]):
        a, b = pairs[pair, 0], pairs[pair, 1]
        limit = radii[a]+radii[b]+clearance
        dx = points[batch, a, 0]-points[batch, b, 0]
        dy = points[batch, a, 1]-points[batch, b, 1]
        dz = points[batch, a, 2]-points[batch, b, 2]
        distance2 = dx*dx+dy*dy+dz*dz
        if limit > 0. and distance2 < limit*limit:
            # Retain the reference subtraction at contact: squaring a
            # tolerance-shifted radius changes decisions by a rounding ulp.
            if limit-sqrt(distance2) > 1e-10:
                return True
    return False


def sphere_feasible(const field_t[:, :, ::1] values, const double[::1] origin,
                    double resolution, const double[:, :, ::1] points,
                    const double[::1] radii, const int64_t[:, ::1] self_pairs,
                    const int64_t[:, ::1] payload_pairs, double margin,
                    double discretization_margin, double self_clearance,
                    bint with_uncertainty=False):
    """Boolean manual-sphere checks, matching the NumPy/SciPy reference.

    Metadata is validated by WholeBodyCollision. Both float32 and float64
    fields are read directly, without copying a scene-sized ESDF.
    """
    result = np.ones(points.shape[0], dtype=np.bool_)
    uncertain_array = np.zeros(points.shape[0], dtype=np.bool_) if with_uncertainty else np.empty(0, dtype=np.bool_)
    cdef unsigned char[::1] valid = result.view(np.uint8)
    cdef unsigned char[::1] uncertain = uncertain_array.view(np.uint8)
    cdef Py_ssize_t batch, sphere, axis, corner, x, y, z
    cdef Py_ssize_t index[3]
    cdef double upper[3]
    cdef double fraction[3]
    cdef double coordinate, clipped, delta, outside2, distance, weight, required
    cdef bint reject
    for axis in range(3):
        upper[axis] = origin[axis]+resolution*(values.shape[axis]-1)
    with nogil:
        for batch in range(points.shape[0]):
            # Self collision can reject before any map interpolation.
            if (_pairs_collide(points, batch, radii, self_pairs, self_clearance)
                    or _pairs_collide(points, batch, radii, payload_pairs, self_clearance)):
                valid[batch] = 0
                continue
            for sphere in range(points.shape[1]):
                reject = False
                outside2 = 0.
                for axis in range(3):
                    coordinate = points[batch, sphere, axis]
                    if (origin[axis]+radii[sphere]-coordinate > 1e-10
                            or coordinate-(upper[axis]-radii[sphere]) > 1e-10):
                        reject = True
                        break
                    clipped = coordinate
                    if clipped < origin[axis]:
                        clipped = origin[axis]
                    elif clipped > upper[axis]:
                        clipped = upper[axis]
                    delta = coordinate-clipped
                    outside2 += delta*delta
                    coordinate = (clipped-origin[axis])/resolution
                    index[axis] = <Py_ssize_t>coordinate
                    if index[axis] >= values.shape[axis]-1:
                        index[axis] = values.shape[axis]-2
                    fraction[axis] = coordinate-index[axis]
                if reject:
                    valid[batch] = 0
                    break
                distance = 0.
                for corner in range(8):
                    x, y, z = corner & 1, (corner >> 1) & 1, (corner >> 2) & 1
                    weight = (fraction[0] if x else 1.-fraction[0])
                    weight *= fraction[1] if y else 1.-fraction[1]
                    weight *= fraction[2] if z else 1.-fraction[2]
                    distance += weight*values[index[0]+x, index[1]+y, index[2]+z]
                required = radii[sphere]+margin+discretization_margin+sqrt(outside2)
                if with_uncertainty and fabs(required-distance-1e-10) <= 1e-12*(1.+fabs(required)+fabs(distance)):
                    # Direct trilinear evaluation and SciPy's B-spline can
                    # differ by rounding ulps. Confirm tolerance ties upstream.
                    uncertain[batch] = 1
                if required-distance > 1e-10:
                    valid[batch] = 0
                    break
    return (result, uncertain_array) if with_uncertainty else result
