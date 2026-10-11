# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Scalar analytic aerial constraints and fixed-quadrature accumulation."""
import numpy as np
from libc.math cimport sqrt, sin, cos, acos

cdef struct Dual:
    double v
    double d[8]

cdef Dual constant(double v) noexcept nogil:
    cdef Dual r
    cdef int k
    r.v = v
    for k in range(8): r.d[k] = 0
    return r

cdef Dual add(Dual a, Dual b) noexcept nogil:
    cdef Dual r
    cdef int k
    r.v = a.v+b.v
    for k in range(8): r.d[k] = a.d[k]+b.d[k]
    return r

cdef Dual sub(Dual a, Dual b) noexcept nogil:
    cdef Dual r
    cdef int k
    r.v = a.v-b.v
    for k in range(8): r.d[k] = a.d[k]-b.d[k]
    return r

cdef Dual mul(Dual a, Dual b) noexcept nogil:
    cdef Dual r
    cdef int k
    r.v = a.v*b.v
    for k in range(8): r.d[k] = a.d[k]*b.v+b.d[k]*a.v
    return r

cdef Dual div(Dual a, Dual b) noexcept nogil:
    cdef Dual r
    cdef int k
    r.v = a.v/b.v
    for k in range(8): r.d[k] = (a.d[k]*b.v-b.d[k]*a.v)/(b.v*b.v)
    return r

cdef Dual root(Dual a) noexcept nogil:
    # Preserve the reference's regularized value AND derivative convention.
    cdef Dual r
    cdef int k
    r.v = sqrt(max(a.v, 1e-16))
    for k in range(8): r.d[k] = a.d[k]/(2*r.v)
    return r

cdef Dual dot(Dual* a, Dual* b) noexcept nogil:
    cdef Dual r = constant(0)
    cdef int i
    for i in range(3): r = add(r, mul(a[i], b[i]))
    return r

cdef void cross(Dual* a, Dual* b, Dual* r) noexcept nogil:
    r[0] = sub(mul(a[1],b[2]), mul(a[2],b[1]))
    r[1] = sub(mul(a[2],b[0]), mul(a[0],b[2]))
    r[2] = sub(mul(a[0],b[1]), mul(a[1],b[0]))

cdef Dual rate_squared(const double[:] a, const double[:] j, double yaw,
                       double yaw_rate, double gravity) noexcept nogil:
    cdef Dual force[3]
    cdef Dual fr[3]
    cdef Dual z[3]
    cdef Dual zd[3]
    cdef Dual heading[3]
    cdef Dual axis[3]
    cdef Dual y[3]
    cdef Dual mag, cn, along, align, axial, yr, result
    cdef int i
    for i in range(3):
        force[i] = constant(-a[i]+(gravity if i==2 else 0))
        force[i].d[i] = -1
        fr[i] = constant(-j[i])
        fr[i].d[3+i] = -1
    mag = root(dot(force, force))
    for i in range(3): z[i] = div(force[i],mag)
    along = dot(z,fr)
    for i in range(3): zd[i] = div(sub(fr[i],mul(z[i],along)),mag)
    heading[0] = constant(cos(yaw)); heading[0].d[6] = -sin(yaw)
    heading[1] = constant(sin(yaw)); heading[1].d[6] = cos(yaw)
    heading[2] = constant(0)
    cross(z,heading,axis)
    cn = root(dot(axis,axis))
    cn.v = max(cn.v,1e-8)
    for i in range(3): y[i] = div(axis[i],cn)
    align = dot(z,heading)
    yr = constant(yaw_rate); yr.d[7] = 1
    axial = add(div(sub(constant(0),mul(dot(y,zd),align)),cn),
                div(mul(z[2],yr),mul(cn,cn)))
    return add(dot(zd,zd),mul(axial,axial))


def flatness_body_rate_squared(const double[:] acceleration, const double[:] jerk,
                               double yaw, double yaw_rate, double gravity):
    if acceleration.shape[0] != 3 or jerk.shape[0] != 3:
        raise ValueError('acceleration and jerk must be 3-vectors')
    cdef Dual result = rate_squared(acceleration,jerk,yaw,yaw_rate,gravity)
    gradient = np.empty(8)
    cdef double[:] g = gradient
    cdef int k
    for k in range(8): g[k] = result.d[k]
    return result.v, gradient

cdef double penalty(double violation, double epsilon, double* derivative) noexcept nogil:
    cdef double ratio
    if violation <= 0:
        derivative[0] = 0
        return 0
    if violation > epsilon:
        derivative[0] = 1
        return violation-0.5*epsilon
    ratio = violation/epsilon
    derivative[0] = ratio*ratio*(-0.5*ratio+3*(epsilon-0.5*violation)/epsilon)
    return (epsilon-0.5*violation)*ratio*ratio*ratio

cdef void accumulate(double violation, double weight, double epsilon, int order,
                     double* jac, double[:, :] gradients, double* cost,
                     double* maximum) noexcept nogil:
    cdef double derivative
    cdef int k
    cost[0] += weight*penalty(violation,epsilon,&derivative)
    for k in range(8): gradients[order,k] += weight*derivative*jac[k]
    maximum[0] = max(maximum[0],violation)


def physical_cost_gradient(const double[:] sigma, const double[:] velocity,
                           const double[:] acceleration, const double[:] jerk,
                           const double[:] lower, const double[:] upper,
                           const double[:, :] bounds, const double[:] parameters,
                           const double[:] joint_v, const double[:] joint_a):
    """Parameters: weight, epsilon, speed, acceleration, yaw rate, yaw acceleration,
    body rate, gravity, mass, motor min/max thrust, maximum tilt."""
    if (sigma.shape[0] != 8 or velocity.shape[0] != 8 or acceleration.shape[0] != 8
        or jerk.shape[0] != 8 or lower.shape[0] != 4 or upper.shape[0] != 4
        or bounds.shape[0] != 2 or bounds.shape[1] != 3 or parameters.shape[0] != 12
        or joint_v.shape[0] != 4 or joint_a.shape[0] != 4):
        raise ValueError('invalid physical constraint input shape')
    gradients = np.zeros((4,8))
    cdef double[:, :] g = gradients
    cdef double w=parameters[0], eps=parameters[1], cost=0, maximum=-float('inf')
    cdef double jac[8]
    cdef double f[3]
    cdef double z[3]
    cdef double v, rho, thrust, ct, st, derivative
    cdef int i,k,order
    cdef Dual br
    for k in range(8): jac[k]=0
    for order in range(1,3):
        v=0
        for i in range(3):
            jac[i]=2*(velocity[i] if order==1 else acceleration[i])
            v+=(velocity[i] if order==1 else acceleration[i])**2
        accumulate(v-parameters[1+order]**2,w,eps,order,jac,g,&cost,&maximum)
        for i in range(3): jac[i]=0
        jac[3]=2*(velocity[3] if order==1 else acceleration[3])
        v=(velocity[3] if order==1 else acceleration[3])**2-parameters[3+order]**2
        accumulate(v,w,eps,order,jac,g,&cost,&maximum)
        jac[3]=0
    for i in range(4):
        jac[4+i]=1
        accumulate(sigma[4+i]-upper[i],w,eps,0,jac,g,&cost,&maximum)
        jac[4+i]=-1
        accumulate(lower[i]-sigma[4+i],w,eps,0,jac,g,&cost,&maximum)
        jac[4+i]=2*velocity[4+i]
        accumulate(velocity[4+i]**2-joint_v[i]**2,w,eps,1,jac,g,&cost,&maximum)
        jac[4+i]=2*acceleration[4+i]
        accumulate(acceleration[4+i]**2-joint_a[i]**2,w,eps,2,jac,g,&cost,&maximum)
        jac[4+i]=0
    rho=0
    for i in range(3):
        f[i]=-acceleration[i]+(parameters[7] if i==2 else 0)
        rho+=f[i]*f[i]
    rho=max(sqrt(rho),1e-8)
    thrust=parameters[8]*rho
    for i in range(3):
        z[i]=f[i]/rho
        jac[i]=parameters[8]*z[i]
    accumulate(4*parameters[9]-thrust,w,eps,2,jac,g,&cost,&maximum)
    for i in range(3): jac[i]=-jac[i]
    accumulate(thrust-4*parameters[10],w,eps,2,jac,g,&cost,&maximum)
    ct=max(-1.,min(1.,z[2])); st=max(sqrt(max(1-ct*ct,0)),1e-8)
    for i in range(3): jac[i]=((1. if i==2 else 0)-ct*z[i])/(rho*st)
    accumulate(acos(ct)-parameters[11],w,eps,2,jac,g,&cost,&maximum)
    br=rate_squared(acceleration,jerk,sigma[3],velocity[3],parameters[7])
    v=br.v-parameters[6]**2
    cost+=w*penalty(v,eps,&derivative)
    g[0,3]+=w*derivative*br.d[6]; g[1,3]+=w*derivative*br.d[7]
    for i in range(3):
        g[2,i]+=w*derivative*br.d[i]
        g[3,i]+=w*derivative*br.d[3+i]
        jac[i]=0
    maximum=max(maximum,v)
    for i in range(3):
        jac[i]=-1
        accumulate(bounds[0,i]-sigma[i],w,eps,0,jac,g,&cost,&maximum)
        jac[i]=1
        accumulate(sigma[i]-bounds[1,i],w,eps,0,jac,g,&cost,&maximum)
        jac[i]=0
    return cost, [gradients[i] for i in range(4)], maximum


def flatness_attitude_tangent_jacobian(const double[:] acceleration, double yaw,
                                      double gravity=9.81):
    if acceleration.shape[0] != 3 or not np.all(np.isfinite(acceleration)):
        raise ValueError('acceleration must be a finite 3-vector')
    cdef Dual f[3]
    cdef Dual z[3]
    cdef Dual h[3]
    cdef Dual axis[3]
    cdef Dual y[3]
    cdef Dual x[3]
    cdef Dual mag, cn
    cdef int i,k
    # Projected normalization derivative exactly matches the reference,
    # including its convention below the normalization floor.
    for i in range(3): f[i]=constant(-acceleration[i]+(gravity if i==2 else 0))
    mag=root(dot(f,f))
    for i in range(3): z[i]=div(f[i],mag)
    for i in range(3):
        for k in range(3): z[i].d[k]=(-(1. if i==k else 0)+z[i].v*z[k].v)/mag.v
    h[0]=constant(cos(yaw)); h[0].d[3]=-sin(yaw)
    h[1]=constant(sin(yaw)); h[1].d[3]=cos(yaw)
    h[2]=constant(0)
    cross(z,h,axis)
    cn=root(dot(axis,axis))
    for i in range(3): y[i]=div(axis[i],cn)
    cross(y,z,x)
    rotation=np.empty((3,3)); tangent=np.empty((3,4))
    cdef double[:, :] r=rotation, t=tangent
    for i in range(3):
        r[i,0]=x[i].v; r[i,1]=y[i].v; r[i,2]=z[i].v
    for k in range(4):
        t[0,k]=x[2].d[k]*x[1].v+y[2].d[k]*y[1].v+z[2].d[k]*z[1].v
        t[1,k]=x[0].d[k]*x[2].v+y[0].d[k]*y[2].v+z[0].d[k]*z[2].v
        t[2,k]=x[1].d[k]*x[0].v+y[1].d[k]*y[0].v+z[1].d[k]*z[0].v
    return rotation,tangent


def integrated_penalty(const double[:] durations, const double[:, :, :] coefficients,
                       int resolution, callback):
    """Keep callback order/ownership; compile polynomial and gradient arithmetic."""
    cdef int pieces=durations.shape[0], piece,sample,order,k,d,power
    if resolution < 1 or coefficients.shape[0] != pieces or coefficients.shape[1] != 6 or coefficients.shape[2] != 8:
        raise ValueError('invalid integration input shape or resolution')
    alpha_array=np.linspace(0.,1.,resolution+1)
    cdef double[:] alpha=alpha_array
    grad_coefficients=np.zeros((pieces,6,8)); grad_times=np.zeros(pieces)
    cdef double[:, :, :] gc=grad_coefficients
    cdef double[:] gt=grad_times
    cdef double[:, :, :] bases, values
    cdef double[:, :] grads
    cdef double duration,local,weight,scale,cost,total=0,derivative,v,factor
    cdef double maximum=-float('inf'), minimum=float('inf'), violation,clearance
    for piece in range(pieces):
        duration=durations[piece]
        bases_array=np.zeros((resolution+1,5,6))
        # Each callback receives distinct views backed by a fresh piece array.
        # Retained callback inputs are never overwritten by later samples/pieces.
        values_array=np.zeros((resolution+1,5,8))
        bases=bases_array; values=values_array
        for sample in range(resolution+1):
            local=alpha[sample]*duration
            for order in range(5):
                for k in range(order,6):
                    factor=1
                    for power in range(order): factor*=k-power
                    bases[sample,order,k]=factor*local**(k-order)
                    for d in range(8): values[sample,order,d]+=bases[sample,order,k]*coefficients[piece,k,d]
        for sample in range(resolution+1):
            cost,gradient_list,violation,clearance=callback(
                *(values_array[sample,order] for order in range(4)))
            gradient_array=np.ascontiguousarray(gradient_list,dtype=np.float64)
            if gradient_array.shape != (4,8): raise ValueError('callback gradients must be (4, 8)')
            grads=gradient_array
            weight=0.5 if sample==0 or sample==resolution else 1
            scale=duration/resolution*weight
            total+=scale*cost
            derivative=0
            for order in range(4):
                for d in range(8):
                    derivative+=grads[order,d]*values[sample,order+1,d]
                    for k in range(6): gc[piece,k,d]+=scale*bases[sample,order,k]*grads[order,d]
            gt[piece]+=scale*alpha[sample]*derivative+weight/resolution*cost
            maximum=max(maximum,violation); minimum=min(minimum,clearance)
    return total,grad_coefficients,grad_times,maximum,minimum


def exact_collision_penalty(double cost, const double[:] state_gradient,
                            const double[:] acceleration_gradient,
                            const double[:] distances, const double[:, :] jacobians,
                            const signed char[:] kinds, const double[:, :] tangent,
                            double world_margin, double self_margin,
                            double world_weight, double self_weight, double epsilon,
                            double obstacle_violation, double self_violation,
                            double payload_violation, double minimum_clearance):
    """Accumulate exact witness penalties without altering geometry or caller state."""
    cdef Py_ssize_t count=distances.shape[0], pair
    if (state_gradient.shape[0] != 8 or acceleration_gradient.shape[0] != 8
        or jacobians.shape[0] != count or jacobians.shape[1] != 10
        or kinds.shape[0] != count or tangent.shape[0] != 3 or tangent.shape[1] != 4):
        raise ValueError('invalid exact collision accumulation input shape')
    gradient_array=np.array(state_gradient,dtype=np.float64,copy=True)
    acceleration_array=np.array(acceleration_gradient,dtype=np.float64,copy=True)
    cdef double[:] gradient=gradient_array, acceleration=acceleration_array
    cdef double margin,weight,violation,derivative,yaw_gradient,acc_gradient
    cdef int kind,i,j
    for pair in range(count):
        kind=kinds[pair]
        margin=world_margin if kind==1 else self_margin
        weight=world_weight if kind==1 else self_weight
        violation=margin-distances[pair]
        cost+=weight*penalty(violation,epsilon,&derivative)
        derivative*=weight
        if derivative != 0:
            yaw_gradient=0
            for i in range(3):
                gradient[i]-=derivative*jacobians[pair,i]
                yaw_gradient+=jacobians[pair,3+i]*tangent[i,3]
                acc_gradient=0
                for j in range(3): acc_gradient+=jacobians[pair,3+j]*tangent[j,i]
                acceleration[i]-=derivative*acc_gradient
            gradient[3]-=derivative*yaw_gradient
            for i in range(4): gradient[4+i]-=derivative*jacobians[pair,6+i]
        if kind==1: obstacle_violation=max(obstacle_violation,violation)
        elif kind==2: self_violation=max(self_violation,violation)
        else: payload_violation=max(payload_violation,violation)
        minimum_clearance=min(minimum_clearance,distances[pair]-margin)
    return (cost,gradient_array,acceleration_array,obstacle_violation,
            minimum_clearance,self_violation,payload_violation)


# Model-derived FK for the fixed/hinge/slide subtree of a floating robot.
# The robot wrapper validates metadata and public configurations before entry.
from libc.stdint cimport int64_t


cdef void _quat_product(const double* a, const double* b, double* out) noexcept nogil:
    out[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
    out[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
    out[2] = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
    out[3] = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]


cdef void _quat_rotate(const double* q, const double* p, double* out) noexcept nogil:
    cdef double tx = 2.*(q[2]*p[2]-q[3]*p[1])
    cdef double ty = 2.*(q[3]*p[0]-q[1]*p[2])
    cdef double tz = 2.*(q[1]*p[1]-q[2]*p[0])
    out[0] = p[0]+q[0]*tx+q[2]*tz-q[3]*ty
    out[1] = p[1]+q[0]*ty+q[3]*tx-q[1]*tz
    out[2] = p[2]+q[0]*tz+q[1]*ty-q[2]*tx


def batch_point_positions(const double[:, ::1] configurations,
                          const int64_t[::1] parents,
                          const double[:, ::1] offsets,
                          const double[:, ::1] orientations,
                          const int64_t[::1] joint_types,
                          const int64_t[::1] joint_columns,
                          const double[:, ::1] axes,
                          const double[:, ::1] anchors,
                          const double[::1] references,
                          const int64_t[::1] point_bodies,
                          const double[:, ::1] local_points):
    """NED/FRD configurations to NED points; no live/scratch MuJoCo mutation."""
    result_array = np.empty((configurations.shape[0], local_points.shape[0], 3))
    transform_array = np.empty((parents.shape[0], 7))
    cdef double[:, :, ::1] result = result_array
    cdef double[:, ::1] transform = transform_array
    cdef Py_ssize_t batch, body, parent, point, axis
    cdef double value, norm, sign
    cdef double position[3]
    cdef double quaternion[4]
    cdef double joint_quaternion[4]
    cdef double local_quaternion[4]
    cdef double local_position[3]
    cdef double rotated[3]
    cdef double vector[3]
    cdef double parent_quaternion[4]
    with nogil:
        for batch in range(configurations.shape[0]):
            for axis in range(3):
                transform[0, axis] = configurations[batch, axis]*(1. if axis == 0 else -1.)
            norm = 0.
            for axis in range(4):
                norm += configurations[batch, axis+3]*configurations[batch, axis+3]
            norm = sqrt(norm)
            for axis in range(4):
                sign = 1. if axis < 2 else -1.
                transform[0, axis+3] = sign*configurations[batch, axis+3]/norm
            for body in range(1, parents.shape[0]):
                parent = parents[body]
                for axis in range(3):
                    local_position[axis] = offsets[body, axis]
                for axis in range(4):
                    local_quaternion[axis] = orientations[body, axis]
                    parent_quaternion[axis] = transform[parent, axis+3]
                if joint_types[body] != -1:
                    value = configurations[batch, joint_columns[body]]
                    if joint_columns[body] == 11:
                        value = (value-.020)*.5
                    value -= references[body]
                    if joint_types[body] == 2:  # MuJoCo slide
                        for axis in range(3):
                            vector[axis] = axes[body, axis]*value
                        _quat_rotate(local_quaternion, vector, rotated)
                        for axis in range(3):
                            local_position[axis] += rotated[axis]
                    else:  # MuJoCo hinge, including an off-origin anchor
                        joint_quaternion[0] = cos(value*.5)
                        for axis in range(3):
                            joint_quaternion[axis+1] = axes[body, axis]*sin(value*.5)
                            vector[axis] = anchors[body, axis]
                        _quat_rotate(joint_quaternion, vector, rotated)
                        for axis in range(3):
                            vector[axis] -= rotated[axis]
                        _quat_rotate(local_quaternion, vector, rotated)
                        for axis in range(3):
                            local_position[axis] += rotated[axis]
                        _quat_product(local_quaternion, joint_quaternion, quaternion)
                        for axis in range(4):
                            local_quaternion[axis] = quaternion[axis]
                _quat_rotate(parent_quaternion, local_position, position)
                _quat_product(parent_quaternion, local_quaternion, quaternion)
                for axis in range(3):
                    transform[body, axis] = transform[parent, axis]+position[axis]
                for axis in range(4):
                    transform[body, axis+3] = quaternion[axis]
            for point in range(local_points.shape[0]):
                body = point_bodies[point]
                for axis in range(4):
                    quaternion[axis] = transform[body, axis+3]
                for axis in range(3):
                    vector[axis] = local_points[point, axis]
                _quat_rotate(quaternion, vector, rotated)
                for axis in range(3):
                    result[batch, point, axis] = (transform[body, axis]+rotated[axis])*(1. if axis == 0 else -1.)
    return result_array
