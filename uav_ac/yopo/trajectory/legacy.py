"""Four-piece MINCO with both lateral-jerk endpoint constraints eliminated."""
import numpy as np
from uav_ac.yopo.contracts import finite_array, PlannerObservation, DepthSpec
from uav_ac.yopo.model.encoding import wrap, LocalFrame
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
from dataclasses import dataclass
from collections import deque
from scipy.linalg import null_space, qr
from uav_ac.planning.trajectory.gcopter.optimization import boundary_derivative_jacobian
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state

class InteriorMap:
    def __init__(self,head_pva,tail_pva,durations=None):
        self.head=finite_array(head_pva,(3,8),'head_pva')
        tail=np.array(tail_pva,dtype=float,copy=True)
        if tail.shape==(3,8): tail[0,3]=self.head[0,3]+wrap(tail[0,3]-self.head[0,3])
        self.tail=finite_array(tail,(3,8),'tail_pva')
        self.durations=finite_array(np.full(4,.75) if durations is None else durations,(4,),'durations')
        if np.any(self.durations<=1e-5): raise ValueError('durations must be positive and not singular')
        self.minco=MINCOQuintic(self.head,self.tail,4)
        self.free_indices=np.array([i for i in range(24) if i not in (0,1,16,17)])
        self.dependent=np.array([0,1,16,17])
        zero=np.zeros((3,8));base=self._jerks(zero)
        response=[]
        for index in range(24):
            points=zero.copy();points.ravel()[index]=1.
            response.append(self._jerks(points)-base)
        matrix=np.array(response).T
        block=matrix[:,self.dependent]
        condition=np.linalg.cond(block)
        if not np.isfinite(condition) or condition>1e10:
            raise ValueError('endpoint jerk elimination is ill conditioned')
        self.condition_number=float(condition)
        self.offset=np.zeros(24);self.offset[self.dependent]=np.linalg.solve(block,-base)
        self.jacobian=np.zeros((24,20));self.jacobian[self.free_indices,np.arange(20)]=1.
        self.jacobian[self.dependent]=np.linalg.solve(block,-matrix[:,self.free_indices])

    def _jerks(self,points):
        coeff,_=self.minco.solve(points,self.durations);c=coeff.reshape(4,6,8)
        t=self.durations[-1]
        return np.r_[6*c[0,3,:2],6*c[-1,3,:2]+24*t*c[-1,4,:2]+60*t*t*c[-1,5,:2]]

    def decode(self,free):
        free=finite_array(free,(20,),'free')
        return (self.offset+self.jacobian@free).reshape(3,8)

    def pullback(self,gradient):
        gradient=finite_array(gradient,(3,8),'points_gradient')
        return self.jacobian.T@gradient.ravel()

    def seed(self):
        fractions=(np.cumsum(self.durations)[:-1]/np.sum(self.durations))[:,None]
        return (self.head[0]+fractions*(self.tail[0]-self.head[0])).ravel()[self.free_indices]

# Deterministic map-free trajectory decoding; no optimizer or search calls.


def decode_trajectory(free,head_pva,goal,*,durations=None,gripper_gap):
    goal=finite_array(goal,(8,),'goal')
    if not np.isfinite(gripper_gap) or not .02<=gripper_gap<=.07:
        raise ValueError('gripper_gap outside model limits')
    tail=np.vstack((goal,np.zeros((2,8))))
    head=finite_array(head_pva,(3,8),'head_pva')
    frame=LocalFrame(head[0,:3],float(head[0,3]))
    local_head=frame.local_pva(head)
    local_tail=np.vstack((frame.local_goal(goal),np.zeros((2,8))))
    mapping=InteriorMap(local_head,local_tail,durations)
    points=frame.world_points8(mapping.decode(free))
    world_tail=local_tail.copy();world_tail[0,:3]=frame.to_world_points(local_tail[0,:3]);world_tail[0,3]+=frame.yaw
    world_minco=InteriorMap(head,world_tail,mapping.durations).minco
    flat,_=world_minco.solve(points,mapping.durations)
    return AerialManipulatorTrajectory(mapping.durations,flat.reshape(4,6,8),np.empty((0,8)),
                                      float('nan'),0,False,'network decode; not optimized or validated')

# Two-piece whole-body MINCO, following YOPO-MINCO's inner/tail-PVA/time layout.
# 
# Official reference: TJU-Aerial-Robotics/YOPO, d5e1e60, policy/state_transform.py
# and policy/poly_solver.py. The solver is the repository's own MINCOQuintic.
# Unlike the quadrotor reference, two terminal-velocity degrees are eliminated
# so that the starting FRD angular velocity also matches the observed boundary.


class RollingMap:
    """30 spatial free variables + two positive physical segment durations.

    Full physical layout: inner(8), tail PVA(24), durations(2).
    Two base terminal-velocity components are solved, not silently overwritten.
    All free values are in the shared start-yaw local frame, without scaling.
    Network activation/bounds belong to the future network decoder.
    """
    def __init__(self, head_pvaj, *, gravity=9.81):
        self.head = finite_array(head_pvaj, (4,8), 'head_pvaj')
        self.gravity = float(gravity)
        force = np.array([-self.head[2,0], -self.head[2,1], self.gravity-self.head[2,2]])
        if np.linalg.norm(force) < 1e-6:
            raise ValueError('singular starting thrust direction')
        self.basis = null_space(force[None]).T
        self.target = self.basis @ self.head[3,:3]
        matrix, _ = self._constraints(np.array([.75,.75]))
        _, _, pivots = qr(matrix[:,16:19], pivoting=True)
        self.dependent = np.array([16+p for p in pivots[:2]])
        self.free_indices = np.array([i for i in range(32) if i not in self.dependent])

    @classmethod
    def from_observation(cls, obs, *, gravity=9.81):
        expected, velocity = recover_state(np.vstack((obs.start_pva,obs.start_jerk)),
                                           obs.configuration[11], gravity)
        if (not np.isclose(abs(expected[3:7]@obs.configuration[3:7]),1.,atol=1e-7,rtol=0)
                or not np.allclose(velocity,obs.velocity,atol=1e-7,rtol=0)):
            raise ValueError('observed pose/rates inconsistent with declared flatness derivatives')
        frame = LocalFrame.from_observation(obs)
        head = np.vstack((frame.local_pva(obs.start_pva),obs.start_jerk.copy()))
        head[3,:3] = frame.to_local_vectors(obs.start_jerk[:3])
        return cls(head, gravity=gravity)

    def _start_jerk(self, physical, durations):
        minco = MINCOQuintic(self.head[:3], physical[8:].reshape(3,8), 2)
        coeff,_ = minco.solve(physical[:8][None], durations)
        return 6*coeff[3,:3]

    def _constraints(self, durations):
        zero = np.zeros(32)
        base = self.basis@self._start_jerk(zero,durations)
        columns=[]
        for i in range(32):
            value=zero.copy();value[i]=1.
            columns.append(self.basis@self._start_jerk(value,durations)-base)
        return np.asarray(columns).T,base

    def _map(self, durations):
        matrix,base = self._constraints(durations)
        block = matrix[:,self.dependent]
        if np.linalg.cond(block)>1e8:
            raise ValueError('angular boundary elimination is ill conditioned')
        offset=np.zeros(32)
        offset[self.dependent]=np.linalg.solve(block,self.target-base)
        jac=np.zeros((32,30));jac[self.free_indices,np.arange(30)]=1.
        jac[self.dependent]=np.linalg.solve(block,-matrix[:,self.free_indices])
        return offset,jac,matrix

    def decode(self, free):
        free=finite_array(free,(32,),'rolling_free')
        durations=free[-2:]
        if np.any(durations<.1) or np.any(durations>3.):
            raise ValueError('rolling durations must lie in [0.1,3] seconds')
        offset,jac,_=self._map(durations)
        physical=offset+jac@free[:30]
        return physical[:8][None],physical[8:].reshape(3,8),durations.copy()

    def seed(self, local_goal, *, horizon=1.5):
        goal=finite_array(local_goal,(8,),'local_goal')
        if not .2<=horizon<=6.:raise ValueError('horizon must lie in [0.2,6]')
        delta=goal-self.head[0];delta[3]=wrap(delta[3])
        length=np.linalg.norm(delta[:3])
        delta[:3]*=min(1.,1.5/max(length,1e-12))
        delta[4:]=np.clip(delta[4:],-.6,.6)
        delta[3]=np.clip(delta[3],-.5,.5)
        tail=np.zeros((3,8));tail[0]=self.head[0]+delta
        tail[1]=delta/horizon
        physical=np.r_[self.head[0]+.5*delta,tail.ravel()]
        return np.r_[physical[self.free_indices],horizon/2,horizon/2]

    def pullback(self, points_gradient, tail_gradient, time_gradient, free):
        points,tail,durations=self.decode(free)
        gp=finite_array(points_gradient,(1,8),'point_gradient')
        ge=finite_array(tail_gradient,(3,8),'tail_gradient')
        gt=finite_array(time_gradient,(2,),'time_gradient')
        raw=np.r_[gp.ravel(),ge.ravel()]
        _,jac,matrix=self._map(durations)
        multipliers=np.linalg.solve(matrix[:,self.dependent].T,raw[self.dependent])
        minco=MINCOQuintic(self.head[:3],tail,2)
        _,_,jerk_time,_,_=boundary_derivative_jacobian(minco,points,durations,3,(0,1,2))
        constraint_time=self.basis@jerk_time[:3]
        return np.r_[jac.T@raw,gt-multipliers@constraint_time]


def decode_rolling(free, obs, *, gravity=9.81):
    mapping=RollingMap.from_observation(obs,gravity=gravity)
    points,tail,durations=mapping.decode(free)
    frame=LocalFrame.from_observation(obs)
    world_tail=tail.copy()
    world_tail[0,:3]=frame.to_world_points(tail[0,:3])
    world_tail[1:,:3]=frame.to_world_vectors(tail[1:,:3])
    world_tail[0,3]+=frame.yaw
    minco=MINCOQuintic(obs.start_pva,world_tail,2)
    coeff,_=minco.solve(frame.world_points8(points),durations)
    return AerialManipulatorTrajectory(durations,coeff.reshape(2,6,8),np.empty((0,8)),
                                      float('nan'),0,False,'rolling_v2; unoptimized/unvalidated')


def observation_at(trajectory,time,goal,depth,mask,*,timestamp,depth_spec=None,gripper_gap=.06):
    if not 0<=time<=trajectory.total_time:
        raise ValueError('reference sample outside trajectory horizon')
    derivatives=np.stack([trajectory.evaluate(time,d) for d in range(4)])
    q,v=recover_state(derivatives,gripper_gap)
    return PlannerObservation(depth,mask,q,v,derivatives[:3],goal,
        DepthSpec() if depth_spec is None else depth_spec,timestamp,derivatives[3],'reference')


class ObservationHistory:
    """Only observed frames, chronologically ordered; no reconstructed geometry."""
    def __init__(self,capacity=4):
        if isinstance(capacity,bool) or not isinstance(capacity,int) or capacity<1:
            raise ValueError('history capacity must be a positive integer')
        self._frames=deque(maxlen=capacity)
    def append(self,observation):
        if self._frames and observation.timestamp<=self._frames[-1].timestamp:
            raise ValueError('observation timestamps must increase')
        self._frames.append(observation)
    @property
    def frames(self):return tuple(self._frames)


class ReferenceSchedule:
    """Nominal reference splice verifier; not a physics/controller execution loop."""
    def __init__(self,trajectory,start_time):
        if not np.isfinite(start_time) or start_time<0:raise ValueError('invalid reference clock')
        self.trajectory=trajectory;self.start_time=float(start_time);self.last_time=float(start_time)
    def _derivatives(self,time):
        from numbers import Real
        if not isinstance(time,Real) or isinstance(time,(bool,np.bool_)) or not np.isfinite(time):
            raise ValueError('reference clock must be a finite scalar')
        if time<self.last_time:raise ValueError('reference clock moved backwards')
        local=time-self.start_time
        if local>self.trajectory.total_time:raise TimeoutError('reference horizon exhausted')
        return np.stack([self.trajectory.evaluate(local,d) for d in range(4)])
    def derivatives_at(self,time):
        value=self._derivatives(time)
        self.last_time=float(time)
        return value
    def handoff(self,trajectory,time,*,gripper_gap=.06):
        old=self._derivatives(time)
        new=np.stack([trajectory.evaluate(0,d) for d in range(4)])
        delta=old[:3]-new[:3];delta[0,3]=wrap(delta[0,3])
        if np.max(abs(delta))>1e-7:raise ValueError('handoff is not C2 continuous')
        qo,vo=recover_state(old,gripper_gap);qn,vn=recover_state(new,gripper_gap)
        if not np.allclose(vo,vn,atol=1e-7,rtol=0):raise ValueError('handoff angular rates are discontinuous')
        self.trajectory=trajectory;self.start_time=float(time);self.last_time=float(time)
