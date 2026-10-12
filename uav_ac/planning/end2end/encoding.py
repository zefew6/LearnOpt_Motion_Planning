"""A shared gravity-aligned local frame, without environment information."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import finite_array


def wrap(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))


@dataclass(frozen=True)
class Normalization:
    joint_lower: tuple
    joint_upper: tuple
    position_scale: float = .5
    yaw_scale: float = .7
    joint_scales: tuple = (.8,.8,.8,.8)

    def __post_init__(self):
        lower=finite_array(self.joint_lower,(4,),'joint_lower')
        upper=finite_array(self.joint_upper,(4,),'joint_upper')
        if np.any(upper<=lower): raise ValueError('joint limits must be ordered')
        scales=np.r_[np.full(3,self.position_scale),self.yaw_scale,self.joint_scales]
        if scales.shape!=(8,) or not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError('normalization scales must be positive')
        object.__setattr__(self,'joint_lower',lower);object.__setattr__(self,'joint_upper',upper)

    @property
    def scales(self): return np.r_[np.full(3,self.position_scale),self.yaw_scale,self.joint_scales]

    def joints(self,values):
        return (2*np.asarray(values)-self.joint_upper-self.joint_lower)/(self.joint_upper-self.joint_lower)


@dataclass(frozen=True)
class LocalFrame:
    origin: np.ndarray
    yaw: float

    def __post_init__(self):
        object.__setattr__(self,'origin',finite_array(self.origin,(3,),'origin'))
        if not np.isfinite(self.yaw): raise ValueError('yaw must be finite')

    @property
    def rotation(self): return Rotation.from_euler('z',-self.yaw).as_matrix()
    @classmethod
    def from_observation(cls,obs): return cls(obs.start_pva[0,:3],float(obs.start_pva[0,3]))
    def to_local_points(self,points): return (np.asarray(points)-self.origin)@self.rotation.T
    def to_world_points(self,points): return np.asarray(points)@self.rotation+self.origin
    def to_local_vectors(self,values): return np.asarray(values)@self.rotation.T
    def to_world_vectors(self,values): return np.asarray(values)@self.rotation

    def local_goal(self,goal):
        out=np.array(goal,copy=True,dtype=float);out[:3]=self.to_local_points(out[:3]);out[3]=wrap(out[3]-self.yaw)
        return out

    def local_pva(self,pva):
        out=np.array(pva,copy=True,dtype=float)
        out[0,:3]=self.to_local_points(out[0,:3]);out[1:,:3]=self.to_local_vectors(out[1:,:3]);out[0,3]=wrap(out[0,3]-self.yaw)
        return out

    def world_points8(self,points):
        out=np.array(points,copy=True);out[:,:3]=self.to_world_points(out[:,:3]);out[:,3]+=self.yaw
        return out

    def pullback_points8(self,grad):
        out=np.array(grad,copy=True);out[:,:3]=self.to_local_vectors(out[:,:3]);return out

    def camera_to_local(self,configuration,spec):
        q=np.asarray(configuration)
        body_to_world=Rotation.from_quat(q[[4,5,6,3]]).as_matrix()
        pose=np.eye(4);pose[:3,:3]=self.rotation@body_to_world
        pose[:3,3]=self.to_local_points(q[:3])
        return pose@spec.camera_extrinsics


STATE_LAYOUT = ('local_pva_24','goal_position_3','goal_yaw_sincos_2','goal_joints_4',
                'body_rotation_first_two_columns_6','velocity_local_3','omega_frd_3',
                'start_joints_4','arm_rates_4','gripper_gap_rate_2','gravity_body_3')

@dataclass(frozen=True)
class EncodedObservation:
    depth: np.ndarray
    state: np.ndarray
    frame: LocalFrame
    camera_to_local: np.ndarray


def encode_observation(obs,normalization):
    obs.validate_joints(normalization.joint_lower,normalization.joint_upper)
    frame=LocalFrame.from_observation(obs);goal=frame.local_goal(obs.goal)
    pva=frame.local_pva(obs.start_pva)/normalization.scales
    q=obs.configuration;rot=Rotation.from_quat(q[[4,5,6,3]]).as_matrix()
    local_rot=frame.rotation@rot
    state=np.r_[pva.ravel(),goal[:3]/normalization.position_scale,np.sin(goal[3]),np.cos(goal[3]),
                normalization.joints(goal[4:]),local_rot[:,:2].T.ravel(),
                frame.to_local_vectors(obs.velocity[:3]),obs.velocity[3:6],normalization.joints(q[7:11]),
                obs.velocity[6:10],q[11],obs.velocity[10],rot.T@np.array([0.,0.,1.])]
    depth=np.stack((obs.depth_m/obs.depth_spec.far_m,obs.depth_valid.astype(float)))[None]
    return EncodedObservation(depth.astype(np.float32),state[None].astype(np.float32),frame,
                              frame.camera_to_local(q,obs.depth_spec))


def unproject_depth(depth,mask,spec):
    rows,cols=np.nonzero(mask);z=np.asarray(depth)[rows,cols];k=spec.intrinsics
    return np.column_stack(((cols+.5-k[0,2])*z/k[0,0],(rows+.5-k[1,2])*z/k[1,1],z))
