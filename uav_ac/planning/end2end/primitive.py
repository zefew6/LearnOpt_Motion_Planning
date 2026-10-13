"""Calibrated camera cells and primitive-frame state for the primitive_v4 profile.

The oracle/decoder interface remains physical gravity-local MINCO34 parameters.
Only the neural representation and observation normalization are profile-specific.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .contracts import DepthSpec
from .encoding import encode_observation
from .anchors import make_anchors

PROFILE = 'primitive_v4'
PRIMITIVE_STATE_LAYOUT = ('local_pva_24','clipped_goal_direction_3','goal_yaw_sincos_2','goal_joints_4',
    'body_rotation_first_two_columns_6','velocity_local_3','omega_frd_3','start_joints_4',
    'arm_rates_4','gripper_gap_rate_2','gravity_body_3')


@dataclass(frozen=True)
class PrimitiveSpec:
    rows: int = 3
    columns: int = 3
    arm_modes: int = 3
    overlap: float = 1.1
    inner_min: float = .2
    inner_max: float = 2.5
    tail_min: float = .2
    tail_max: float = 3.
    goal_clip: float = 3.
    max_speed: float = 3.
    max_acceleration: float = 3.
    max_yaw_rate: float = 1.2
    max_yaw_acceleration: float = 2.5
    max_joint_rate: float = 2.
    max_joint_acceleration: float = 4.

    def __post_init__(self):
        if (self.rows,self.columns,self.arm_modes)!=(3,3,3) or any(isinstance(x,bool) for x in (self.rows,self.columns,self.arm_modes)):
            raise ValueError('primitive_v4 requires a 3x3 grid and three arm modes')
        values=[getattr(self,n) for n in self.__dataclass_fields__ if n not in ('rows','columns','arm_modes')]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values) or not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('primitive settings must be positive finite numbers')
        if not 1<=self.overlap<=1.3 or self.inner_min>=self.inner_max or self.tail_min>=self.tail_max:
            raise ValueError('invalid primitive overlap/reach')


def geometry(depth_spec, spec):
    """Cell-major order: top→bottom, left→right; R columns are axes in FRD."""
    h,v=np.deg2rad([depth_spec.hfov_deg,depth_spec.vfov_deg])
    angles=np.array([(yaw,pitch) for pitch in [-v/3,0,v/3] for yaw in [-h/3,0,h/3]])
    rays=np.c_[np.tan(angles[:,0]),np.tan(angles[:,1]),np.ones(9)]
    forward=rays@depth_spec.camera_extrinsics[:3,:3].T
    forward/=np.linalg.norm(forward,axis=1,keepdims=True)
    down=depth_spec.camera_extrinsics[:3,1]
    right=np.cross(down,forward);right/=np.linalg.norm(right,axis=1,keepdims=True)
    third=np.cross(forward,right)
    rotations=np.stack((forward,right,third),axis=-1)
    xy=np.tan(angles)/np.tan([h/2,v/2])
    return angles,rotations,xy


def local_body_rotation(state):
    columns=state[:,33:39].reshape(-1,2,3).transpose(1,2)
    return torch.cat((columns,torch.linalg.cross(columns[:,:,0],columns[:,:,1],dim=1)[:,:,None]),dim=2)


def primitive_state(state, rotations):
    """Rotate spatial vectors only; retain physical joint/yaw/attitude semantics."""
    body=local_body_rotation(state)
    # Local gravity-aligned vectors → body FRD → each primitive frame.
    local_to_primitive=rotations.transpose(-1,-2)[None]@body.transpose(-1,-2)[:,None]
    values=state[:,None].expand(-1,9,-1).clone()
    for start in (0,8,16,24,39):
        values[:,:,start:start+3]=torch.einsum('bnij,bj->bni',local_to_primitive,state[:,start:start+3])
    for start in (42,55):
        values[:,:,start:start+3]=torch.einsum('nij,bj->bni',rotations.transpose(-1,-2),state[:,start:start+3])
    return values


def encode_primitive(obs,normalization,spec=None):
    spec=PrimitiveSpec() if spec is None else spec
    encoded=encode_observation(obs,normalization)
    delta=encoded.frame.local_goal(obs.goal)[:3]
    encoded.state[0,24:27]=delta/max(spec.goal_clip,float(np.linalg.norm(delta)))
    encoded.state[0,39:42]/=spec.max_speed
    encoded.state[0,42:45]/=2.1
    encoded.state[0,49:53]/=spec.max_joint_rate
    return encoded


def primitive_anchors(obs,normalization,spec=None):
    spec=PrimitiveSpec() if spec is None else spec
    encoded=encode_primitive(obs,normalization,spec)
    state=torch.tensor(encoded.state);body=local_body_rotation(state)[0].numpy()
    _,rotations,_=geometry(obs.depth_spec,spec)
    anchors=make_anchors(obs,normalization,count=27).copy()
    offset=obs.depth_spec.camera_extrinsics[:3,3]
    anchors[:,:3]=np.repeat((offset+(spec.inner_min+spec.inner_max)/2*rotations[:,:,0])@body.T,3,axis=0)
    tail_body=offset+obs.depth_spec.camera_extrinsics[:3,2]*(spec.tail_min+spec.tail_max)/2
    anchors[:,8:11]=body@tail_body
    anchors[:,3]=0.;anchors[:,11]=0.
    anchors[:,16:32]=0.;anchors[:,-2:]=1.
    return anchors
