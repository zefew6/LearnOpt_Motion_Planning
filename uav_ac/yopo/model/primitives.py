"""Analytic spatial/arm candidate priors; no environment-map dependency."""
import numpy as np
from uav_ac.yopo.model.encoding import LocalFrame, wrap, encode_observation
from dataclasses import dataclass
import torch
from uav_ac.yopo.contracts import DepthSpec

def parameter_scales(normalization):
    return np.r_[normalization.scales,np.tile(normalization.scales,3),1.,1.]


def make_anchors(obs,normalization,*,count=27):
    if count not in (1,27):raise ValueError('candidate count must be 1 or 27')
    obs.validate_joints(normalization.joint_lower,normalization.joint_upper)
    frame=LocalFrame.from_observation(obs);head=frame.local_pva(obs.start_pva)
    goal=frame.local_goal(obs.goal);delta=goal-head[0];delta[3]=wrap(delta[3])
    delta[:3]*=min(1.,1.5/max(np.linalg.norm(delta[:3]),1e-12))
    delta[3]=np.clip(delta[3],-.5,.5);delta[4:]=np.clip(delta[4:],-.6,.6)
    tail=np.zeros((3,8));tail[0]=head[0]+delta;tail[1]=delta/1.5
    if count==1:return np.r_[head[0]+.5*delta,tail.ravel(),.75,.75][None]
    direct=(head[0]+tail[0])*.5
    # These are shape priors, not promises of self/environment feasibility.
    folded=np.array([direct[4],-.8,1.2,-.4]);extended=np.array([direct[4],.3,-.45,.15])
    arm_modes=[direct[4:],.5*direct[4:]+.5*folded,.5*direct[4:]+.5*extended]
    radius=np.clip(.5*np.linalg.norm(delta[:3]),.35,.9)
    anchors=[]
    for pitch in np.deg2rad([-20.,0.,20.]):
        for yaw in np.deg2rad([-30.,0.,30.]):
            for arm in arm_modes:
                inner=direct.copy()
                inner[:3]=radius*np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
                inner[4:]=np.clip(arm,normalization.joint_lower,normalization.joint_upper)
                anchors.append(np.r_[inner,tail.ravel(),.75,.75])
    return np.asarray(anchors)

# Calibrated camera cells and primitive-frame state for the primitive_v4 profile.
# 
# The oracle/decoder interface remains physical gravity-local MINCO34 parameters.
# Only the neural representation and observation normalization are profile-specific.

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
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values) or not np.isfinite(values).all() or any(getattr(self,n)<=0 for n in self.__dataclass_fields__ if n not in ('rows','columns','arm_modes','inner_min','tail_min')) or self.tail_min<0 or self.inner_min<0:
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
