"""Map-free UAV inference adapter around the pinned official implementation."""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from uav_ac.yopo.contracts import DepthSpec
from uav_ac.yopo.vendor.yopo_minco.policy.poly_solver import MincoTraj
from .checkpoint import load_checkpoint

NED_TO_NWU=np.diag([1.,-1.,-1.])


def preprocess_depth(depth):
    depth=np.asarray(depth,dtype=np.float32)
    if depth.shape not in ((90,160),(96,160)) or not np.isfinite(depth).all() or np.any(depth<.1):
        raise ValueError('UAV depth requires finite optical-z90/96x160 depths >=0.1m')
    if depth.shape[0]!=96:depth=depth[np.arange(96)*depth.shape[0]//96]
    # Upstream ROS inference quantizes to uint8 before inpainting. Our ideal
    # observer guarantees finite >=0.1m returns, so its inpaint mask is empty.
    return (np.uint8(np.minimum(depth,20.)/20.*255).astype(np.float32)/255.)[None,None]


def body_observation(state,acceleration,goal,goal_length):
    state=np.asarray(state,dtype=float);acceleration=np.asarray(acceleration,dtype=float);goal=np.asarray(goal,dtype=float)
    if state.shape!=(13,) or acceleration.shape!=(3,) or goal.shape!=(3,) or not np.isfinite(np.r_[state,acceleration,goal]).all():
        raise ValueError('invalid UAV current state/acceleration/goal')
    rotation_ned=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
    rotation=NED_TO_NWU@rotation_ned@NED_TO_NWU
    delta=NED_TO_NWU@(goal-state[:3])
    # Official ROS process_odom preserves world height error when clipping a far goal.
    if np.linalg.norm(delta)>goal_length:
        gz=np.clip(delta[2],-goal_length,goal_length)
        hxy=delta[:2]/(np.linalg.norm(delta[:2])+1e-9)*np.sqrt(goal_length**2-gz**2)
        delta=np.r_[hxy,gz]
    obs=np.r_[rotation.T@(NED_TO_NWU@state[7:10]),rotation.T@(NED_TO_NWU@acceleration),rotation.T@delta]
    return obs.astype(np.float32),rotation


def select_candidate(scores,inner_world,radius,*,topk=1,previous_inner=None):
    scores=np.asarray(scores);radius=np.asarray(radius);nr=radius.shape[1]//2
    mu_lo=(radius[:,:nr]-radius[:,nr:]).min(axis=1)
    safe_mu=1.-np.exp(-.05/2.)  # pinned upstream runtime safety admission
    candidates=np.flatnonzero(mu_lo>=safe_mu)
    if not len(candidates):return int(np.argmax(scores)),True
    if topk<=1 or previous_inner is None:return int(candidates[np.argmax(scores[candidates])]),False
    k=min(topk,len(candidates));ids=candidates[np.argpartition(scores[candidates],-k)[-k:]]
    distances=np.linalg.norm(inner_world[ids]-previous_inner,axis=1)
    return int(ids[np.argmin(distances)]),False


@dataclass
class UavPlan:
    trajectory: MincoTraj
    trajectories: MincoTraj
    selected_index: int
    scores: np.ndarray
    corridor: np.ndarray
    inner_ned: np.ndarray
    brake: bool


class UavPlanner:
    def __init__(self,checkpoint,*,device='cpu',velocity=6.,topk=1):
        if isinstance(topk,bool) or not isinstance(topk,int) or not 1<=topk<=15:raise ValueError('invalid UAV topk')
        self.core,self.metadata=load_checkpoint(checkpoint,device=device,velocity=velocity)
        self.depth_spec=DepthSpec(height=90,vfov_deg=float(np.rad2deg(2*np.arctan(45/80))),far_m=20.);self.topk=topk;self.previous_inner=None;self.generation=0
    @torch.inference_mode()
    def warm_up(self):
        # Pinned official ROS initialization pays CUDA/BLAS startup before flight.
        self.core.inference(torch.zeros(1,1,96,160,device=self.core.device),torch.zeros(1,9,device=self.core.device))
        duration=self.core.network.state_transform.lattice_primitive.segment_time/2.
        MincoTraj().solve(np.zeros((3,3)),np.array([[1.,0,0],[0,0,0],[0,0,0]]),np.array([.5,0,0]),[duration,duration])
        if self.core.device.type=='cuda':torch.cuda.synchronize(self.core.device)
    def reset(self):
        self.generation+=1;self.previous_inner=None
    @torch.inference_mode()
    def plan(self,depth,state,acceleration,goal):
        generation=self.generation;previous=self.previous_inner
        obs,rotation=body_observation(state,acceleration,goal,self.core.config['goal_length'])
        outputs=self.core.inference(torch.from_numpy(preprocess_depth(depth)),torch.from_numpy(obs[None]))
        inner,tail,times,score,_,radius=[v.detach().cpu().numpy()[0] for v in outputs]
        score=score.reshape(-1)
        position=NED_TO_NWU@state[:3]
        inner_world=inner@rotation.T+position
        tail_world=np.einsum('ij,ndj->ndi',rotation,tail);tail_world[:,0]+=position
        head=np.stack((position,NED_TO_NWU@state[7:10],NED_TO_NWU@acceleration))
        index,brake=select_candidate(score,inner_world,radius,topk=self.topk,previous_inner=previous)
        if not all(np.isfinite(x).all() for x in (inner_world,tail_world,times,score,radius)):raise FloatingPointError('nonfinite UAV inference')
        all_trajectories=MincoTraj().solve(np.broadcast_to(head,(15,3,3)),tail_world,inner_world,times)
        chosen=MincoTraj().solve(head,tail_world[index],inner_world[index],times[index])
        if generation==self.generation:self.previous_inner=inner_world[index].copy()
        return UavPlan(chosen,all_trajectories,index,score,radius,inner_world@NED_TO_NWU,brake)
