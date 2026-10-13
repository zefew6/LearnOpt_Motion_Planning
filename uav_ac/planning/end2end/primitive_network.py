"""Shared camera-cell head, body-camera primitive alignment and MINCO34 decode."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .network import ResidualBlock,NetworkOutput,camera_feature_grid
from .contracts import DepthSpec
from .primitive import PROFILE,PrimitiveSpec,geometry,primitive_state,local_body_rotation


class PrimitivePlannerNet(nn.Module):
    profile=PROFILE

    def __init__(self,normalization,*,score_mode='ranking',width=16,depth_spec=None,spec=None):
        super().__init__()
        if score_mode not in ('cost','ranking') or width<4:raise ValueError('invalid primitive network settings')
        self.normalization=normalization;self.score_mode=score_mode;self.width=width
        self.depth_spec=DepthSpec() if depth_spec is None else depth_spec
        self.spec=PrimitiveSpec() if spec is None else spec
        self.backbone=nn.Sequential(nn.Conv2d(2,width,7,2,3,bias=False),nn.BatchNorm2d(width),nn.ReLU(),
            nn.MaxPool2d(3,2,1),ResidualBlock(width,width),ResidualBlock(width,width),
            ResidualBlock(width,2*width,2),ResidualBlock(2*width,2*width),
            ResidualBlock(2*width,4*width,2),ResidualBlock(4*width,4*width),
            ResidualBlock(4*width,8*width,2),ResidualBlock(8*width,8*width),nn.Conv2d(8*width,64,1))
        self.state_encoder=nn.Sequential(nn.Linear(58,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.mode_embedding=nn.Embedding(3,8)
        self.head=nn.Sequential(nn.Conv2d(200,128,1),nn.ReLU(),nn.Conv2d(128,128,1),nn.ReLU())
        self.parameter_head=nn.Conv2d(128,34,1);self.score_head=nn.Conv2d(128,1,1)
        nn.init.zeros_(self.parameter_head.weight);nn.init.zeros_(self.parameter_head.bias)
        angles,rotations,xy=geometry(self.depth_spec,self.spec)
        for name,value in {'lower':normalization.joint_lower,'upper':normalization.joint_upper,
            'camera_rotation':self.depth_spec.camera_extrinsics[:3,:3],
            'camera_offset':self.depth_spec.camera_extrinsics[:3,3],
            'fov_tangent':np.tan(np.deg2rad([self.depth_spec.hfov_deg,self.depth_spec.vfov_deg])/2),
            'spec_values':list(asdict(self.spec).values()),'bin_angles':angles,'primitive_rotations':rotations,'sampling_xy':xy}.items():
            self.register_buffer(name,torch.tensor(np.array(value,copy=True),dtype=torch.float32))

    def decode(self,raw,state,anchors):
        spec=self.spec;body=local_body_rotation(state)
        tan=torch.tanh(raw);centers=self.bin_angles.repeat_interleave(3,dim=0)
        half=torch.atan(self.fov_tangent)
        angles=centers+tan[:,:,:2]*half/3*spec.overlap
        angles=torch.maximum(torch.minimum(angles,half),-half)
        radius=spec.inner_min+(spec.inner_max-spec.inner_min)*torch.sigmoid(raw[:,:,2])
        def points(angles,radius):
            optical=torch.cat((torch.tan(angles),torch.ones_like(radius)[:,:,None]),dim=-1)
            optical=optical/torch.linalg.vector_norm(optical,dim=-1,keepdim=True)
            pos=optical@self.camera_rotation.T*radius[:,:,None]+self.camera_offset
            return torch.einsum('bij,bnj->bni',body,pos)
        inner=points(angles,radius)
        tail_angles=tan[:,:,8:10]*half
        tail_radius=spec.tail_min+(spec.tail_max-spec.tail_min)*torch.sigmoid(raw[:,:,10])
        tail=points(tail_angles,tail_radius)
        def joints(start):
            center=(self.lower+self.upper)/2;span=(self.upper-self.lower)/2
            baseline=((anchors[:,:,start:start+4]-center)/span).clamp(-.99999,.99999)
            return center+span*torch.tanh(torch.atanh(baseline)+raw[:,:,start:start+4])
        rotations=self.primitive_rotations.repeat_interleave(3,dim=0)
        primitive_to_local=body[:,None]@rotations[None]
        velocity=torch.einsum('bnij,bnj->bni',primitive_to_local,tan[:,:,16:19]*spec.max_speed)
        acceleration=torch.einsum('bnij,bnj->bni',primitive_to_local,tan[:,:,24:27]*spec.max_acceleration)
        times=.1+2.9*torch.sigmoid(raw[:,:,32:34]+np.log(.9/2.))
        return torch.cat((inner,.7*tan[:,:,3:4],joints(4),tail,.7*tan[:,:,11:12],joints(12),
            velocity,spec.max_yaw_rate*tan[:,:,19:20],spec.max_joint_rate*tan[:,:,20:24],
            acceleration,spec.max_yaw_acceleration*tan[:,:,27:28],spec.max_joint_acceleration*tan[:,:,28:32],times),dim=-1)

    def forward(self,depth,state,anchors):
        if depth.ndim!=4 or depth.shape[1]!=2 or state.shape!=(len(depth),58) or anchors.shape!=(len(depth),27,34):
            raise ValueError('primitive_v4 requires depth/state and 27 anchors')
        feature=self.backbone(depth);batch=len(depth)
        xy=self.sampling_xy[None].expand(batch,-1,-1)
        grid=camera_feature_grid(xy,depth.shape[-2:],feature.shape[-2:]).reshape(batch,3,3,2)
        local=nn.functional.grid_sample(feature,grid,align_corners=True,padding_mode='border')
        state_feature=self.state_encoder(primitive_state(state,self.primitive_rotations)).transpose(1,2).reshape(batch,64,3,3)
        global_feature=feature.mean((-2,-1),keepdim=True).expand(-1,-1,3,3)
        outputs=[];scores=[]
        for mode in range(3):
            condition=self.mode_embedding.weight[mode][None,:,None,None].expand(batch,-1,3,3)
            hidden=self.head(torch.cat((local,state_feature,global_feature,condition),dim=1))
            outputs.append(self.parameter_head(hidden));scores.append(self.score_head(hidden))
        raw=torch.stack(outputs,dim=-1).permute(0,2,3,4,1).reshape(batch,27,34)
        score=torch.stack(scores,dim=-1).reshape(batch,27)
        if self.score_mode=='cost':score=nn.functional.softplus(score)
        return NetworkOutput(self.decode(raw,state,anchors),score)
