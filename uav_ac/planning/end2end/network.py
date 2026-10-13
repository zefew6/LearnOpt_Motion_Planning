"""YOPO-style spatial CNN and shared candidate-conditioned prediction head."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .anchors import parameter_scales
from .contracts import DepthSpec


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(),
                                nn.Conv2d(cout,cout,3,1,1,bias=False),nn.BatchNorm2d(cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(
            nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x):return torch.relu(self.main(x)+self.skip(x))


@dataclass
class NetworkOutput:
    parameters: torch.Tensor
    scores: torch.Tensor
    raw_parameters: torch.Tensor | None = None


def camera_feature_grid(xy,input_shape,feature_shape):
    """Optical normalized rays → stride32 CNN centers (input pixel index zero)."""
    height,width=input_shape;fh,fw=feature_shape
    pixels=(xy+1)*xy.new_tensor([width,height])/2-.5
    return 2*(pixels/32)/xy.new_tensor([fw-1,fh-1])-1


class WholeBodyPlannerNet(nn.Module):
    def __init__(self,normalization,*,score_mode='cost',width=16,depth_spec=None):
        super().__init__()
        if score_mode not in ('cost','ranking'):raise ValueError('unknown score mode')
        if width<4:raise ValueError('network width must be >=4')
        self.normalization=normalization;self.score_mode=score_mode;self.width=width
        self.depth_spec=DepthSpec() if depth_spec is None else depth_spec
        self.backbone=nn.Sequential(nn.Conv2d(2,width,7,2,3,bias=False),nn.BatchNorm2d(width),nn.ReLU(),
            nn.MaxPool2d(3,2,1),ResidualBlock(width,width),ResidualBlock(width,width),
            ResidualBlock(width,2*width,2),ResidualBlock(2*width,2*width),
            ResidualBlock(2*width,4*width,2),ResidualBlock(4*width,4*width),
            ResidualBlock(4*width,8*width,2),ResidualBlock(8*width,8*width),nn.Conv2d(8*width,64,1))
        self.state_encoder=nn.Sequential(nn.Linear(58,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.anchor_encoder=nn.Sequential(nn.Linear(34,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64*4+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.parameter_head=nn.Linear(128,34);self.score_head=nn.Linear(128,1)
        nn.init.zeros_(self.parameter_head.weight);nn.init.zeros_(self.parameter_head.bias)
        self.register_buffer('scales',torch.tensor(parameter_scales(normalization),dtype=torch.float32))
        self.register_buffer('lower',torch.tensor(normalization.joint_lower,dtype=torch.float32))
        self.register_buffer('upper',torch.tensor(normalization.joint_upper,dtype=torch.float32))
        self.register_buffer('camera_rotation',torch.tensor(self.depth_spec.camera_extrinsics[:3,:3],dtype=torch.float32))
        self.register_buffer('camera_offset',torch.tensor(self.depth_spec.camera_extrinsics[:3,3],dtype=torch.float32))
        self.register_buffer('fov_tangent',torch.tensor(np.tan(np.deg2rad([self.depth_spec.hfov_deg,self.depth_spec.vfov_deg])/2),dtype=torch.float32))

    def forward(self,depth,state,anchors):
        if depth.ndim!=4 or depth.shape[1]!=2 or state.shape!=(len(depth),58) or anchors.ndim!=3 or anchors.shape[0]!=len(depth) or anchors.shape[2]!=34:
            raise ValueError('invalid network input shapes')
        feature=self.backbone(depth);batch,k,_=anchors.shape
        global_feature=feature.mean(dim=(-2,-1))[:,None].expand(-1,k,-1)
        # Sample spatial features at each anchor direction. Coordinate metadata
        # remains in the state/descriptor; no oracle information enters here.
        columns=state[:,33:39].reshape(batch,2,3).transpose(1,2)
        rotation=torch.cat((columns,torch.linalg.cross(columns[:,:,0],columns[:,:,1],dim=1)[:,:,None]),dim=2)
        camera_rotation=rotation@self.camera_rotation
        camera_offset=rotation@self.camera_offset
        relative=anchors[:,:,:3]-camera_offset[:,None]
        optical=torch.einsum('bki,bij->bkj',relative,camera_rotation)
        z=optical[:,:,2]
        xy=optical[:,:,:2]/z.clamp(min=1e-4)[:,:,None]/self.fov_tangent
        visible=(z>0)&(xy.abs().amax(dim=-1)<=1)
        grid=camera_feature_grid(xy,depth.shape[-2:],feature.shape[-2:])[:,:,None,:]
        local=nn.functional.grid_sample(feature,grid,align_corners=True,padding_mode='border').squeeze(-1).transpose(1,2)
        local=local*visible[:,:,None]
        state_feature=self.state_encoder(state)[:,None].expand(-1,k,-1)
        anchor_feature=self.anchor_encoder(anchors/self.scales)
        hidden=self.head(torch.cat((global_feature,local,state_feature,anchor_feature,visible[:,:,None].to(depth.dtype)),dim=-1))
        raw=self.parameter_head(hidden)
        parameters=anchors+.5*self.scales*torch.tanh(raw)
        # Positive bounded times with exact anchor-centered initialization.
        time_center=((anchors[:,:,-2:]-.1)/2.9).clamp(1e-5,1-1e-5)
        times=.1+2.9*torch.sigmoid(torch.logit(time_center)+2*torch.tanh(raw[:,:,-2:]))
        parameters=torch.cat((parameters[:,:,:32],times),dim=-1)
        for start in (4,12):
            center=(self.lower+self.upper)/2;half=(self.upper-self.lower)/2
            baseline=((anchors[:,:,start:start+4]-center)/half).clamp(-.99999,.99999)
            joints=center+half*torch.tanh(torch.atanh(baseline)+.35*torch.tanh(raw[:,:,start:start+4]))
            parameters=torch.cat((parameters[:,:,:start],joints,parameters[:,:,start+4:]),dim=-1)
        score=self.score_head(hidden).squeeze(-1)
        if self.score_mode=='cost':score=nn.functional.softplus(score)
        return NetworkOutput(parameters,score)
