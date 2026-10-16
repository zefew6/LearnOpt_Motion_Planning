"""YOPO-style spatial CNN and shared candidate-conditioned prediction head."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from uav_ac.yopo.whole_body.model.primitives import parameter_scales, PROFILE, PrimitiveSpec, geometry, primitive_state, local_body_rotation
from uav_ac.yopo.whole_body.contracts import DepthSpec

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
    radius: torch.Tensor | None = None


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

# Shared camera-cell head, body-camera primitive alignment and MINCO34 decode.


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
        # Keep Conv2d initialization: free terminal outputs must not start identically zero.
        self._register_geometry(normalization)

    def _register_geometry(self,normalization):
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
        return NetworkOutput(self.decode(raw,state,anchors),score,raw)

# Versioned whole-body goal terminal: a body-origin ball includes hold/reverse.
# 
# The intermediate direction keeps its camera-cell constraint, with zero reach
# available for stopping. Terminal spatial vectors are encoded in the primitive
# frame and rotated to the body/gravity-local frame. Legacy v4
# angular terminal semantics remain unchanged in PrimitivePlannerNet.


class GoalPrimitivePlannerNet(PrimitivePlannerNet):
    profile='primitive_goal_v5'
    def __init__(self,normalization,*,score_mode='ranking',width=16,depth_spec=None,spec=None):
        spec=PrimitiveSpec(inner_min=0.,tail_min=0.) if spec is None else spec
        if spec.tail_min!=0. or spec.inner_min!=0.:raise ValueError('goal profile requires zero minimum reach')
        super().__init__(normalization,score_mode=score_mode,width=width,depth_spec=depth_spec,spec=spec)
        self._align_body_grid()
        # Ordinary nonzero weights; sensible forward reach/time biases are shared initial priors only.
        with torch.no_grad():
            self.parameter_head.bias[8]+=1/np.sqrt(3.)
            self.parameter_head.bias[32:34]+=np.log(1.4/1.5)-np.log(.9/2.)

    def _align_body_grid(self):
        # Project body-origin primitive centres into the physically mounted camera.
        spec=self.spec
        nominal=(spec.inner_min+spec.inner_max)/2*self.primitive_rotations[:,:,0]
        optical=(nominal-self.camera_offset)@self.camera_rotation
        self.sampling_xy.copy_(optical[:,:2]/optical[:,2:3]/self.fov_tangent)
    def decode(self,raw,state,anchors):
        z=super().decode(raw,state,anchors)
        free=raw[:,:,8:11]
        body_position=self.spec.tail_max*free/torch.sqrt(1+(free*free).sum(-1,keepdim=True))
        body=local_body_rotation(state)
        primitive=self.primitive_rotations.repeat_interleave(3,dim=0)
        tail=torch.einsum('bnij,bnj->bni',body[:,None]@primitive[None],body_position)
        inner=z[:,:,:3]-torch.einsum('bij,j->bi',body,self.camera_offset)[:,None]
        return torch.cat((inner,z[:,:,3:8],tail,z[:,:,11:]),dim=-1)


class OfficialWholeBodyPlannerNet(GoalPrimitivePlannerNet):
    """Official ResNet18 + shared YopoHead, extended to articulated MINCO34.

    Depth/mask replace the UAV's one-channel depth. Primitive-frame state has
    58 features, and a three-way arm embedding conditions each spatial cell.
    The extra 20 channels predict sampled whole-body environmental clearances;
    they do not certify a safe corridor or filter candidates at deployment.
    """
    profile='official_whole_body_v1'

    def __init__(self,normalization,*,score_mode='ranking',width=64,depth_spec=None,spec=None):
        from uav_ac.yopo.vendor.yopo_minco.policy.models.backbone import YopoBackbone
        from uav_ac.yopo.vendor.yopo_minco.policy.models.head import YopoHead
        nn.Module.__init__(self)
        if score_mode!='ranking' or width!=64:raise ValueError('official whole-body requires ranking and width 64')
        self.normalization=normalization;self.score_mode=score_mode;self.width=width
        self.depth_spec=DepthSpec() if depth_spec is None else depth_spec
        self.spec=PrimitiveSpec(inner_min=0.,tail_min=0.) if spec is None else spec
        if self.spec.inner_min!=0. or self.spec.tail_min!=0.:raise ValueError('official whole-body requires zero minimum reach')
        self._register_geometry(normalization);self._align_body_grid()
        self.backbone=YopoBackbone(64)
        self.backbone.cnn.conv1=nn.Conv2d(2,64,7,2,3,bias=False)
        self.mode_embedding=nn.Embedding(3,8)
        self.head=YopoHead(64+58+8,34+1+20)
        with torch.no_grad():
            self.head.model[-1].bias[8]+=1/np.sqrt(3.)
            self.head.model[-1].bias[32:34]+=np.log(1.4/1.5)-np.log(.9/2.)

    def forward(self,depth,state,anchors):
        if depth.shape[1:]!=(2,self.depth_spec.height,self.depth_spec.width) or state.shape!=(len(depth),58) or anchors.shape!=(len(depth),27,34):
            raise ValueError('official whole-body requires calibrated depth/state and 27 anchors')
        feature=self.backbone(depth);batch=len(depth)
        grid=camera_feature_grid(self.sampling_xy[None].expand(batch,-1,-1),depth.shape[-2:],feature.shape[-2:]).reshape(batch,3,3,2)
        local=nn.functional.grid_sample(feature,grid,align_corners=True,padding_mode='border')
        encoded=primitive_state(state,self.primitive_rotations).transpose(1,2).reshape(batch,58,3,3)
        predictions=[]
        for mode in range(3):
            condition=self.mode_embedding.weight[mode][None,:,None,None].expand(batch,-1,3,3)
            predictions.append(self.head(torch.cat((local,encoded,condition),dim=1)))
        pred=torch.stack(predictions,dim=-1).permute(0,2,3,4,1).reshape(batch,27,55)
        raw=pred[:,:,:34];score=nn.functional.softplus(pred[:,:,34])
        radius=torch.cat((torch.sigmoid(pred[:,:,35:45]),.02+.48*torch.sigmoid(pred[:,:,45:55])),dim=-1)
        return NetworkOutput(self.decode(raw,state,anchors),score,raw,radius)
