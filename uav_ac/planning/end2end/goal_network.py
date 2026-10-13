"""Versioned whole-body goal terminal: a body-origin ball includes hold/reverse.

The intermediate direction keeps its camera-cell constraint, with zero reach
available for stopping. Terminal spatial vectors are encoded in the primitive
frame and rotated to the body/gravity-local frame. Legacy v4
angular terminal semantics remain unchanged in PrimitivePlannerNet.
"""
import numpy as np
import torch
from .primitive import PrimitiveSpec,local_body_rotation
from .primitive_network import PrimitivePlannerNet


class GoalPrimitivePlannerNet(PrimitivePlannerNet):
    profile='primitive_goal_v5'
    def __init__(self,normalization,*,score_mode='ranking',width=16,depth_spec=None,spec=None):
        spec=PrimitiveSpec(inner_min=0.,tail_min=0.) if spec is None else spec
        if spec.tail_min!=0. or spec.inner_min!=0.:raise ValueError('goal profile requires zero minimum reach')
        super().__init__(normalization,score_mode=score_mode,width=width,depth_spec=depth_spec,spec=spec)
        # Project body-origin primitive centres into the physically mounted camera.
        nominal=(spec.inner_min+spec.inner_max)/2*self.primitive_rotations[:,:,0]
        optical=(nominal-self.camera_offset)@self.camera_rotation
        self.sampling_xy.copy_(optical[:,:2]/optical[:,2:3]/self.fov_tangent)
        # Ordinary nonzero weights; sensible forward reach/time biases are shared initial priors only.
        with torch.no_grad():
            self.parameter_head.bias[8]+=1/np.sqrt(3.)
            self.parameter_head.bias[32:34]+=np.log(1.4/1.5)-np.log(.9/2.)
    def decode(self,raw,state,anchors):
        z=super().decode(raw,state,anchors)
        free=raw[:,:,8:11]
        body_position=self.spec.tail_max*free/torch.sqrt(1+(free*free).sum(-1,keepdim=True))
        body=local_body_rotation(state)
        primitive=self.primitive_rotations.repeat_interleave(3,dim=0)
        tail=torch.einsum('bnij,bnj->bni',body[:,None]@primitive[None],body_position)
        inner=z[:,:,:3]-torch.einsum('bij,j->bi',body,self.camera_offset)[:,None]
        return torch.cat((inner,z[:,:,3:8],tail,z[:,:,11:]),dim=-1)
