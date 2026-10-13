"""Map-free learned candidate selection and deterministic MINCO decode."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .checkpoint import load_checkpoint
from .encoding import encode_observation
from .anchors import make_anchors
from .direct import decode_direct

# The fixed opening is a servo target; retain the measured state within 0.5 mm.
GRIPPER_SERVO_TOLERANCE = .0005


@dataclass
class LearnedPlan:
    trajectory: object
    candidate_count: int
    selected_index: int
    scores: np.ndarray
    parameters: np.ndarray
    seconds: float
    observation: object = None


class End2EndPlanner:
    def __init__(self,checkpoint,*,device='cpu',expected_robot_hash=None,gripper_tolerance=GRIPPER_SERVO_TOLERANCE,speed_scale=1.):
        if isinstance(gripper_tolerance,bool) or not np.isfinite(gripper_tolerance) or not 0<=gripper_tolerance<=.005:
            raise ValueError('gripper tolerance must lie in [0,0.005] m')
        if isinstance(speed_scale,bool) or not np.isfinite(speed_scale) or not .5<=speed_scale<=2.:
            raise ValueError('speed_scale must lie in [0.5,2]')
        self.speed_scale=float(speed_scale)
        self.gripper_tolerance=float(gripper_tolerance)
        self.model,self.metadata=load_checkpoint(checkpoint,device=device,expected_robot_hash=expected_robot_hash)
        self.device=device;self.count=int(self.metadata['candidate_count'])
        if self.count not in (1,27):raise ValueError('unsupported candidate count')

    @torch.inference_mode()
    def plan(self,observation):
        started=time.perf_counter()
        if observation.depth_spec.as_mapping()!=self.metadata['depth_spec']:
            raise ValueError('observation depth calibration mismatch')
        if 'gripper_gap' not in self.metadata or not np.isclose(observation.configuration[11],self.metadata['gripper_gap'],rtol=0.,atol=self.gripper_tolerance):
            raise ValueError('observation gripper opening mismatch')
        n=self.model.normalization;e=encode_observation(observation,n)
        anchors=make_anchors(observation,n,count=self.count)
        output=self.model(torch.as_tensor(e.depth,device=self.device),torch.as_tensor(e.state,device=self.device),
                          torch.as_tensor(anchors[None],dtype=torch.float32,device=self.device))
        parameters=output.parameters[0].cpu().numpy();scores=output.scores[0].cpu().numpy()
        if not np.isfinite(parameters).all() or not np.isfinite(scores).all():raise FloatingPointError('nonfinite network output')
        if self.speed_scale!=1.:
            parameters=parameters.copy()
            parameters[:,16:24]*=self.speed_scale
            parameters[:,24:32]*=self.speed_scale**2
            parameters[:,-2:]=np.clip(parameters[:,-2:]/self.speed_scale,.1,3.)
        index=int(np.argmin(scores) if self.model.score_mode=='cost' else np.argmax(scores))
        return LearnedPlan(decode_direct(parameters[index],observation),self.count,index,scores,parameters,time.perf_counter()-started,observation)
