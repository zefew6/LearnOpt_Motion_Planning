"""Analytic guidance backward, with detached cost regression targets."""
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import replace

class _Guidance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,free,obs,oracle,scale):
        result=oracle.evaluate(obs,free.detach().cpu().numpy())
        gradient=torch.as_tensor(result.free_gradients,dtype=free.dtype,device=free.device)/scale
        ctx.save_for_backward(gradient)
        return torch.as_tensor(result.costs,dtype=free.dtype,device=free.device)/scale
    @staticmethod
    def backward(ctx,grad_output):
        gradient,=ctx.saved_tensors
        return gradient*grad_output[:,None],None,None,None


def guidance_loss(free,obs,oracle,*,scale=1.):
    if not np.isfinite(scale) or scale<=0: raise ValueError('cost scale must be positive')
    return _Guidance.apply(free,obs,oracle,float(scale)).mean()


def score_loss(predicted,costs):
    return F.smooth_l1_loss(predicted,costs.detach(),beta=1.)


def calibrate_cost_scale(costs):
    values=np.asarray(costs,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('calibration costs must be finite nonnegative candidates')
    return max(1.,float(np.median(values[:256])))


def guidance_costs(parameters,obs,oracle,*,scale=1.):
    if not np.isfinite(scale) or scale<=0:raise ValueError('cost scale must be positive')
    return _Guidance.apply(parameters,obs,oracle,float(scale))


class _PrecomputedGuidance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,parameters,result,scale):
        gradient=torch.as_tensor(result.free_gradients,dtype=parameters.dtype,device=parameters.device)/scale
        ctx.save_for_backward(gradient)
        return torch.as_tensor(result.costs,dtype=parameters.dtype,device=parameters.device)/scale
    @staticmethod
    def backward(ctx,grad_output):
        gradient,=ctx.saved_tensors
        return gradient*grad_output[:,None],None,None


def guidance_costs_from_result(parameters,result,*,scale=1.):
    if not np.isfinite(scale) or scale<=0:raise ValueError('cost scale must be positive')
    return _PrecomputedGuidance.apply(parameters,result,float(scale))


def ranking_loss(scores,costs,unsafe):
    if scores.shape!=costs.shape or unsafe.shape!=costs.shape:raise ValueError('ranking shape mismatch')
    detached=costs.detach()
    offset=detached.amax(dim=1,keepdim=True)-detached.amin(dim=1,keepdim=True)+1.
    target=(detached+offset*unsafe).argmin(dim=1)
    return F.cross_entropy(scores,target),target


def augment_observation(obs,rng,normalization,*,goal_std=0.,velocity_std=0.):
    if goal_std<0 or velocity_std<0:raise ValueError('augmentation std must be nonnegative')
    goal=obs.goal.copy();goal[:3]+=rng.normal(0,goal_std,3)
    goal[4:]=np.clip(goal[4:]+rng.normal(0,goal_std,4),normalization.joint_lower,normalization.joint_upper)
    velocity=obs.velocity.copy();pva=obs.start_pva.copy()
    if velocity_std:
        velocity[:10]+=rng.normal(0,velocity_std,10)
        pva[1,:3]=velocity[:3];pva[1,4:]=velocity[6:10]
        pva[2,:3]+=rng.normal(0,velocity_std,3)
    return replace(obs,goal=goal,velocity=velocity,start_pva=pva,derivative_source='augmentation')
