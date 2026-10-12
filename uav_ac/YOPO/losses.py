"""Analytic guidance backward, with detached cost regression targets."""
import numpy as np
import torch
from torch.nn import functional as F


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
