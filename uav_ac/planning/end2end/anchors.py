"""Analytic spatial/arm candidate priors; no environment-map dependency."""
import numpy as np
from .encoding import LocalFrame,wrap


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
