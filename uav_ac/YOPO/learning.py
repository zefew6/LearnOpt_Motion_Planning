"""Direct-PVA guidance and independent offline-observation augmentation."""
from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import logsumexp
from .privileged import PrivilegedEvaluator,GuidanceBatch
from uav_ac.planning.end2end.direct import world_problem
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective
from uav_ac.planning.trajectory.gcopter.mappings import polynomial_basis_matrix
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state


class DirectEvaluator(PrivilegedEvaluator):
    """Full 34-variable costs; model residuals do not reject measured states."""
    def __init__(self,*args,peak_weight=0.,**kwargs):
        super().__init__(*args,**kwargs);self.peak_weight=float(peak_weight)
        if not np.isfinite(self.peak_weight) or self.peak_weight<0:raise ValueError('invalid peak weight')
        self.last_result=None

    def _peak(self,durations,coeff):
        values=[];partials=[];time_partials=[]
        for piece,duration in enumerate(durations):
            for fraction in np.linspace(0.,1.,17):
                bases=np.stack([polynomial_basis_matrix(np.array([fraction*duration]),d)[0] for d in range(5)])
                state=bases@coeff[piece]
                env=self._parts['environment'].collision_cost_gradient(state[0],state[2])
                own=self._parts['self'].collision_cost_gradient(state[0],state[2])
                value=env[0]+own[0];gradient=np.zeros((4,8));gradient[0]=env[1]+own[1];gradient[2]=env[2]+own[2]
                gc=np.zeros_like(coeff);gc[piece]=np.einsum('dk,dc->kc',bases[:4],gradient)
                gt=np.zeros(2);gt[piece]=fraction*np.sum(gradient*state[1:5])
                values.append(value);partials.append(gc);time_partials.append(gt)
        scale=max(1.,self.config.obstacle_weight*self.config.obstacle_clearance)
        logits=10*np.asarray(values)/scale;normalizer=logsumexp(logits)
        weights=np.exp(logits-normalizer)
        peak=scale*(normalizer-np.log(len(logits)))/10
        factor=self.peak_weight*np.sum(durations)
        return factor*peak,factor*np.einsum('n,nijk->ijk',weights,partials),factor*np.einsum('n,ni->i',weights,time_partials)+self.peak_weight*peak

    def evaluate(self,obs,parameters):
        obs.validate_joints(self.robot.limits.joint_lower,self.robot.limits.joint_upper)
        if not np.isclose(obs.configuration[11],self.gripper_gap):raise ValueError('gripper mismatch')
        batch=np.asarray(parameters,dtype=float)
        if batch.ndim!=2 or batch.shape[1]!=34 or len(batch)==0 or not np.isfinite(batch).all():
            raise ValueError('parameters must be finite (K,34)')
        costs=[];grads=[];violations=[];parts={}
        scales=np.r_[np.full(3,self.config.position_scale),self.config.yaw_scale,self.config.joint_scales]
        for z in batch:
            minco,points,times,tail,frame=world_problem(z,obs)
            cost,gp,gt,ge,measured,worst=self._objective(minco,points,times,time_weight=self.config.time_weight)
            error=tail[0]-obs.goal;ids=np.r_[0:3,4:8]
            progress=10.*(np.sum((error[ids]/scales[ids])**2)+2*(1-np.cos(error[3]))/scales[3]**2)
            derivative=20.*error/scales**2;derivative[3]=20.*np.sin(error[3])/scales[3]**2
            ge[0]+=derivative;cost+=progress;measured['progress']=float(progress)
            # The initial rate mismatch is a soft differentiable cost, not admission.
            def rate_term(durations,coeff):
                values=np.stack([obs.start_pva[0],obs.start_pva[1],obs.start_pva[2],6*coeff[0,3]])
                _,velocity=recover_state(values,self.gripper_gap,self.evaluator.quad.g)
                difference=velocity[3:6]-obs.velocity[3:6]
                jac=[]
                for axis in range(3):
                    shifted=values.copy();shifted[3,axis]+=1.
                    _,v=recover_state(shifted,self.gripper_gap,self.evaluator.quad.g)
                    jac.append(v[3:6]-velocity[3:6])
                gradient=20*np.asarray(jac)@difference
                gc=np.zeros_like(coeff);gc[0,3,:3]=6*gradient
                return 10*float(difference@difference),gc,np.zeros(2)
            extra=(rate_term,self._peak) if self.peak_weight else (rate_term,)
            value,xgp,xgt,_,_,xge=evaluate_minco_objective(minco,points,times,terms=extra,
                energy_weights=np.zeros(8),boundary_gradients=True)
            cost+=value;gp+=xgp;gt+=xgt;ge+=xge;measured['continuity_and_peak']=value
            local_ge=ge.copy();local_ge[:,:3]=frame.to_local_vectors(ge[:,:3])
            gradient=np.r_[frame.pullback_points8(gp).ravel(),local_ge.ravel(),gt]
            if not np.isfinite(cost) or not np.isfinite(gradient).all():raise FloatingPointError('nonfinite direct guidance')
            costs.append(cost);grads.append(gradient);violations.append(worst)
            for name,value in measured.items():parts.setdefault(name,[]).append(value)
        result=GuidanceBatch(np.asarray(costs),np.asarray(grads),{k:np.asarray(v) for k,v in parts.items()},np.asarray(violations))
        self.last_result=result
        return result


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
