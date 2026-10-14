"""Training-only ESDF/whole-body cost oracle, retaining MINCO's derivatives."""
from copy import copy
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
import numpy as np
from uav_ac.yopo.whole_body.trajectory.parameterization import InteriorMap
from uav_ac.yopo.whole_body.model.encoding import LocalFrame
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective, integrate_samples
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import AerialManipulatorTrajectoryEvaluator
from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.geometry.esdf import ESDF
import torch
from torch.nn import functional as F
from scipy.special import logsumexp
from uav_ac.yopo.whole_body.trajectory.minco import world_problem
from uav_ac.planning.trajectory.gcopter.mappings import polynomial_basis_matrix
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state

@dataclass(frozen=True)
class GuidanceBatch:
    costs: np.ndarray
    free_gradients: np.ndarray
    components: dict
    violations: np.ndarray
    collided: np.ndarray | None = None
    detached_gradients: np.ndarray | None = None
    collision_components: dict | None = None


class PrivilegedEvaluator:
    def __init__(self,robot,esdf,quad,bounds,*,gripper_gap=.06,config=None,occupancy=None):
        base_config=AerialManipulatorMINCOConfig() if config is None else config
        self.config=replace(base_config,integral_resolution=max(16,base_config.integral_resolution),
                            integral_resolution_floor_unloaded=16)
        self.robot=robot;self.gripper_gap=float(gripper_gap)
        self.evaluator=AerialManipulatorTrajectoryEvaluator(robot,esdf,quad,self.config,bounds,
                                                           gripper_gap,False,occupancy=occupancy)
        self._parts={}
        for kind in ('environment','self'):
            part=copy(self.evaluator);settings=asdict(self.config)
            settings['self_collision_weight' if kind=='environment' else 'obstacle_weight']=0.
            part.config=SimpleNamespace(**settings);self._parts[kind]=part

    @classmethod
    def from_simulation(cls,simulation,*,gripper_gap=.06,config=None):
        cfg=AerialManipulatorMINCOConfig() if config is None else config
        bounds=simulation.space_limits
        if bounds is None: raise ValueError('scene must declare planning_bounds')
        occupancy=GridMap.from_scene_geometries(simulation.scene_geometries,bounds[0]-.8,bounds[1]+.8,cfg.esdf_resolution)
        return cls(simulation.robot,ESDF.from_occupancy(occupancy),simulation.quad,bounds,
                   gripper_gap=gripper_gap,config=cfg,occupancy=occupancy)

    def _objective(self,minco,points,durations,*,time_weight=0.):
        measured={};worst=[-np.inf]
        def term(kind):
            def integrated(times,coeff):
                def sample(piece,alpha,values):
                    if kind=='physical':
                        cost,gradient,violation=self.evaluator._physical_cost_gradient(*values[:4])
                    else:
                        result=self._parts[kind].collision_cost_gradient(values[0],values[2])
                        cost=result[0];gradient=np.zeros((4,8));gradient[0]=result[1];gradient[2]=result[2]
                        violation=max(result[3],result[5])
                    worst[0]=max(worst[0],violation)
                    return cost,np.asarray(gradient)
                result=integrate_samples(times,coeff,self.config.integral_resolution,sample)
                measured[kind]=result[0]
                return result
            return integrated
        cost,gp,gt,coeff,gh,ge=evaluate_minco_objective(minco,points,durations,
            terms=tuple(term(k) for k in ('environment','self','physical')),
            time_weight=time_weight,energy_weights=np.asarray(self.config.jerk_weights),
            boundary_gradients=True)
        measured['jerk']=minco.jerk_energy(coeff,durations,np.asarray(self.config.jerk_weights))[0]
        if time_weight:measured['time']=time_weight*float(np.sum(durations))
        return cost,gp,gt,ge,measured,worst[0]

    def evaluate(self,obs,free_batch):
        obs.validate_joints(self.robot.limits.joint_lower,self.robot.limits.joint_upper)
        if not np.isclose(obs.configuration[11],self.gripper_gap): raise ValueError('observation gripper does not match oracle')
        # V1 is strictly stationary-to-stationary; dynamic data requires a new profile.
        if np.max(abs(obs.velocity))>1e-8 or np.max(abs(obs.start_pva[1:]))>1e-8:
            raise ValueError('reference_v1 requires a stationary start')
        from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
        expected,_=recover_state(np.vstack((obs.start_pva,np.zeros((1,8)))),self.gripper_gap)
        if not np.isclose(abs(expected[3:7]@obs.configuration[3:7]),1.,atol=1e-7,rtol=0):
            raise ValueError('start attitude is inconsistent with reference_v1 flatness')
        batch=np.asarray(free_batch,dtype=float)
        if batch.ndim!=2 or batch.shape[1]!=20 or not len(batch) or not np.isfinite(batch).all():
            raise ValueError('free_batch must be finite (K,20), K>0')
        frame=LocalFrame.from_observation(obs)
        local_head=frame.local_pva(obs.start_pva);local_tail=np.vstack((frame.local_goal(obs.goal),np.zeros((2,8))))
        mapping=InteriorMap(local_head,local_tail)
        # The world boundary uses the same unwrapped yaw as the local decoder.
        world_tail=local_tail.copy();world_tail[0,:3]=frame.to_world_points(local_tail[0,:3]);world_tail[0,3]+=frame.yaw
        world_minco=InteriorMap(obs.start_pva,world_tail).minco
        costs=[];grads=[];violations=[];components={name:[] for name in ('jerk','environment','self','physical')}
        for free in batch:
            points=frame.world_points8(mapping.decode(free))
            cost,gp,_,_,measured,worst=self._objective(world_minco,points,mapping.durations)
            grad=mapping.pullback(frame.pullback_points8(gp))
            if not np.isfinite(cost) or not np.isfinite(grad).all(): raise FloatingPointError('nonfinite guidance candidate')
            costs.append(cost);grads.append(grad);violations.append(worst)
            for name in components: components[name].append(measured[name])
        return GuidanceBatch(np.asarray(costs),np.asarray(grads),{k:np.asarray(v) for k,v in components.items()},np.asarray(violations))


class RollingEvaluator(PrivilegedEvaluator):
    """Training-only rolling costs; same whole-body geometry, free terminal PVA/time."""
    def evaluate(self,obs,free_batch):
        from uav_ac.yopo.whole_body.trajectory.parameterization import RollingMap
        from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
        obs.validate_joints(self.robot.limits.joint_lower,self.robot.limits.joint_upper)
        if not np.isclose(obs.configuration[11],self.gripper_gap):
            raise ValueError('observation gripper does not match oracle')
        batch=np.asarray(free_batch,dtype=float)
        if batch.ndim!=2 or batch.shape[1]!=32 or not len(batch) or not np.isfinite(batch).all():
            raise ValueError('rolling_free must be finite (K,32)')
        mapping=RollingMap.from_observation(obs,gravity=self.evaluator.quad.g)
        frame=LocalFrame.from_observation(obs)
        costs=[];gradients=[];violations=[];components={}
        scales=np.r_[np.full(3,self.config.position_scale),self.config.yaw_scale,self.config.joint_scales]
        for free in batch:
            points,tail,durations=mapping.decode(free)
            world_tail=tail.copy();world_tail[0,:3]=frame.to_world_points(tail[0,:3])
            world_tail[1:,:3]=frame.to_world_vectors(tail[1:,:3]);world_tail[0,3]+=frame.yaw
            minco=MINCOQuintic(obs.start_pva,world_tail,2)
            cost,gp,gt,ge,measured,worst=self._objective(minco,frame.world_points8(points),durations,
                                                     time_weight=self.config.time_weight)
            error=world_tail[0]-obs.goal
            progress=10.*float(np.sum((error[np.r_[0:3,4:8]]/scales[np.r_[0:3,4:8]])**2)
                               +2*(1-np.cos(error[3]))/scales[3]**2)
            derivative=20.*error/scales**2;derivative[3]=20.*np.sin(error[3])/scales[3]**2
            ge[0]+=derivative;cost+=progress;measured['progress']=progress
            local_ge=ge.copy();local_ge[:,:3]=frame.to_local_vectors(ge[:,:3])
            grad=mapping.pullback(frame.pullback_points8(gp),local_ge,gt,free)
            if not np.isfinite(cost) or not np.isfinite(grad).all():
                raise FloatingPointError('nonfinite rolling guidance')
            costs.append(cost);gradients.append(grad);violations.append(worst)
            for name,value in measured.items():components.setdefault(name,[]).append(value)
        return GuidanceBatch(np.asarray(costs),np.asarray(gradients),
                             {k:np.asarray(v) for k,v in components.items()},np.asarray(violations))

# Direct-PVA guidance and independent offline-observation augmentation.


class _GeometryRobot:
    """Use actual collision geoms for broad/narrow phase, not authored sphere costs."""
    def __init__(self,robot):self.robot=robot
    def planning_collision_spheres(self):return ()
    def __getattr__(self,name):return getattr(self.robot,name)


class DirectEvaluator(PrivilegedEvaluator):
    """Full 34-variable costs; model residuals do not reject measured states."""
    def __init__(self,*args,peak_weight=0.,goal_stop_weight=0.,goal_stop_distance=1.5,**kwargs):
        super().__init__(*args,**kwargs);self.peak_weight=float(peak_weight)
        if not np.isfinite(self.peak_weight) or self.peak_weight<0:raise ValueError('invalid peak weight')
        self.goal_stop_weight=float(goal_stop_weight);self.goal_stop_distance=float(goal_stop_distance)
        if not np.isfinite([self.goal_stop_weight,self.goal_stop_distance]).all() or self.goal_stop_weight<0 or self.goal_stop_distance<=0:raise ValueError('invalid stopping cost')
        self.last_result=None

    @classmethod
    def from_simulation(cls,simulation,*,gripper_gap=.06,config=None,goal_stop_weight=0.,goal_stop_distance=1.5,collision_model='planning_spheres'):
        if collision_model not in ('planning_spheres','exact_geometry'):raise ValueError('unknown collision model')
        if not np.isfinite([goal_stop_weight,goal_stop_distance]).all() or goal_stop_weight<0 or goal_stop_distance<=0:raise ValueError('invalid stopping cost')
        if collision_model=='exact_geometry':
            from types import SimpleNamespace
            simulation=SimpleNamespace(robot=_GeometryRobot(simulation.robot),quad=simulation.quad,space_limits=simulation.space_limits,scene_geometries=simulation.scene_geometries)
        result=super().from_simulation(simulation,gripper_gap=gripper_gap,config=config)
        result.goal_stop_weight=float(goal_stop_weight);result.goal_stop_distance=float(goal_stop_distance)
        return result

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
            gain=self.goal_stop_weight*max(0.,1.-np.linalg.norm(obs.goal[:3]-obs.configuration[:3])/self.goal_stop_distance)
            stop=gain*float(np.sum((tail[1]/scales)**2))
            ge[1]+=2*gain*tail[1]/scales**2;cost+=stop;measured['goal_stop']=stop
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
