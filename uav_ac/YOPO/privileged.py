"""Training-only ESDF/whole-body cost oracle, retaining MINCO's derivatives."""
from copy import copy
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
import numpy as np
from uav_ac.planning.end2end.parameterization import InteriorMap
from uav_ac.planning.end2end.encoding import LocalFrame
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective, integrate_samples
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import AerialManipulatorTrajectoryEvaluator
from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.geometry.esdf import ESDF


@dataclass(frozen=True)
class GuidanceBatch:
    costs: np.ndarray
    free_gradients: np.ndarray
    components: dict
    violations: np.ndarray


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
        from uav_ac.planning.end2end.rolling import RollingMap
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
