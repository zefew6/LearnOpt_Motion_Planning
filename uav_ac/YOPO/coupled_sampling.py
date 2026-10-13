"""Validated authored MINCO sources for arm-sensitive independent observations.

These are geometric planning witnesses, not optimized demonstrations or physics
rollouts. The frozen-arm comparison keeps exactly the same base motion.
"""
from dataclasses import replace
import numpy as np
from .privileged import PrivilegedEvaluator
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.planning.end2end.contracts import PlannerObservation
from uav_ac.simulation.depth_observation import IdealDepthObserver

VALIDATION_DT=.025
WORLD_CLEARANCE=.05


def authored_reference(seed,variant):
    rng=np.random.default_rng(seed)
    fold=np.array([0. if variant==0 else 1.5,-1.6,2.15,-.55])
    height=1.65+float(rng.uniform(-.005,.005))
    states=np.zeros((7,8));states[:,0]=[-.10,.25,.75,1.55,2.5,2.9,3.15]
    states[:,2]=-height
    states[:,1]=(.025 if variant%2 else -.025)*np.sin(np.linspace(0,np.pi,7))
    states[:,4:]=np.array([0.,.55,1.,1.,1.,.45,0.])[:,None]*fold
    states[0,4:]=rng.uniform(-.06,.06,4)
    states[-1,4:]=rng.uniform(-.10,.10,4)
    if variant==1:
        # Lateral extension makes front-gap width matter independently of height.
        states[:,4]=1.5
        states[0,5:]=[-1.25,-.05,0.]
        states[-1,5:]=[-1.25,-.05,0.]
    head=np.vstack((states[0],np.zeros((2,8))));tail=np.vstack((states[-1],np.zeros((2,8))))
    durations=np.full(6,2.)
    coeff,_=MINCOQuintic(head,tail,6).solve(states[1:-1],durations)
    return AerialManipulatorTrajectory(durations,coeff.reshape(6,6,8),states,float('nan'),0,False,
                                      'authored MINCO state source; not optimized')


def frozen_diagnostic(sim,trajectory,gap):
    frames=0;base_frames=0;first=None;pairs=set();minimum=np.inf
    base_names={g['name'] for g in sim.robot.collision_geometries() if g['body']=='quadrotor'}
    for t in np.linspace(0,trajectory.total_time,int(np.ceil(trajectory.total_time/VALIDATION_DT))+1):
        derivatives=np.stack([trajectory.evaluate(t,d) for d in range(4)])
        derivatives[0,4:]=trajectory.evaluate(0)[4:]
        derivatives[1:,4:]=0.
        q,_=recover_state(derivatives,gap)
        result=sim.robot.check_collision(q,clearance=WORLD_CLEARANCE,self_clearance=0.)
        minimum=min(minimum,result['minimum_world_distance'])
        # Physical penetration is stronger evidence than a clearance-only failure.
        if result['collision']:
            frames+=1
            if first is None:first=float(t)
            pairs.update(tuple(p['geoms']) for p in result['pairs'])
        if any(a in base_names or b in base_names for a,b in (p['geoms'] for p in result['pairs'])):base_frames+=1
    return {'frozen_arm_collision_frames':frames,'base_collision_frames':base_frames,
            'frozen_first_collision_time':first,'frozen_minimum_world_distance_m':float(minimum),
            'frozen_collision_pairs':[list(p) for p in sorted(pairs)]}


def build_validated_sources(sim,gap,*,seed):
    cfg=replace(AerialManipulatorMINCOConfig(),self_clearance=0.,obstacle_clearance=WORLD_CLEARANCE,
                validation_dt=VALIDATION_DT,esdf_resolution=.04,esdf_discretization_margin=.07)
    evaluator=PrivilegedEvaluator.from_simulation(sim,gripper_gap=gap,config=cfg).evaluator
    sources=[]
    for variant in range(2):
        trajectory=authored_reference(seed,variant)
        passed,clearance,violation,dt=evaluator.dense_validate(trajectory)
        metrics=dict(evaluator.last_validation_metrics)
        if not passed:raise RuntimeError(f'coupled source rejected variant={variant}: {metrics}')
        trajectory=replace(trajectory,validation_passed=True,minimum_clearance=clearance,maximum_violation=violation,
                           validation_sample_dt=dt,validation_performed=True)
        counter=frozen_diagnostic(sim,trajectory,gap)
        if counter['frozen_arm_collision_frames']==0 or counter['base_collision_frames']:
            raise RuntimeError(f'scene does not establish arm-sensitive witness: {counter}')
        diagnostic={**metrics,**counter,'variant':variant,'validation_passed':True,'validation_dt':dt,
                    'world_clearance_m':WORLD_CLEARANCE,'self_clearance_m':0.,'maximum_violation':violation,
                    'maximum_joint_change_rad':float(np.max(np.abs(trajectory.rrt_path[:,4:]-trajectory.rrt_path[0,4:]))),
                    'source':'authored MINCO reference; no search/optimizer/physics rollout'}
        sources.append((trajectory,diagnostic))
    pairs=[name for _,d in sources for pair in d['frozen_collision_pairs'] for name in pair]
    if not any(n.startswith('obstacle_low_') for n in pairs) or not any(n.startswith(('obstacle_front_','obstacle_trunk_')) for n in pairs):
        raise RuntimeError('sources must establish both height and width arm coupling')
    return sources


def sample_coupled_scene(sim,config,seed):
    sources=build_validated_sources(sim,config.gripper_gap,seed=seed)
    samples=[];records=[]
    # Observe vertical and lateral starting shapes across all movement phases.
    with IdealDepthObserver(sim.model,sim.data,spec=config.depth) as observer:
        allocations=[config.samples_per_scene//2,config.samples_per_scene-config.samples_per_scene//2]
        for variant,((trajectory,diagnostic),count) in enumerate(zip(sources,allocations)):
            for timestamp in np.linspace(.35,trajectory.total_time-.35,count):
                derivatives=np.stack([trajectory.evaluate(timestamp,d) for d in range(4)])
                q,v=recover_state(derivatives,config.gripper_gap)
                sim.robot.reset(q.copy(),v.copy());sim.data.time=float(timestamp)
                depth,mask=observer.observe()
                goal=trajectory.evaluate(trajectory.total_time)
                obs=PlannerObservation(depth,mask,sim.robot.configuration,sim.robot.velocity,derivatives[:3],goal,
                                       config.depth,float(timestamp),derivatives[3],'reference')
                samples.append(obs)
                x=float(obs.configuration[0]);phase='approach' if x<.65 else 'traverse' if x<2.5 else 'restore'
                records.append({'variant':variant,'timestamp':float(timestamp),'phase':phase})
    return samples,sources,records
