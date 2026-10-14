"""Independent current-state sampling for four simple whole-body planning families."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from dataclasses import replace
from .scene_sampling import MODEL_ROOT
from .privileged import PrivilegedEvaluator
from uav_ac.planning.end2end.contracts import DepthSpec,PlannerObservation
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.tasks.whole_body_reaching import heading_and_rate

FAMILIES=('free','obstacle','low','wall')


def write_curriculum_scene(directory,index,seed,spec):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=directory/'model';model.mkdir(exist_ok=True)
    if not (model/'aerial_manipulator.xml').exists():shutil.copyfile(MODEL_ROOT/'model/aerial_manipulator.xml',model/'aerial_manipulator.xml')
    family=FAMILIES[index%4];sid=f'scene_{index:03d}'
    from .simple_scenes import stage_scene_root
    root=stage_scene_root(family,model_root=MODEL_ROOT);root.set('model',sid)
    root.find('worldbody/body/include').set('file','model/aerial_manipulator.xml')
    if not np.allclose(spec.camera_extrinsics,DepthSpec().camera_extrinsics):raise ValueError('curriculum camera mount mismatch')
    camera=root.find('worldbody/body/camera');camera.set('resolution',f'{spec.width} {spec.height}')
    camera.set('focalpixel',f'{spec.intrinsics[0,0]:.12g} {spec.intrinsics[1,1]:.12g}')
    rng=np.random.default_rng(seed)
    root.find("worldbody/geom[@name='ground']").set('pos',f'0 0 {rng.uniform(-.08,-.01):.12g}')
    offset=rng.uniform(-.05,.05)
    for geom in root.findall('worldbody/geom'):
        if geom.get('name','').startswith('obstacle_'):
            pos=np.fromstring(geom.get('pos'),sep=' ');pos[0]+=offset;geom.set('pos',' '.join(map(str,pos)))
    ET.indent(root,space='  ');path=directory/f'{sid}.xml';ET.ElementTree(root).write(path,encoding='unicode')
    return sid,family,path


def witness(sim,family,gap):
    start=np.array([0,0,-1.8,0,0,0,0,0.]);goal=start.copy();goal[0]=3.
    mid=(start+goal)/2
    if family=='obstacle':mid[1]=1.
    if family in ('low','wall'):mid[4:]=[0,-1.6,1.9,-.3]
    durations=np.array([2.5,2.5]);head=np.vstack((start,np.zeros((2,8))));tail=np.vstack((goal,np.zeros((2,8))))
    coeff,_=MINCOQuintic(head,tail,2).solve(mid[None],durations)
    tr=AerialManipulatorTrajectory(durations,coeff.reshape(2,6,8),np.vstack((start,mid,goal)),0.,0,False,'offline scene witness; not a training label')
    cfg=replace(AerialManipulatorMINCOConfig(),esdf_resolution=.04,esdf_discretization_margin=.07,self_clearance=0.)
    oracle=PrivilegedEvaluator.from_simulation(sim,gripper_gap=gap,config=cfg)
    passed,clearance,violation,dt=oracle.evaluator.dense_validate(tr)
    report={**oracle.evaluator.last_validation_metrics,'passed':bool(passed),'clearance':float(clearance),'violation':float(violation),'sample_dt':float(dt)}
    if not passed:raise RuntimeError(f'curriculum {family} witness failed: {report}')
    return tr,report


def sample_curriculum_scene(sim,config,seed,family):
    source=witness(sim,family,config.gripper_gap);rng=np.random.default_rng(seed)
    samples=[];records=[]
    with IdealDepthObserver(sim.model,sim.data,spec=config.depth) as observer:
        for attempt in range(config.samples_per_scene*config.max_attempts_per_sample):
            y=np.zeros(8);y[:3]=rng.uniform([-.1,-1.05 if family=='obstacle' else -.3,-1.83],[2.6,1.05 if family=='obstacle' else .3,-1.77]);y[3]=rng.uniform(-.2,.2)
            mode=rng.choice([0.,.5,1.],p=[.5,.25,.25]);y[4:]=mode*np.array([0,-1.6,1.9,-.3])+rng.uniform(-.04,.04,4)
            y[2]=rng.uniform(-2.5,-1.2) if family in ('free','obstacle') else rng.uniform(-1.95,-1.65)
            derivatives=np.zeros((4,8));derivatives[0]=y
            derivatives[1,:3]=rng.uniform([0.,-.2,-.5],[1.5,.2,.5]);derivatives[1,3:]=rng.uniform(-.15,.15,5)
            derivatives[2,:3]=rng.uniform(-1.5,1.5,3);derivatives[2,3:]=rng.uniform(-1.5,1.5,5);derivatives[3,:3]=rng.uniform(-3.,3.,3)
            q,v=recover_state(derivatives,config.gripper_gap)
            v[3:6]+=rng.uniform(-.15,.15,3);derivatives[1,3]=heading_and_rate(q,v)[1]
            goal=np.array([rng.uniform(2.7,3.),rng.uniform(-.1,.1),-1.8,0,0,0,0,0.])
            qgoal,_=recover_state(np.vstack((goal,np.zeros((3,8)))),config.gripper_gap)
            if sim.robot.check_collision(q,clearance=.05,self_clearance=0.)['pairs'] or sim.robot.check_collision(qgoal,clearance=.05,self_clearance=0.)['pairs']:continue
            sim.robot.reset(q,v);depth,mask=observer.observe()
            samples.append(PlannerObservation(depth,mask,sim.robot.configuration,sim.robot.velocity,derivatives[:3],goal,config.depth,0.,derivatives[3],'augmentation'))
            records.append({'sample':len(samples)-1,'attempt':attempt,'source':'independent current-state PVA, no trajectory sampling'})
            if len(samples)==config.samples_per_scene:return samples,[source],records
    raise RuntimeError(f'curriculum {family} sampling exhausted')
