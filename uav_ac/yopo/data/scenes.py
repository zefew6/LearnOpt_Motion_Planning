"""Generate small guidance/acceptance scenes from the single YOPO benchmark.

Scene instances are artifacts written to caller-owned directories under runs;
there are no separate static training/debug XML templates.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import numpy as np
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.yopo.contracts import PlannerObservation, DepthSpec
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from dataclasses import replace
from uav_ac.yopo.training.legacy import PrivilegedEvaluator
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.tasks.whole_body_reaching import heading_and_rate

MODEL_ROOT=Path(__file__).resolve().parents[2]/'simulation'
FAMILIES=('free','obstacle','low','wall')
OBSTACLES={
    'free':(),
    'obstacle':(('block',(1.5,0,1.1),(.20,.30,1.1)),),
    'low':(('low',(1.5,0,.64),(.22,1.6,.64)),('ceiling',(1.5,0,2.07),(.40,1.6,.13))),
    'wall':(('wall_bottom',(1.5,0,.64),(.16,1.6,.64)),('wall_top',(1.5,0,2.39),(.16,1.6,.41)),
            ('wall_left',(1.5,-1.125,1.63),(.16,.475,.35)),('wall_right',(1.5,1.125,1.63),(.16,.475,.35))),
}


def stage_scene_root(family,*,model_root=MODEL_ROOT):
    if family not in FAMILIES:raise ValueError('unknown guidance scene family')
    root=ET.parse(Path(model_root)/'models/yopo_bamboo.xml').getroot()
    root.set('model',f'yopo_stage_{family}')
    world=root.find('worldbody');custom=root.find('custom')
    for geom in list(world.findall('geom')):
        if geom.get('name','').startswith('obstacle_'):world.remove(geom)
    root.find('statistic').set('center','2.6 0 1.3');root.find('statistic').set('extent','4.5')
    bounds='-.8 -1.3 -2.0 4.2 1.3 -1.6' if family in ('low','wall') else '-.8 -1.3 -3.0 4.2 1.3 -.8'
    custom.find("numeric[@name='planning_bounds']").set('data',bounds)
    custom.find("numeric[@name='whole_body_start']").set('data','0 0 -1.8 0 0 0 0 0')
    custom.find("numeric[@name='whole_body_goal']").set('data','3 0 -1.8 0 0 0 0 0')
    ground=world.find("geom[@name='ground']");ground.set('size','8 5 0.1');ground.attrib.pop('pos',None)
    world.find('body').set('pos','0 0 1.8')
    world.find("site[@name='start']").set('pos','0 0 1.8')
    world.find("site[@name='goal']").set('pos','3 0 1.8')
    for name,position,size in OBSTACLES[family]:
        ET.SubElement(world,'geom',name='obstacle_'+name,type='box',pos=' '.join(map(str,position)),
            size=' '.join(map(str,size)),rgba='.5 .55 .6 1')
    return root


def write_debug_scene(directory,family):
    root=stage_scene_root(family)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    root.find('worldbody/body/include').set('file',str((MODEL_ROOT/'model/aerial_manipulator.xml').resolve()))
    path=directory/f'yopo_stage_{family}.xml'
    ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='unicode')
    return path.resolve()

# Seeded XML instances and endpoint sampling, without path-solver filtering.



def legacy_scene_root():
    """Derive a free scene from the benchmark; retain older dataset bounds."""
    root=stage_scene_root('free',model_root=MODEL_ROOT)
    custom=root.find('custom');world=root.find('worldbody')
    custom.find("numeric[@name='planning_bounds']").set('data','-1.2 -1.6 -2.4 2.8 1.6 -0.75')
    for numeric in list(custom.findall('numeric')):
        if numeric.get('name','').startswith('whole_body_'):custom.remove(numeric)
    for site in list(world.findall('site')):
        if site.get('name') in ('start','goal'):world.remove(site)
    world.find('body').set('pos','0 0 1.5')
    return root


def write_scene(directory,index,seed,spec,*,profile='stationary_v1'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    models=directory/'model';models.mkdir(exist_ok=True)
    robot=models/'aerial_manipulator.xml'
    if not robot.exists():shutil.copyfile(MODEL_ROOT/'model'/'aerial_manipulator.xml',robot)
    families=('free','obstacle','corridor','dense') if profile=='rolling_v2' else ('free','obstacle','corridor')
    family=families[index%len(families)]
    root=legacy_scene_root()
    world=root.find('worldbody')
    obstacles={'obstacle':(('box',(1.3,.7,1.4),(.2,.25,.4)),),
        'corridor':(('left',(1.4,-.95,1.3),(.6,.18,1.3)),('right',(1.4,.95,1.3),(.6,.18,1.3)))}
    for name,position,size in obstacles.get(family,()):
        ET.SubElement(world,'geom',name='obstacle_'+name,type='box',pos=' '.join(map(str,position)),
            size=' '.join(map(str,size)),rgba='.5 .5 .6 1')
    scene_id=f'scene_{index:03d}';root.set('model',scene_id)
    root.find("worldbody/body/include").set('file','model/aerial_manipulator.xml')
    camera=root.find("worldbody/body/camera")
    # XML camera is a fixed body mount; custom configuration must retain that extrinsic.
    from uav_ac.yopo.contracts import DepthSpec
    if not np.allclose(spec.camera_extrinsics,DepthSpec().camera_extrinsics):
        raise ValueError('dataset camera mount must match the robot XML')
    camera.set('resolution',f'{spec.width} {spec.height}')
    camera.set('focalpixel',f'{spec.intrinsics[0,0]:.12g} {spec.intrinsics[1,1]:.12g}')
    rng=np.random.default_rng(seed)
    if family=='dense':
        for i in range(8):
            side=-1 if i%2 else 1
            ET.SubElement(root.find('worldbody'),'geom',name=f'obstacle_dense_{i}',type='box',
                pos=f'{.65+.25*i} {side*(.35+.10*(i%3))} 1.25',
                size='.075 .08 .5',rgba='.5 .5 .6 1')
    for geom in root.findall("worldbody/geom"):
        if not geom.get('name','').startswith('obstacle_'):continue
        pos=np.fromstring(geom.get('pos'),sep=' ');size=np.fromstring(geom.get('size'),sep=' ')
        pos[0]+=rng.uniform(-.2,.2);pos[1]+=rng.uniform(-.12,.12);size*=rng.uniform(.9,1.1,3)
        geom.set('pos',' '.join(f'{v:.12g}' for v in pos));geom.set('size',' '.join(f'{v:.12g}' for v in size))
    # Even free-space scene instances vary physically: obstacle-free ground height.
    ground=root.find("worldbody/geom[@name='ground']")
    ground.set('pos',f'0 0 {rng.uniform(-.12,-.01):.12g}')
    ET.indent(root);path=directory/f'{scene_id}.xml';ET.ElementTree(root).write(path,encoding='unicode')
    return scene_id,family,path


def sample_scene(path,config,seed):
    rng=np.random.default_rng(seed);sim=MujocoSimulation(path,planning_path_capacity=0,record_actual_trajectory=False)
    lower,upper=sim.robot.limits.joint_lower,sim.robot.limits.joint_upper
    rejected={'limits':0,'start_collision':0,'goal_collision':0,'dynamic_constraints':0,'snapshot_collision':0};samples=[]
    max_attempts=config.samples_per_scene*config.max_attempts_per_sample
    with IdealDepthObserver(sim.model,sim.data,spec=config.depth) as observer:
        for attempt in range(max_attempts):
            start=np.r_[rng.uniform([-.2,-.6,-1.8],[.2,.6,-1.3]),rng.uniform(-.35,.35),rng.uniform(lower,upper)]
            distance=rng.uniform(.5,2.);angle=start[3]+rng.uniform(-.35,.35)
            goal=np.r_[start[:3]+[distance*np.cos(angle),distance*np.sin(angle),rng.uniform(-.15,.15)],
                       rng.uniform(-.7,.7),rng.uniform(lower,upper)]
            if np.any(goal[:3]<sim.space_limits[0]) or np.any(goal[:3]>sim.space_limits[1]):
                rejected['limits']+=1;continue
            head=np.vstack((start,np.zeros((3,8))))
            tail=np.vstack((goal,np.zeros((3,8))))
            qs,_=recover_state(head,config.gripper_gap);qg,_=recover_state(tail,config.gripper_gap)
            if sim.robot.check_collision(qs,clearance=.05,self_clearance=.015)['pairs']:
                rejected['start_collision']+=1;continue
            if sim.robot.check_collision(qg,clearance=.05,self_clearance=.015)['pairs']:
                rejected['goal_collision']+=1;continue
            timestamp=0.;jerk=np.zeros(8);source='stationary';velocity=np.zeros(11)
            pva=head[:3]
            if config.profile=='rolling_v2':
                from types import SimpleNamespace
                from uav_ac.yopo.model.encoding import LocalFrame
                from uav_ac.yopo.trajectory.legacy import InteriorMap
                from uav_ac.yopo.trajectory.legacy import decode_trajectory
                from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import PhysicalConstraints
                from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
                frame=LocalFrame(start[:3],start[3])
                mapping=InteriorMap(frame.local_pva(head[:3]),np.vstack((frame.local_goal(goal),np.zeros((2,8)))))
                trajectory=decode_trajectory(mapping.seed(),head[:3],goal,gripper_gap=config.gripper_gap)
                timestamp=float(rng.uniform(config.snapshot_min,config.snapshot_max))
                derivatives=np.stack([trajectory.evaluate(timestamp,d) for d in range(4)])
                try:qs,velocity=recover_state(derivatives,config.gripper_gap)
                except ValueError:
                    rejected['dynamic_constraints']+=1;continue
                context=SimpleNamespace(config=AerialManipulatorMINCOConfig(),quad=sim.quad,
                    robot=sim.robot,mass=sim.robot.mass,bounds=sim.space_limits)
                _,_,violation=PhysicalConstraints._physical_cost_gradient_python(context,*derivatives)
                if violation>1e-6 or np.linalg.norm(velocity)<1e-4:
                    rejected['dynamic_constraints']+=1;continue
                if sim.robot.check_collision(qs,clearance=.05,self_clearance=.015)['pairs']:
                    rejected['snapshot_collision']+=1;continue
                pva=derivatives[:3];jerk=derivatives[3];source='reference'
            sim.robot.reset(qs,velocity)
            sim.data.time=timestamp
            depth,mask=observer.observe()
            obs=PlannerObservation(depth,mask,sim.robot.configuration,sim.robot.velocity,pva,goal,config.depth,
                                   timestamp,jerk,source)
            samples.append(obs)
            if len(samples)==config.samples_per_scene:break
        else:raise RuntimeError(f'could not sample endpoints: accepted={len(samples)}, rejected={rejected}')
    return samples,rejected

# Independent current-state sampling for four simple whole-body planning families.



def write_curriculum_scene(directory,index,seed,spec):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=directory/'model';model.mkdir(exist_ok=True)
    if not (model/'aerial_manipulator.xml').exists():shutil.copyfile(MODEL_ROOT/'model/aerial_manipulator.xml',model/'aerial_manipulator.xml')
    family=FAMILIES[index%4];sid=f'scene_{index:03d}'
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
