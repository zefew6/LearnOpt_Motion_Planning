"""Seeded XML instances and endpoint sampling, without path-solver filtering."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.planning.end2end.contracts import PlannerObservation
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.simulation.mujoco_sim import MujocoSimulation

MODEL_ROOT=Path(__file__).resolve().parents[1]/'simulation'


def write_scene(directory,index,seed,spec,*,profile='stationary_v1'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    models=directory/'model';models.mkdir(exist_ok=True)
    robot=models/'aerial_manipulator.xml'
    if not robot.exists():shutil.copyfile(MODEL_ROOT/'model'/'aerial_manipulator.xml',robot)
    families=('free','obstacle','corridor','dense') if profile=='rolling_v2' else ('free','obstacle','corridor')
    family=families[index%len(families)]
    root=ET.parse(MODEL_ROOT/'models'/f"yopo_{'free' if family=='dense' else family}.xml").getroot()
    scene_id=f'scene_{index:03d}';root.set('model',scene_id)
    root.find("worldbody/body/include").set('file','model/aerial_manipulator.xml')
    camera=root.find("worldbody/body/camera")
    # XML camera is a fixed body mount; custom configuration must retain that extrinsic.
    from uav_ac.planning.end2end.contracts import DepthSpec
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
                from uav_ac.planning.end2end.encoding import LocalFrame
                from uav_ac.planning.end2end.parameterization import InteriorMap
                from uav_ac.planning.end2end.decoder import decode_trajectory
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
