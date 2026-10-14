"""Independent single-frame states in fixed, large forest/factory maps.

No expert trajectories, rollouts or claim of global start-to-goal reachability.
Each map stays fixed while its initial states vary. Validation/test use other maps.
"""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from .scene_sampling import MODEL_ROOT
from uav_ac.planning.end2end.contracts import DepthSpec,PlannerObservation
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.tasks.whole_body_reaching import heading_and_rate


def write_fixed_dense_scene(directory,index,seed,spec):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=directory/'model';model.mkdir(exist_ok=True)
    if not (model/'aerial_manipulator.xml').exists():shutil.copyfile(MODEL_ROOT/'model/aerial_manipulator.xml',model/'aerial_manipulator.xml')
    root=ET.parse(MODEL_ROOT/'models/whole_body_dense_outdoor.xml').getroot()
    sid=f'scene_{index:03d}';root.set('model',sid)
    root.find('worldbody/body/include').set('file','model/aerial_manipulator.xml')
    if not np.allclose(spec.camera_extrinsics,DepthSpec().camera_extrinsics):raise ValueError('fixed dense camera mount mismatch')
    camera=root.find('worldbody/body/camera');camera.set('resolution',f'{spec.width} {spec.height}')
    camera.set('focalpixel',f'{spec.intrinsics[0,0]:.12g} {spec.intrinsics[1,1]:.12g}')
    # Keep the authored portals/passages. Vary complete tree positions together
    # with their crown, and reflect maps to obtain scene-disjoint geometry.
    rng=np.random.default_rng(seed);offsets={}
    for geom in root.findall('worldbody/geom'):
        name=geom.get('name','')
        if not name.startswith('obstacle_'):continue
        pos=np.fromstring(geom.get('pos'),sep=' ')
        if index%2:pos[1]=-pos[1]
        if 'tree' in name or 'crown' in name or 'bamboo' in name:
            key=name.replace('trunk','tree').replace('crown','tree')
            if key not in offsets:offsets[key]=rng.uniform(-.12,.12,2)
            offset=offsets[key]
            pos[:2]+=offset
        # Unique geometry across maps, even if obstacle names differ from above.
        if 'wall' in name:pos[0]+=.025*index
        geom.set('pos',' '.join(map(str,pos)))
    ET.indent(root,space='  ');path=directory/f'{sid}.xml';ET.ElementTree(root).write(path,encoding='unicode')
    return sid,'dense',path


def sample_fixed_dense_scene(sim,config,seed,scene_path):
    rng=np.random.default_rng(seed);samples=[];rejected={'start_collision':0,'goal_collision':0,'invalid_state':0}
    root=ET.parse(scene_path).getroot()
    goal=np.fromstring(root.find("custom/numeric[@name='whole_body_goal']").get('data'),sep=' ')
    qgoal,_=recover_state(np.vstack((goal,np.zeros((3,8)))),config.gripper_gap)
    if sim.robot.check_collision(qgoal,clearance=.05,self_clearance=0.)['pairs']:raise ValueError('fixed goal collides')
    with IdealDepthObserver(sim.model,sim.data,spec=config.depth) as observer:
        for attempt in range(config.samples_per_scene*config.max_attempts_per_sample):
            y=np.zeros(8);y[:3]=rng.uniform([.0,-4.9,-2.9],[22.,4.9,-1.25])
            direction=goal[:3]-y[:3];yaw=np.arctan2(direction[1],direction[0]);y[3]=yaw+rng.uniform(-.85,.85)
            mode=rng.choice([0.,.5,1.]);y[4:]=mode*np.array([0.,-1.6,1.9,-.3])+rng.uniform(-.12,.12,4)
            derivatives=np.zeros((4,8));derivatives[0]=y
            # Mix stationary/slow starts with moving starts, all independent.
            speed=0. if rng.random()<.2 else rng.uniform(0.,2.5)
            derivatives[1,:3]=speed*direction/max(np.linalg.norm(direction),1e-6)+rng.uniform(-.15,.15,3)
            derivatives[1,3:]=rng.uniform(-.3,.3,5)
            derivatives[2,:3]=rng.uniform(-1.5,1.5,3);derivatives[2,3:]=rng.uniform(-.6,.6,5)
            derivatives[3,:3]=rng.uniform(-3.,3.,3)
            try:
                q,v=recover_state(derivatives,config.gripper_gap)
                v[3:6]+=rng.uniform(-.15,.15,3);derivatives[1,3]=heading_and_rate(q,v)[1]
                if np.any(y[4:]<sim.robot.limits.joint_lower) or np.any(y[4:]>sim.robot.limits.joint_upper):
                    rejected['invalid_state']+=1;continue
                if sim.robot.check_collision(q,clearance=.05,self_clearance=0.)['pairs']:
                    rejected['start_collision']+=1;continue
                sim.robot.reset(q,v);depth,mask=observer.observe()
                obs=PlannerObservation(depth,mask,sim.robot.configuration,sim.robot.velocity,derivatives[:3],goal,config.depth,0.,derivatives[3],'augmentation')
            except ValueError:
                rejected['invalid_state']+=1;continue
            samples.append(obs)
            if len(samples)==config.samples_per_scene:return samples,rejected
    raise RuntimeError(f'fixed dense sampling exhausted: {rejected}')
