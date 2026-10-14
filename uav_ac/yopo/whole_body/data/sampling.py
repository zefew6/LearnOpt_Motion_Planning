"""Motion-only corridor/bamboo XML instances with arm-sensitive lower obstacles."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from uav_ac.yopo.whole_body.data.scenes import MODEL_ROOT, legacy_scene_root
from uav_ac.yopo.whole_body.contracts import DepthSpec, PlannerObservation
from dataclasses import replace
from uav_ac.yopo.whole_body.training.oracle import PrivilegedEvaluator
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.simulation.depth_observation import IdealDepthObserver

def write_coupled_scene(directory,index,seed,spec):
    if not np.allclose(spec.camera_extrinsics,DepthSpec().camera_extrinsics):
        raise ValueError('coupled scenes retain the authored robot camera mount')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    models=directory/'model';models.mkdir(exist_ok=True)
    if not (models/'aerial_manipulator.xml').exists():
        shutil.copyfile(MODEL_ROOT/'model'/'aerial_manipulator.xml',models/'aerial_manipulator.xml')
    root=legacy_scene_root()
    sid=f'scene_{index:03d}';family=('coupled_corridor','coupled_bamboo')[index%2]
    root.set('model',sid)
    root.find('worldbody/body/include').set('file','model/aerial_manipulator.xml')
    root.find("custom/numeric[@name='planning_bounds']").set('data','-.4 -.65 -1.72 3.5 .65 -1.55')
    camera=root.find('worldbody/body/camera');camera.set('resolution',f'{spec.width} {spec.height}')
    camera.set('focalpixel',f'{spec.intrinsics[0,0]:.12g} {spec.intrinsics[1,1]:.12g}')
    rng=np.random.default_rng(seed);world=root.find('worldbody')
    def geom(name,kind,pos,size,color):
        return ET.SubElement(world,'geom',name='obstacle_'+name,type=kind,
                             pos=' '.join(f'{v:.10g}' for v in pos),size=' '.join(f'{v:.10g}' for v in size),rgba=color)
    # Both the physical ceiling and the base altitude band prevent unconstrained climbing.
    geom('ceiling','box',[1.55,0,1.93],[2.,1.,.05],'.68 .72 .75 .35')
    shift=float(rng.uniform(-.04,.04));height=float(rng.uniform(1.165,1.185))
    ground=root.find("worldbody/geom[@name='ground']");ground.set('pos',f'0 0 {rng.uniform(-.04,-.01):.10g}')
    if family=='coupled_corridor':
        for side in (-1,1):
            geom(f'side_{side}','box',[1.55,side*.86,1.],[2.,.08,1.],'.42 .50 .58 1')
        for j,(x,top,width) in enumerate(((1.45+shift,height,.78),(2.18+shift,1.105,.78))):
            geom(f'low_{j}','box',[x,0,top/2],[.13,width,top/2],'.68 .48 .28 1')
        for j,(x,gap) in enumerate(((1.15+shift,rng.uniform(.62,.72)),(2.35+shift,rng.uniform(.72,.90)))):
            for side in (-1,1):
                geom(f'front_{j}_{side}','box',[x,side*(gap/2+.10),1.],[.10,.10,1.],'.45 .55 .60 1')
    else:
        for j,(x,top,radius) in enumerate(((1.45+shift,height,.22),(2.18+shift,1.105,.18))):
            geom(f'low_{j}','cylinder',[x,0,top/2],[radius,top/2],'.42 .58 .29 1')
        for j,x in enumerate((.85,1.75,2.55)):
            gap=float(rng.uniform(.60,.80));radius=float(rng.uniform(.07,.11))
            for side in (-1,1):
                y=side*(gap/2+radius);h=float(rng.uniform(1.95,2.35))
                geom(f'trunk_{j}_{side}','cylinder',[x+rng.uniform(-.035,.035),y,h/2],[radius,h/2],'.28 .48 .22 1')
        for j in range(8):
            side=(-1,1)[j%2];x=float(rng.uniform(.2,3.0));y=side*float(rng.uniform(.85,1.3));h=float(rng.uniform(.7,2.4))
            geom(f'outer_trunk_{j}','cylinder',[x,y,h/2],[float(rng.uniform(.05,.09)),h/2],'.32 .52 .25 1')
    ET.SubElement(root.find('custom'),'numeric',name='coupled_gate',data=f'{1.45+shift:.10g} {height:.10g}')
    ET.indent(root);path=directory/f'{sid}.xml';ET.ElementTree(root).write(path,encoding='unicode')
    return sid,family,path

# Finite-depth wall openings assembled from exact physical boxes in closed rooms.


def write_portal_scene(directory,index,seed,spec):
    # Reuse camera/model setup only; all coupled obstacles are replaced below.
    sid,_,path=write_coupled_scene(directory,index,seed,spec)
    root=ET.parse(path).getroot();world=root.find('worldbody');custom=root.find('custom')
    for g in list(world.findall('geom')):
        if g.get('name','').startswith('obstacle_'):world.remove(g)
    old=custom.find("numeric[@name='coupled_gate']")
    if old is not None:custom.remove(old)
    family=('portal_rectangle','portal_step','portal_offset')[index%3]
    rng=np.random.default_rng(seed);shift=float(rng.uniform(-.025,.025))
    bottom=float(rng.uniform(1.160,1.175));upper_half=float(rng.uniform(.36,.39))
    if family=='portal_offset':
        upper_half+=.04
        bottom-=.02  # lower wall lip must stay clear during the exit/reconfiguration
    depth=float(rng.uniform(.14,.20))  # half thickness; wall volume is 28–40 cm deep
    def box(name,x,y0,y1,z0,z1,halfx=depth):
        if y1<=y0 or z1<=z0:raise ValueError('empty wall box')
        ET.SubElement(world,'geom',name='obstacle_'+name,type='box',
            pos=f'{x:.10g} {(y0+y1)/2:.10g} {(z0+z1)/2:.10g}',
            size=f'{halfx:.10g} {(y1-y0)/2:.10g} {(z1-z0)/2:.10g}',rgba='.48 .57 .65 1')
    # All room surfaces are physical and retained in sensor images.
    box('ceiling',1.55,-.94,.94,1.88,1.98,2.15)
    for sign in (-1,1):
        box(f'room_side_{sign}',1.55,sign*.89-.05,sign*.89+.05,0,1.98,2.15)
    box('room_end_back',-.60,-.94,.94,0,1.98,.08)
    box('room_end_front',3.70,-.94,.94,0,1.98,.08)
    if family=='portal_offset':walls=[(1.25+shift,-.12),(2.15+shift,.12)]
    else:walls=[(1.45+shift,float(rng.uniform(-.01,.01)))]
    openings=[]
    for i,(x,center) in enumerate(walls):
        stepped=family!='portal_rectangle'
        split=1.46 if stepped else 1.88
        lower_left=center-.16 if stepped else center-upper_half
        lower_right=center+.34 if stepped else center+upper_half
        box(f'low_wall_{i}',x,-.89,.89,0,bottom)
        box(f'front_wall_{i}_top',x,-.89,.89,1.88,1.98)
        box(f'front_wall_{i}_left_lower',x,-.89,lower_left,bottom,split)
        box(f'front_wall_{i}_right_lower',x,lower_right,.89,bottom,split)
        if stepped:
            box(f'front_wall_{i}_left_upper',x,-.89,center-upper_half,split,1.88)
            box(f'front_wall_{i}_right_upper',x,center+upper_half,.89,split,1.88)
        openings.append([x,depth,bottom,split,lower_left,lower_right,center-upper_half,center+upper_half,1.88])
    route=np.zeros((7,2));route[:,0]=[-.10,.25,.75,1.55,2.5,2.9,3.15]
    if family=='portal_offset':
        route[:,0]=[-.10,.25,.90,1.25+shift,2.15+shift,2.85,3.15]
        route[:,1]=[0.,0.,.10,.12,-.12,-.02,0.]
    # Internal vegetation-like obstacles coexist with the wall apertures.
    # Their branches/crowns are conservative primitives, not visual-only decor.
    intervals=[(-.15,walls[0][0]),(walls[-1][0],3.35)]
    intervals.extend((walls[j][0],walls[j+1][0]) for j in range(len(walls)-1))
    obstacle_index=0
    for left,right in intervals:
        x=(left+right)/2
        for side in (-1,1):
            for kind,y,radius,height in [('bamboo',.58,.055,float(rng.uniform(.75,1.78))),
                                          ('tree',.64,.105,float(rng.uniform(1.50,1.78)))]:
                xpos=x+(-.045 if kind=='bamboo' else .045)
                ypos=side*y
                ET.SubElement(world,'geom',name=f'obstacle_{kind}_{obstacle_index}',type='cylinder',
                    pos=f'{xpos:.10g} {ypos:.10g} {height/2:.10g}',size=f'{radius:.10g} {height/2:.10g}',
                    rgba='.26 .45 .22 1' if kind=='bamboo' else '.42 .30 .18 1')
                if kind=='tree':
                    ET.SubElement(world,'geom',name=f'obstacle_crown_{obstacle_index}',type='sphere',
                        pos=f'{xpos:.10g} {ypos:.10g} {height-.08:.10g}',size='.14',rgba='.22 .39 .19 1')
                obstacle_index+=1
    for j in range(len(walls)-1):
        x=(walls[j][0]+walls[j+1][0])/2
        height=float(rng.uniform(1.125,1.145))
        ET.SubElement(world,'geom',name=f'obstacle_tree_middle_low_{j}',type='cylinder',
            pos=f'{x:.10g} 0 {height/2:.10g}',size=f'.14 {height/2:.10g}',rgba='.48 .34 .20 1')
    ET.SubElement(custom,'numeric',name='portal_route_xy_ned',data=' '.join(f'{v:.10g}' for v in route.ravel()))
    ET.SubElement(custom,'numeric',name='portal_openings',data=' '.join(f'{v:.10g}' for v in np.asarray(openings).ravel()))
    ET.indent(root);ET.ElementTree(root).write(path,encoding='unicode')
    return sid,family,Path(path)

# Validated authored MINCO sources for arm-sensitive independent observations.
# 
# These are geometric planning witnesses, not optimized demonstrations or physics
# rollouts. The frozen-arm comparison keeps exactly the same base motion.

VALIDATION_DT=.025
WORLD_CLEARANCE=.05


def authored_reference(seed,variant,*,route_xy=None):
    rng=np.random.default_rng(seed)
    fold=np.array([0. if variant==0 else 1.5,-1.6,2.15,-.55])
    height=1.65+float(rng.uniform(-.005,.005))
    states=np.zeros((7,8));states[:,0]=[-.10,.25,.75,1.55,2.5,2.9,3.15]
    states[:,2]=-height
    states[:,1]=(.025 if variant%2 else -.025)*np.sin(np.linspace(0,np.pi,7))
    if route_xy is not None:
        route_xy=np.asarray(route_xy,dtype=float)
        if route_xy.shape!=(7,2) or not np.isfinite(route_xy).all() or np.any(np.diff(route_xy[:,0])<=0):
            raise ValueError('source route must be finite, forward-ordered (7,2) NED knots')
        states[:,:2]=route_xy
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
    route_element=ET.parse(sim.scene.model_path).getroot().find("custom/numeric[@name='portal_route_xy_ned']")
    route_xy=None if route_element is None else np.fromstring(route_element.get('data'),sep=' ').reshape(7,2)
    sources=[]
    for variant in range(2):
        trajectory=authored_reference(seed,variant,route_xy=route_xy)
        passed,clearance,violation,dt=evaluator.dense_validate(trajectory)
        metrics=dict(evaluator.last_validation_metrics)
        if not passed or metrics['minimum_world_distance']<WORLD_CLEARANCE:
            raise RuntimeError(f'coupled source rejected variant={variant}: {metrics}')
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
    opening_element=ET.parse(sim.scene.model_path).getroot().find("custom/numeric[@name='portal_openings']")
    openings=None if opening_element is None else np.fromstring(opening_element.get('data'),sep=' ').reshape(-1,9)
    # Observe vertical and lateral starting shapes across all movement phases.
    with IdealDepthObserver(sim.model,sim.data,spec=config.depth) as observer:
        allocations=[config.samples_per_scene//2,config.samples_per_scene-config.samples_per_scene//2]
        for variant,((trajectory,diagnostic),count) in enumerate(zip(sources,allocations)):
            timestamps=np.linspace(.35,trajectory.total_time-.35,count)
            if openings is not None:
                from scipy.optimize import brentq
                used=set()
                for opening in openings:
                    crossing=brentq(lambda t:trajectory.evaluate(t)[0]-opening[0],0.,trajectory.total_time)
                    candidates=[i for i in range(1,count-1) if i not in used]
                    if not candidates:raise ValueError('insufficient snapshots for every portal crossing')
                    index=min(candidates,key=lambda i:abs(timestamps[i]-crossing))
                    timestamps[index]=crossing;used.add(index)
                timestamps.sort()
            for timestamp in timestamps:
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
