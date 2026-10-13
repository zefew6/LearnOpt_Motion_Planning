"""Motion-only corridor/bamboo XML instances with arm-sensitive lower obstacles."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from .scene_sampling import MODEL_ROOT
from uav_ac.planning.end2end.contracts import DepthSpec


def write_coupled_scene(directory,index,seed,spec):
    if not np.allclose(spec.camera_extrinsics,DepthSpec().camera_extrinsics):
        raise ValueError('coupled scenes retain the authored robot camera mount')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    models=directory/'model';models.mkdir(exist_ok=True)
    if not (models/'aerial_manipulator.xml').exists():
        shutil.copyfile(MODEL_ROOT/'model'/'aerial_manipulator.xml',models/'aerial_manipulator.xml')
    root=ET.parse(MODEL_ROOT/'models'/'yopo_free.xml').getroot()
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
