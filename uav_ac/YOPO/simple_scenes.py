"""Generate small guidance/acceptance scenes from the single YOPO benchmark.

Scene instances are artifacts written to caller-owned directories under runs;
there are no separate static training/debug XML templates.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

MODEL_ROOT=Path(__file__).resolve().parents[1]/'simulation'
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
    root=ET.parse(Path(model_root)/'models/whole_body_dense_outdoor.xml').getroot()
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
