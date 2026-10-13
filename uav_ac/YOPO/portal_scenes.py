"""Finite-depth wall openings assembled from exact physical boxes in closed rooms."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .coupled_scenes import write_coupled_scene


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
