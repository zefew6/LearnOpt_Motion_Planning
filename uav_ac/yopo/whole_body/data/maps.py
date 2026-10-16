"""Convert official NWU point maps to a finite NED voxel-box scene."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from uav_ac.yopo.common.collection import PROJECT_ROOT
from uav_ac.yopo.common.depth import DepthSpec


def read_points(path):
    """Read scalar vertex properties from official PCL binary or ASCII PLY."""
    types={'float':'f4','float32':'f4','double':'f8','float64':'f8',
           'uchar':'u1','uint8':'u1','char':'i1','int8':'i1',
           'int':'i4','int32':'i4','uint':'u4','uint32':'u4',
           'short':'i2','int16':'i2','ushort':'u2','uint16':'u2'}
    with Path(path).open('rb') as stream:
        if stream.readline().strip()!=b'ply':raise ValueError('expected official PLY map')
        properties=[];count=None;fmt=None;vertex=False
        while True:
            line=stream.readline()
            if not line:raise ValueError('incomplete PLY header')
            fields=line.decode('ascii').split()
            if not fields:continue
            if fields[0]=='end_header':break
            if fields[0]=='format':fmt=fields[1]
            if fields[0]=='element':
                vertex=fields[1]=='vertex'
                if vertex:count=int(fields[2])
            if fields[0]=='property' and vertex:
                if fields[1] not in types:raise ValueError('unsupported PLY vertex property')
                properties.append((fields[2],types[fields[1]]))
        if not count or not {'x','y','z'}<={p[0] for p in properties}:raise ValueError('map has no XYZ vertices')
        if fmt=='ascii':
            rows=np.array([[float(v) for v in stream.readline().split()] for _ in range(count)])
            points=rows[:,[[p[0] for p in properties].index(k) for k in ('x','y','z')]]
        elif fmt in ('binary_little_endian','binary_big_endian'):
            endian='<' if fmt=='binary_little_endian' else '>'
            data=np.fromfile(stream,dtype=np.dtype([(n,endian+t) for n,t in properties]),count=count)
            if len(data)!=count:raise ValueError('incomplete PLY vertices')
            points=np.column_stack([data[k] for k in ('x','y','z')])
        else:raise ValueError('unsupported PLY format')
    if not np.isfinite(points).all():raise ValueError('nonfinite map vertices')
    return points.astype(float)


def voxel_boxes(points, resolution):
    """Merge occupied z-runs exactly, without filling gaps or convexifying trees.

    Return NED [center xyz, half-size xyz]. The z=0 ground samples are replaced
    by the scene's horizontal plane. Point surfaces become occupied voxel boxes;
    this does not reconstruct solid interiors from an incomplete point cloud.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or not np.isfinite(resolution) or resolution<=0:
        raise ValueError('expected finite XYZ points and positive resolution')
    cells=np.unique(np.floor(points[points[:,2]>1e-6]*[1.,-1.,-1.]/resolution+1e-7).astype(np.int64),axis=0)
    boxes=[];i=0
    while i<len(cells):
        j=i+1
        while j<len(cells) and np.array_equal(cells[j,:2],cells[i,:2]) and cells[j,2]==cells[j-1,2]+1:j+=1
        lower=cells[i]*resolution;upper=(cells[j-1]+1)*resolution
        boxes.append(np.r_[(lower+upper)/2,(upper-lower)/2]);i=j
    return np.asarray(boxes,dtype=float).reshape(-1,6)


def write_scene(directory, scene_id, points, config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    models=directory/'model';models.mkdir(exist_ok=True)
    robot=models/'aerial_manipulator.xml'
    if not robot.exists():shutil.copyfile(PROJECT_ROOT/'uav_ac/simulation/model/aerial_manipulator.xml',robot)
    root=ET.parse(PROJECT_ROOT/'uav_ac/simulation/models/yopo_bamboo.xml').getroot()
    root.set('model',scene_id);world=root.find('worldbody')
    for element in list(world):
        if element.tag=='site' or (element.tag=='geom' and element.get('name','').startswith('obstacle_')):world.remove(element)
    body=world.find('body');body.find('include').set('file','model/aerial_manipulator.xml')
    body.set('pos','0 0 2')
    # Per-frame starts/goals live in the observation shards; do not retain the
    # unrelated interactive mission endpoints from the robot/camera template.
    custom=root.find('custom')
    for name in ('whole_body_start','whole_body_goal'):
        custom.remove(custom.find(f"numeric[@name='{name}']"))
    camera=body.find('camera');spec=DepthSpec();k=spec.intrinsics
    camera.set('resolution',f'{spec.width} {spec.height}')
    camera.set('focalpixel',f'{k[0,0]} {k[1,1]}')
    root.find("custom/numeric[@name='planning_bounds']").set('data',' '.join(map(str,np.asarray(config.bounds).ravel())))
    root.find("custom/numeric[@name='whole_body_gripper']").set('data',str(config.gripper_gap))
    extent=np.max(np.abs(np.asarray(config.bounds)))+2
    root.find('statistic').set('extent',str(extent));root.find('statistic').set('center','0 0 2')
    world.find("geom[@name='ground']").set('pos','0 0 0')
    # Some official families (rooms) use [0,L] XY, others use [-L/2,L/2].
    # Center the finite map consistently and record this raw-map translation.
    offset=(points[:,:2].min(axis=0)+points[:,:2].max(axis=0))/2
    centered=points.copy();centered[:,:2]-=offset
    boxes=voxel_boxes(centered,config.voxel_resolution)
    for i,box in enumerate(boxes):
        ET.SubElement(world,'geom',name=f'obstacle_voxel_{i}',type='box',
                      pos=' '.join(map(str,box[:3]*[1.,-1.,-1.])),
                      size=' '.join(map(str,box[3:])),rgba='.35 .45 .3 1')
    path=directory/f'{scene_id}.xml';ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='unicode')
    return path,len(boxes),[-float(offset[0]),-float(offset[1]),0.]
