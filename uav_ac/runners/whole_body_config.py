"""Settings and XML-owned endpoints for online whole-body reaching."""
from dataclasses import dataclass,asdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class WholeBodySettings:
    replan_dt: float = .2
    planning_timeout: float = 1.
    max_plan_age: float = .5
    position_tolerance: float = .10
    yaw_tolerance: float = .10
    joint_tolerance: float = .10
    velocity_tolerance: float = .10
    body_rate_tolerance: float = .20
    joint_rate_tolerance: float = .10
    settle_time: float = .5
    tracking_error_limit: float = 2.
    stall_time: float = 12.
    progress_distance: float = .05
    log_dt: float = 1.
    threads: int = 2
    enforce_height_ceiling: bool = True

    def __post_init__(self):
        if not isinstance(self.enforce_height_ceiling,bool):
            raise ValueError('whole_body.enforce_height_ceiling must be a boolean')
        for name,value in asdict(self).items():
            if name=='enforce_height_ceiling':continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(f'whole_body.{name} must be positive and finite')
        if not isinstance(self.threads,int):raise ValueError('whole_body.threads must be an integer')

    @classmethod
    def from_mapping(cls,values):
        if not isinstance(values,dict) or set(values)-set(cls.__dataclass_fields__):
            raise ValueError('unknown whole_body options')
        return cls(**values)


def benchmark_endpoints(path):
    root=ET.parse(path).getroot();custom=root.find('custom')
    def numeric(name,size):
        node=None if custom is None else custom.find(f"numeric[@name='{name}']")
        data=np.fromstring('' if node is None else node.get('data',''),sep=' ')
        if data.shape!=(size,) or not np.isfinite(data).all():raise ValueError(f'XML requires {name} ({size})')
        return data
    start=numeric('whole_body_start',8);goal=numeric('whole_body_goal',8);gap=float(numeric('whole_body_gripper',1)[0])
    bounds=numeric('planning_bounds',6).reshape(2,3)
    for state in (start,goal):
        if np.any(state[:3]<bounds[0]) or np.any(state[:3]>bounds[1]):raise ValueError('whole-body endpoint outside XML bounds')
    if not .02<=gap<=.07:raise ValueError('invalid XML whole-body gripper opening')
    timestep=float(root.find('option').get('timestep','.002'))
    return start,goal,gap,bounds,timestep


def validate_whole_body(config,path):
    settings=WholeBodySettings.from_mapping(config['whole_body'])
    _,_,_,_,physics_dt=benchmark_endpoints(config['scene'])
    for value,base,label in ((config['control_dt'],physics_dt,'control_dt'),(settings.replan_dt,config['control_dt'],'replan_dt')):
        ratio=value/base
        if ratio<1 or not np.isclose(ratio,round(ratio)):raise ValueError(f'{label} must be an integer multiple of its inner period')
    model=config['yopo']
    if not isinstance(model,dict) or set(model)-{'checkpoint','device','gripper_tolerance','speed_scale'}:raise ValueError('unknown yopo options')
    if not isinstance(model.get('checkpoint'),str) or not model['checkpoint']:raise ValueError('yopo.checkpoint is required')
    checkpoint=(Path(path).parent/model['checkpoint']).resolve()
    if not all((checkpoint/name).is_file() for name in ('metadata.json','weights.pt')):raise ValueError('missing yopo checkpoint files')
    if model.get('device','cpu')!='cpu' and not str(model['device']).startswith('cuda'):raise ValueError('invalid yopo.device')
    model['checkpoint']=str(checkpoint);model.setdefault('device','cpu')
    tolerance=model.setdefault('gripper_tolerance',.0005)
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not np.isfinite(tolerance) or not 0<=tolerance<=.005:
        raise ValueError('yopo.gripper_tolerance must lie in [0,0.005] m')
    scale=model.setdefault('speed_scale',1.)
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not np.isfinite(scale) or not .5<=scale<=2.:
        raise ValueError('yopo.speed_scale must lie in [0.5,2]')
    config['whole_body']=asdict(settings)
