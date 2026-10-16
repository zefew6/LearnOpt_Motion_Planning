"""One whole-body sampling contract over the official random-map generator."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml

from uav_ac.yopo.common.collection import PROJECT_ROOT
from uav_ac.yopo.common.config import _UniqueLoader


@dataclass(frozen=True)
class CollectionConfig:
    map_config: str = str(PROJECT_ROOT/'configs/yopo/dataset/yopo_uav.yaml')
    map_options: dict = field(default_factory=dict)
    seed: int = 47
    split_counts: tuple = (1, 1, 1)
    samples_per_scene: int = 8
    max_attempts_per_sample: int = 300
    voxel_resolution: float = .15
    gripper_gap: float = .06
    bounds: tuple = ((-4., -4., -3.2), (4., 4., -1.2))
    clearance: float = .05
    self_clearance: float = .015

    def __post_init__(self):
        for name in ('seed', 'samples_per_scene', 'max_attempts_per_sample'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value < (0 if name=='seed' else 1):
                raise ValueError(f'invalid {name}')
        counts=tuple(self.split_counts)
        if len(counts)!=3 or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in counts):
            raise ValueError('split_counts needs positive train/validation/test map counts')
        bounds=np.asarray(self.bounds,dtype=float)
        if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]) or bounds[1,2]>=0:
            raise ValueError('bounds must be ordered finite NED bounds above ground')
        for name in ('voxel_resolution','clearance','self_clearance','gripper_gap'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(f'invalid {name}')
        if not .02<=self.gripper_gap<=.07:raise ValueError('invalid gripper_gap')
        object.__setattr__(self,'split_counts',counts)
        self.official_settings()

    def official_settings(self):
        with Path(self.map_config).open() as stream:values=yaml.load(stream,Loader=_UniqueLoader)
        if not isinstance(values,dict):raise ValueError('official map config must be a mapping')
        if not isinstance(self.map_options,dict) or set(self.map_options)-set(values):
            raise ValueError('unknown map_options')
        if set(self.map_options)&{'save_path','tree_file','env_num','image_num','seed'}:
            raise ValueError('map_options cannot override collection ownership')
        values.update(self.map_options,seed=self.seed,env_num=sum(self.split_counts),image_num=self.samples_per_scene)
        for key in ('x_length','y_length','z_length','room_number'):
            value=values[key]
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f'invalid official map {key}')
        for key in ('resolution','ply_res','tree_dist','road_width','wall_thick'):
            value=values[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(f'invalid official map {key}')
        if isinstance(values['maze_type'],bool) or values['maze_type'] not in range(1,8):
            raise ValueError('invalid official map maze_type')
        if values['maze_type']==6:
            count=values['max_windows']
            if isinstance(count,bool) or not isinstance(count,int) or count<1:
                raise ValueError('room max_windows must be positive')
            lo,hi=values['window_size_min'],values['window_size_max']
            if not np.isfinite([lo,hi]).all() or not 0<lo<=hi:
                raise ValueError('invalid room window sizes')
        bounds=np.asarray(self.bounds)
        extent=np.array([values['x_length'],values['y_length']])/2
        if np.any(bounds[0,:2]<=-extent) or np.any(bounds[1,:2]>=extent) or -bounds[0,2]>=values['z_length']:
            raise ValueError('bounds must lie inside the official map extent')
        return values

    def as_mapping(self):return asdict(self)


def load_config(path):
    path=Path(path).resolve()
    with path.open() as stream:values=yaml.load(stream,Loader=_UniqueLoader)
    if not isinstance(values,dict) or set(values)-set(CollectionConfig.__dataclass_fields__):
        raise ValueError('unknown whole-body collection settings')
    if 'map_config' in values:values['map_config']=str((path.parent/values['map_config']).resolve())
    return CollectionConfig(**values)
