"""Strict independent dataset configuration; no flight-controller presets."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
from uav_ac.planning.end2end.contracts import DepthSpec


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader,node,deep=False):
    values={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node,deep=deep)
        if key in values: raise ValueError(f'duplicate setting: {key}')
        values[key]=loader.construct_object(value_node,deep=deep)
    return values

_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_mapping)

@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 17
    scene_count: int = 12
    samples_per_scene: int = 32
    split_counts: tuple = (8,2,2)
    max_attempts_per_sample: int = 500
    gripper_gap: float = .06
    depth: DepthSpec = field(default_factory=DepthSpec)
    profile: str = 'stationary_v1'
    snapshot_min: float = .2
    snapshot_max: float = 1.2

    def __post_init__(self):
        for name in ('seed','scene_count','samples_per_scene','max_attempts_per_sample'):
            val=getattr(self,name)
            if isinstance(val,bool) or not isinstance(val,int) or val<(0 if name=='seed' else 1):
                raise ValueError(f'{name} must be a valid integer')
        if self.profile not in ('stationary_v1','rolling_v2','coupled_v3','portal_v4'):
            raise ValueError('unknown dataset profile')
        if not np.isfinite([self.snapshot_min,self.snapshot_max]).all() or not 0<self.snapshot_min<self.snapshot_max<3:
            raise ValueError('snapshot bounds must lie within the source trajectory')
        counts=tuple(self.split_counts)
        if len(counts)!=3 or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in counts) or sum(counts)!=self.scene_count:
            raise ValueError('split_counts must be positive train/validation/test counts summing to scene_count')
        if not np.isfinite(self.gripper_gap) or not .02<=self.gripper_gap<=.07:
            raise ValueError('invalid gripper gap')
        if self.profile=='coupled_v3' and (self.samples_per_scene<6 or any(v%2 for v in counts)):
            raise ValueError('coupled_v3 requires >=6 samples per scene and even scene counts in each split')
        if self.profile=='portal_v4' and (self.samples_per_scene<8 or any(v%3 for v in counts)):
            raise ValueError('portal_v4 requires >=8 samples and a multiple of 3 scenes in each split')
        object.__setattr__(self,'split_counts',counts)

    def as_mapping(self):
        values={name:getattr(self,name) for name in self.__dataclass_fields__}
        values['depth']=self.depth.as_mapping()
        values['split_counts']=list(self.split_counts)
        return values


def load_config(path):
    with Path(path).open(encoding='utf-8') as stream:values=yaml.load(stream,Loader=_UniqueLoader)
    if not isinstance(values,dict):raise ValueError('configuration must be a mapping')
    allowed=set(DatasetConfig.__dataclass_fields__)
    if unknown:=set(values)-allowed:raise ValueError(f'unknown settings: {sorted(unknown)}')
    values=dict(values)
    if 'depth' in values:
        depth=values['depth']
        if not isinstance(depth,dict):raise ValueError('depth must be a mapping')
        if set(depth)-set(DepthSpec.__dataclass_fields__):raise ValueError('unknown depth settings')
        values['depth']=DepthSpec(**depth)
    return DatasetConfig(**values)
