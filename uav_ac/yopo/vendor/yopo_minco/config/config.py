"""Scoped config adapter; retains the pinned upstream derived parameters."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import copy
import torch
import yaml


class Config:
    def __init__(self,*,device='cpu',velocity=6.,train=False):
        self._data=yaml.safe_load(Path(__file__).with_name('traj_opt.yaml').read_text())
        self._data.update(train=train,velocity=velocity,piece_num=2)
        self._data['goal_length']=self._data['tail_radio_max']
        self._data['sgm_time']=self._data['tail_radio_max']/self._data['vel_max_train']
        self._data['traj_num']=self._data['horizon_num']*self._data['vertical_num']
        self._data['piece_duration']=self._data['sgm_time']/self._data['piece_num']
        self._data['image_height']=self._data['vertical_num']*self._data['downsample_factor']
        self._data['image_width']=self._data['horizon_num']*self._data['downsample_factor']
        self.device=torch.device(device);self._primitive=None
    def __getitem__(self,key):return self._data[key]
    def __setitem__(self,key,value):self._data[key]=value
    def snapshot(self):return copy.deepcopy(self._data)
    def primitive(self,factory):
        if self._primitive is None:self._primitive=factory()
        return self._primitive


_active=ContextVar('official_yopo_config',default=None)


class _Proxy:
    def current(self):
        result=_active.get()
        if result is None:raise RuntimeError('official YOPO construction requires a scoped config')
        return result
    def __getitem__(self,key):return self.current()[key]
    def __setitem__(self,key,value):self.current()[key]=value
    @property
    def device(self):return self.current().device
    def primitive(self,factory):return self.current().primitive(factory)


cfg=_Proxy()


@contextmanager
def configuration(value):
    token=_active.set(value)
    try:yield
    finally:_active.reset(token)
