"""Instantiate the official network without global config or primitive sharing."""
import numpy as np
import torch
from uav_ac.yopo.vendor.yopo_minco.config.config import Config,configuration
from uav_ac.yopo.vendor.yopo_minco.policy.yopo_network import YopoNetwork

UPSTREAM_COMMIT='d5e1e60f83cc991602f142d9b6b96204a24d6b9f'
PROFILE='uav_official_minco_v1'


class OfficialCore:
    def __init__(self,*,device='cpu',velocity=6.):
        if isinstance(velocity,bool) or not isinstance(velocity,(int,float)) or not np.isfinite(velocity) or not .5<=velocity<=6.:
            raise ValueError('official UAV velocity must lie in [0.5,6] m/s')
        if str(device)!='cpu' and not str(device).startswith('cuda'):raise ValueError('invalid official UAV device')
        if str(device).startswith('cuda') and not torch.cuda.is_available():raise ValueError('requested CUDA is unavailable')
        self.config=Config(device=device,velocity=float(velocity));self.device=torch.device(device)
        with configuration(self.config):self.network=YopoNetwork().to(self.device).eval()
    def inference(self,depth,state):
        if depth.shape[1:]!=(1,96,160) or state.shape!=(len(depth),9):raise ValueError('official UAV requires 1x96x160 depth and9 state features')
        if not torch.isfinite(depth).all() or not torch.isfinite(state).all():raise ValueError('nonfinite official UAV input')
        return self.network.inference(depth.to(self.device),state.to(self.device).clone())
