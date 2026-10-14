"""Shared calibrated optical depth contract; body mounts use FRD."""
from dataclasses import dataclass, field
import numpy as np

def finite_array(value, shape, name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    array.setflags(write=False)
    return array


def _camera_extrinsics():
    pose = np.eye(4)
    # Optical right/down/forward expressed in the FRD body frame.
    pose[:3,:3] = [[0,0,1],[1,0,0],[0,1,0]]
    pose[:3,3] = [.13,0,0]
    return pose


@dataclass(frozen=True)
class DepthSpec:
    height: int = 96
    width: int = 160
    hfov_deg: float = 90.
    vfov_deg: float = 60.
    near_m: float = .1
    far_m: float = 6.
    camera_extrinsics: np.ndarray = field(default_factory=_camera_extrinsics)

    def __post_init__(self):
        for name in ('height','width'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<2:
                raise ValueError(f'{name} must be an integer >= 2')
        for name in ('hfov_deg','vfov_deg'):
            value=getattr(self,name)
            if not np.isfinite(value) or not 0<value<180:
                raise ValueError(f'{name} must be between 0 and 180')
        if not np.isfinite([self.near_m,self.far_m]).all() or not 0<self.near_m<self.far_m:
            raise ValueError('depth range must be positive and ordered')
        pose=finite_array(self.camera_extrinsics,(4,4),'camera_extrinsics')
        if (not np.allclose(pose[3],[0,0,0,1]) or
            not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-8) or
            not np.isclose(np.linalg.det(pose[:3,:3]),1.)):
            raise ValueError('camera_extrinsics must be a rigid transform')
        object.__setattr__(self,'camera_extrinsics',pose)

    @property
    def intrinsics(self):
        return np.array([[self.width/(2*np.tan(np.deg2rad(self.hfov_deg)/2)),0,self.width/2],
                         [0,self.height/(2*np.tan(np.deg2rad(self.vfov_deg)/2)),self.height/2],
                         [0,0,1.]])

    def as_mapping(self):
        return {name:(getattr(self,name).tolist() if name=='camera_extrinsics' else getattr(self,name))
                for name in self.__dataclass_fields__}
