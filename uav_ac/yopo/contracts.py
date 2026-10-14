"""Map-free observations; all positions use NED and body quantities use FRD."""
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


@dataclass(frozen=True)
class PlannerObservation:
    depth_m: np.ndarray
    depth_valid: np.ndarray
    configuration: np.ndarray
    velocity: np.ndarray
    start_pva: np.ndarray
    goal: np.ndarray
    depth_spec: DepthSpec = field(default_factory=DepthSpec)
    timestamp: float = 0.
    start_jerk: np.ndarray = field(default_factory=lambda: np.zeros(8))
    derivative_source: str = 'stationary'

    def __post_init__(self):
        shape=(self.depth_spec.height,self.depth_spec.width)
        for name,dim in [('depth_m',shape),('configuration',(12,)),('velocity',(11,)),
                         ('start_pva',(3,8)),('goal',(8,)),('start_jerk',(8,))]:
            object.__setattr__(self,name,finite_array(getattr(self,name),dim,name))
        from numbers import Real
        if not isinstance(self.timestamp,Real) or isinstance(self.timestamp,(bool,np.bool_)) or not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('timestamp must be a finite nonnegative scalar')
        object.__setattr__(self,'timestamp',float(self.timestamp))
        if self.derivative_source not in ('stationary', 'reference', 'simulation', 'imu', 'augmentation'):
            raise ValueError('unknown derivative source')
        mask=np.array(self.depth_valid,copy=True)
        if mask.shape!=shape or mask.dtype!=np.bool_:
            raise ValueError('depth_valid must be a boolean depth-sized mask')
        mask.setflags(write=False);object.__setattr__(self,'depth_valid',mask)
        spec=self.depth_spec
        tolerance=2*np.finfo(np.float32).eps*max(1.,spec.far_m)
        if np.any(self.depth_m<spec.near_m-tolerance) or np.any(self.depth_m>spec.far_m+tolerance):
            raise ValueError('depth must be clipped to sensor range')
        if not np.allclose(self.depth_m[~mask],spec.far_m,rtol=0,atol=tolerance):
            raise ValueError('invalid depth must equal far_m')
        canonical=np.clip(self.depth_m,spec.near_m,spec.far_m)
        canonical[~mask]=spec.far_m;canonical.setflags(write=False)
        object.__setattr__(self,'depth_m',canonical)
        q=self.configuration
        if not np.isclose(np.linalg.norm(q[3:7]),1.,atol=1e-7,rtol=0):
            raise ValueError('configuration quaternion must be unit length')
        if not .020<=q[11]<=.070:
            raise ValueError('gripper opening outside model limits')
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(q[[4,5,6,3]]).as_matrix()
        # Flatness heading is determined by body-y, not Euler yaw under tilt.
        yaw=np.arctan2(-rotation[0,1],rotation[1,1])
        if (not np.allclose(self.start_pva[0,:3],q[:3],atol=1e-7) or
            not np.allclose(self.start_pva[0,4:],q[7:11],atol=1e-7) or
            abs(np.arctan2(np.sin(self.start_pva[0,3]-yaw),np.cos(self.start_pva[0,3]-yaw)))>1e-7 or
            not np.allclose(self.start_pva[1,:3],self.velocity[:3],atol=1e-7) or
            not np.allclose(self.start_pva[1,4:],self.velocity[6:10],atol=1e-7)):
            raise ValueError('start_pva does not match configuration/velocity')

    def validate_joints(self, lower, upper):
        for name,values in [('start',self.configuration[7:11]),('goal',self.goal[4:])]:
            if np.any(values<lower) or np.any(values>upper):
                raise ValueError(f'{name} joint configuration outside limits')
