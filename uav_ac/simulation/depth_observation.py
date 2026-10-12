"""Ideal first-surface optical-z depth; no mapping or ESDF dependencies."""
from copy import copy
import mujoco
import numpy as np
from uav_ac.planning.end2end.contracts import DepthSpec


class IdealDepthObserver:
    def __init__(self,model,data,*,camera_name='yopo_depth',spec=None):
        self.model,self.data=model,data;self.spec=DepthSpec() if spec is None else spec
        self.camera_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera_name)
        if self.camera_id<0: raise ValueError(f'missing depth camera: {camera_name}')
        self.intrinsics=self.spec.intrinsics
        self.far_clip=float(model.stat.extent*model.vis.map.zfar)
        near_clip=float(model.stat.extent*model.vis.map.znear)
        if near_clip>self.spec.near_m or self.far_clip<self.spec.far_m:
            raise ValueError('render clipping planes do not cover DepthSpec range')
        cid=self.camera_id
        if np.all(model.cam_sensorsize[cid]>0):
            sensor=model.cam_sensorsize[cid];intr=model.cam_intrinsic[cid]
            focal=intr[:2]*np.array([self.spec.width,self.spec.height])/sensor
        else:
            fy=self.spec.height/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
            focal=np.array([fy,fy])
        if np.any(abs(model.cam_intrinsic[cid,2:4])>1e-8):
            raise ValueError('camera principal point must match centered DepthSpec')
        quad=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'quadrotor')
        if quad>=0:
            if model.cam_bodyid[cid]!=quad:
                raise ValueError('depth camera mount must be attached to quadrotor')
            matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,model.cam_quat[cid])
            flip=np.diag([1.,-1.,-1.]);mount=np.eye(4)
            mount[:3,:3]=flip@matrix.reshape(3,3)@flip
            mount[:3,3]=flip@model.cam_pos[cid]
            if not np.allclose(mount,self.spec.camera_extrinsics,atol=1e-7):
                raise ValueError('camera mount does not match declared optical extrinsics')
        if not np.allclose(focal,[self.intrinsics[0,0],self.intrinsics[1,1]],rtol=1e-5):
            raise ValueError('camera intrinsics do not match DepthSpec')
        self.options=mujoco.MjvOption();self.options.sitegroup[:]=0
        self.options.geomgroup[2]=0;self.options.geomgroup[5]=0
        self.debug_ids={i for i in range(model.ngeom) if any(
            (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(prefix)
            for prefix in ('planning_path_','trajectory_segment','actual_trajectory_segment',
                           'pick_target','place_target','payload_marker','drone_tracking_beacon'))}
        # Render-only model copy: group filtering also works in depth ID rendering.
        # The live physics model and the physical robot geometry stay unchanged.
        self._render_model=copy(model)
        if self.debug_ids:self._render_model.geom_group[list(self.debug_ids)]=5
        self._render_data=mujoco.MjData(self._render_model)
        self.renderer=mujoco.Renderer(self._render_model,height=self.spec.height,width=self.spec.width)
        self.renderer.enable_depth_rendering()

    def observe(self):
        if self.renderer is None: raise RuntimeError('observer is closed')
        mujoco.mj_copyData(self._render_data,self._render_model,self.data)
        self.renderer.update_scene(self._render_data,camera=self.camera_id,scene_option=self.options)
        raw=self.renderer.render().copy()
        valid=np.isfinite(raw)&(raw>0)&(raw<self.spec.far_m)&(raw<self.far_clip*(1.-1e-5))
        depth=np.where(valid,np.maximum(raw,self.spec.near_m),self.spec.far_m).astype(np.float32)
        return depth,valid

    def close(self):
        if self.renderer is not None: self.renderer.close();self.renderer=None
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
