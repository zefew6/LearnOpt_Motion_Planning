"""Strict YAML and interactive orchestration for isolated pure-UAV YOPO."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class UavSettings:
    velocity: float=3.
    topk: int=1
    replan_dt: float=.1
    planning_timeout: float=1.
    max_plan_age: float=.5
    goal_tolerance: float=.5
    velocity_tolerance: float=.3
    settle_time: float=.5
    log_dt: float=1.
    threads: int=2
    def __post_init__(self):
        for name,value in asdict(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:raise ValueError(f'yopo_uav.{name} must be positive and finite')
        if not .5<=self.velocity<=6.:raise ValueError('yopo_uav.velocity must lie in [0.5,6]')
        if not isinstance(self.topk,int) or not 1<=self.topk<=15:raise ValueError('invalid yopo_uav.topk')
        if not isinstance(self.threads,int):raise ValueError('yopo_uav.threads must be an integer')


def validate_options(values):
    allowed=set(UavSettings.__dataclass_fields__)|{'checkpoint','device'}
    if not isinstance(values,dict) or set(values)-allowed:raise ValueError('unknown yopo_uav options')
    return UavSettings(**{k:v for k,v in values.items() if k in UavSettings.__dataclass_fields__})


def validate_config(config,path):
    values=config['yopo_uav'];settings=validate_options(values)
    root=ET.parse(config['scene']).getroot();include=root.find('worldbody/body/include')
    if include is None or Path(include.get('file','')).name!='quadrotor.xml':raise ValueError('yopo_navigation requires a pure quadrotor XML')
    if root.find('worldbody/body/camera[@name="yopo_depth"]') is None or root.find('worldbody/site[@name="goal"]') is None:raise ValueError('UAV XML requires yopo_depth camera and goal site')
    dt=float(root.find('option').get('timestep','.002'))
    for value,base,name in ((config['control_dt'],dt,'control_dt'),(settings.replan_dt,config['control_dt'],'replan_dt')):
        ratio=value/base
        if ratio<1 or not np.isclose(ratio,round(ratio)):raise ValueError(f'{name} must be an integer multiple of its inner period')
    checkpoint=values.get('checkpoint')
    if not isinstance(checkpoint,str) or not checkpoint:raise ValueError('yopo_uav.checkpoint is required')
    checkpoint=(Path(path).parent/checkpoint).resolve()
    try:info=json.loads((checkpoint/'metadata.json').read_text())
    except (OSError,ValueError) as error:raise ValueError('missing/invalid UAV checkpoint metadata') from error
    if info.get('profile')!='uav_official_minco_v1' or info.get('upstream_commit')!='d5e1e60f83cc991602f142d9b6b96204a24d6b9f' or not (checkpoint/'weights.pth').is_file():raise ValueError('incompatible UAV checkpoint')
    device=values.get('device','cpu')
    if device!='cpu' and not str(device).startswith('cuda'):raise ValueError('invalid yopo_uav.device')
    config['yopo_uav']={**asdict(settings),'checkpoint':str(checkpoint),'device':device}


def run_yopo_navigation(config):
    import torch
    import time
    from uav_ac.control import CascadedController,CascadedConfig
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    from uav_ac.simulation.depth_observation import IdealDepthObserver
    from uav_ac.yopo.uav.deployment import UavPlanner
    from uav_ac.tasks.yopo_navigation import UavNavigation
    values=config['yopo_uav'];settings=validate_options(values);torch.set_num_threads(settings.threads)
    sim=MujocoSimulation(config['scene'],planning_path_capacity=6 if config['visualize'] else 0,record_actual_trajectory=config['visualize'])
    planner=UavPlanner(values['checkpoint'],device=values['device'],velocity=settings.velocity,topk=settings.topk)
    planner.warm_up()
    observer=IdealDepthObserver(sim.model,sim.data,spec=planner.depth_spec,exclude_robot=True)
    CascadedConfig(**config['cascaded']).apply_to(sim.quad)
    controller=CascadedController(sim.quad.g,sim.quad.dt);initial=sim.quad.X.copy();runtime={}
    def reset():
        if runtime.get('task') is not None:runtime['task'].close()
        hover=np.full(4,np.sqrt(sim.quad.m*sim.quad.g/(4*sim.quad.kf)))
        sim.reset(initial,hover);controller.reset();planner.reset()
        runtime['task']=UavNavigation(sim,controller,planner,observer,settings,config['control_dt'],config['duration'],visualize=config['visualize'])
    def prepare(sync):reset();sync();return True
    def step():runtime['task'].step()
    try:
        if config['visualize']:
            sim.run_interactive(step,reset,chase_camera=config['follow_camera'],prepare=prepare)
            if runtime.get('task') is None:return {'state':'FAILED','failure_reason':'viewer_closed_before_start','success':False}
            if not runtime['task'].done:runtime['task'].finish('FAILED','viewer_closed')
        else:
            reset();wall_start=time.perf_counter()
            while not runtime['task'].done:
                step();sim.step()
                # Online inference and physics share real time, including without a viewer.
                delay=sim.time-(time.perf_counter()-wall_start)
                if delay>0:time.sleep(delay)
        result=runtime['task'].result();print(f'UAV YOPO result: {result}',flush=True);return result
    finally:
        if runtime.get('task') is not None:runtime['task'].close()
        observer.close()
