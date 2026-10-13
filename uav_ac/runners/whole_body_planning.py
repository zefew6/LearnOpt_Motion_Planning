"""Live-viewer task orchestration; physics/controller keep running while planning."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from uav_ac.control import CascadedController,CascadedConfig
from uav_ac.control.aerial_manipulator_controller import AerialManipulatorController
from uav_ac.robot.aerial_manipulator import AerialManipulator
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.planning.end2end.planner import End2EndPlanner
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.tasks.whole_body_reaching import WholeBodyReaching
from .whole_body_config import WholeBodySettings,benchmark_endpoints


def run_whole_body_planning(config):
    simulation=MujocoSimulation(config['scene'],record_actual_trajectory=config['visualize'],
                                planning_path_capacity=1 if config['visualize'] else 0)
    if not isinstance(simulation.robot,AerialManipulator):raise ValueError('whole-body task requires an aerial manipulator XML')
    start,goal,gap,_,_=benchmark_endpoints(config['scene'])
    limits=simulation.robot.limits
    for state in (start,goal):
        if np.any(state[4:]<limits.joint_lower) or np.any(state[4:]>limits.joint_upper):raise ValueError('XML joint endpoint outside robot limits')
    settings=WholeBodySettings.from_mapping(config['whole_body'])
    torch.set_num_threads(settings.threads)
    physical=Path(__file__).resolve().parents[1]/'simulation/model/aerial_manipulator.xml'
    planner=End2EndPlanner(config['end2end']['checkpoint'],device=config['end2end']['device'],
                          expected_robot_hash=hashlib.sha256(physical.read_bytes()).hexdigest(),
                          gripper_tolerance=config['end2end']['gripper_tolerance'])
    CascadedConfig(**config['cascaded']).apply_to(simulation.quad)
    controller=AerialManipulatorController(CascadedController(simulation.quad.g,simulation.quad.dt),simulation.robot,simulation.quad)
    runtime={}
    def initialize():
        if runtime.get('execution') is not None:runtime['execution'].close()
        controller.reset()
        q,v=recover_state(np.vstack((start,np.zeros((3,8)))),gap)
        simulation.robot.reset(q.copy(),v.copy())
        if 'observer' not in runtime:
            runtime['observer']=IdealDepthObserver(simulation.model,simulation.data,spec=planner.model.depth_spec)
        def on_plan(result,age):
            if config['visualize']:
                trajectory=result.trajectory
                times=np.linspace(age,trajectory.total_time,65)
                simulation.set_planning_paths([trajectory.evaluate(times)[:,:3]],[(.1,.8,.25,.9)])
        runtime['execution']=WholeBodyReaching(simulation,controller,planner,runtime['observer'],goal,gap,settings,
            config['control_dt'],config['duration'],on_plan=on_plan)
    def prepare(sync_viewer):initialize();sync_viewer();return True
    def step():runtime['execution'].step()
    try:
        if config['visualize']:
            simulation.run_interactive(step,initialize,chase_camera=config['follow_camera'],prepare=prepare)
            execution=runtime.get('execution')
            if execution is None:return {'state':'FAILED','success':False,'failure_reason':'viewer_closed_before_start'}
            if not execution.done:execution.abort('viewer_closed')
        else:
            initialize();execution=runtime['execution']
            while not execution.done:
                execution.step();simulation.step()
        result=execution.result()
        print(f'Whole-body result: {result["state"]} success={result["success"]} reason={result["failure_reason"]}',flush=True)
        return result
    finally:
        if runtime.get('execution') is not None:runtime['execution'].close()
        if runtime.get('observer') is not None:runtime['observer'].close()
