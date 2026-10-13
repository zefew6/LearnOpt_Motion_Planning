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
from uav_ac.planning.end2end.direct import decode_direct
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.tasks.whole_body_reaching import WholeBodyReaching
from .whole_body_config import WholeBodySettings,benchmark_endpoints


def run_whole_body_planning(config):
    simulation=MujocoSimulation(config['scene'],record_actual_trajectory=config['visualize'],
                                planning_path_capacity=8 if config['visualize'] else 0)
    if not isinstance(simulation.robot,AerialManipulator):raise ValueError('whole-body task requires an aerial manipulator XML')
    start,goal,gap,_,_=benchmark_endpoints(config['scene'])
    limits=simulation.robot.limits
    for state in (start,goal):
        if np.any(state[4:]<limits.joint_lower) or np.any(state[4:]>limits.joint_upper):raise ValueError('XML joint endpoint outside robot limits')
    settings=WholeBodySettings.from_mapping(config['whole_body'])
    torch.set_num_threads(settings.threads)
    physical=Path(__file__).resolve().parents[1]/'simulation/model/aerial_manipulator.xml'
    planner=End2EndPlanner(config['yopo']['checkpoint'],device=config['yopo']['device'],
                          expected_robot_hash=hashlib.sha256(physical.read_bytes()).hexdigest(),
                          gripper_tolerance=config['yopo']['gripper_tolerance'],speed_scale=config['yopo']['speed_scale'])
    CascadedConfig(**config['cascaded']).apply_to(simulation.quad)
    controller=AerialManipulatorController(CascadedController(simulation.quad.g,simulation.quad.dt),simulation.robot,simulation.quad)
    runtime={}
    def initialize():
        if runtime.get('execution') is not None:runtime['execution'].close()
        runtime.pop('previous_path',None);runtime.pop('next_plan_log',None)
        controller.reset()
        q,v=recover_state(np.vstack((start,np.zeros((3,8)))),gap)
        simulation.robot.reset(q.copy(),v.copy())
        if 'observer' not in runtime:
            runtime['observer']=IdealDepthObserver(simulation.model,simulation.data,spec=planner.model.depth_spec)
        def on_plan(result,age):
            if config['visualize']:
                trajectory=result.trajectory
                times=np.linspace(age,trajectory.total_time,65)
                paths=[trajectory.evaluate(times)[:,:3]];colors=[(.1,.95,.25,1.)]
                # Retain spatial and arm-mode diversity, not only adjacent ranks.
                candidates=result.parameters
                descriptor=np.c_[candidates[:,:3],.3*candidates[:,4:8]]
                selected=[result.selected_index]
                for _ in range(min(5,len(candidates)-1)):
                    distances=np.linalg.norm(descriptor[:,None]-descriptor[selected][None],axis=2).min(axis=1)
                    distances[selected]=-1.;index=int(np.argmax(distances));selected.append(index)
                    alternative=decode_direct(candidates[index],result.observation)
                    local=np.linspace(age,alternative.total_time,33)
                    if age>=alternative.total_time:continue
                    paths.append(alternative.evaluate(local)[:,:3]);colors.append((.8,.8,.9,.28))
                tip=[]
                for t in times[::4]:
                    reference=trajectory.reference(t,simulation.robot,gripper_opening=gap)
                    position,_=simulation.robot.forward_kinematics(reference.configuration,frame='grasp',check_limits=False)
                    tip.append(position)
                paths.append(np.asarray(tip));colors.append((1.,.55,.08,.9))
                if 'previous_path' in runtime:
                    paths.append(runtime['previous_path']);colors.append((.65,.2,.9,.5))
                simulation.set_planning_paths(paths,colors)
                runtime['previous_path']=paths[0].copy()
                if simulation.time>=runtime.get('next_plan_log',0.):
                    print(f'YOPO candidates: selected={result.selected_index}/{result.candidate_count} horizon={trajectory.total_time:.2f}s inner_xyz_std={candidates[:,:3].std(0).round(3)}',flush=True)
                    runtime['next_plan_log']=simulation.time+settings.log_dt
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
