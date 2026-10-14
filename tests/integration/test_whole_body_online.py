from tests.yopo_scenes import scene_path
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.control import CascadedController
from uav_ac.control.aerial_manipulator_controller import AerialManipulatorController
from uav_ac.yopo.whole_body.contracts import DepthSpec
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.runners.whole_body_config import WholeBodySettings
from uav_ac.tasks.whole_body_reaching import WholeBodyReaching


class ImmediateExecutor:
    def submit(self,fn,*args):
        future=Future()
        try:future.set_result(fn(*args))
        except Exception as error:future.set_exception(error)
        return future
    def shutdown(self,**kwargs):pass


class Observer:
    spec=DepthSpec()
    def observe(self):return np.full((96,160),6.),np.zeros((96,160),dtype=bool)


class Planner:
    def __init__(self):self.observations=[]
    def plan(self,o):
        self.observations.append(o)
        tail=np.vstack((o.goal,np.zeros((2,8))))
        coeff,_=MINCOQuintic(o.start_pva,tail,1).solve(np.empty((0,8)),np.array([2.]))
        return SimpleNamespace(trajectory=AerialManipulatorTrajectory(np.array([2.]),coeff.reshape(1,6,8),np.empty((0,8)),0.,0,False,'test'))


def setup(goal_delta):
    sim=MujocoSimulation(scene_path('free'),planning_path_capacity=0,record_actual_trajectory=False)
    start=np.array([0,0,-1.65,0,0,0,0,0.]);q,v=recover_state(np.vstack((start,np.zeros((3,8)))),.06)
    sim.robot.reset(q.copy(),v.copy());goal=start.copy();goal[0]+=goal_delta
    ctrl=AerialManipulatorController(CascadedController(sim.quad.g,sim.quad.dt),sim.robot,sim.quad)
    planner=Planner();settings=WholeBodySettings(replan_dt=.05,settle_time=.05,stall_time=10.)
    task=WholeBodyReaching(sim,ctrl,planner,Observer(),goal,.06,settings,.01,2.,executor=ImmediateExecutor())
    return sim,task,planner


def test_replanning_observes_actual_controller_driven_motion_without_resets():
    sim,task,planner=setup(.5)
    initial=sim.robot.configuration.copy()
    # Any reset during online motion would be trajectory playback, not physical tracking.
    sim.robot.reset=lambda *a,**k:(_ for _ in ()).throw(AssertionError('online robot reset forbidden'))
    for _ in range(800):
        before=sim.robot.configuration.copy();task.step()
        assert np.array_equal(before,sim.robot.configuration)
        sim.step()
    assert task.accepted_count>2 and np.linalg.norm(sim.robot.configuration[:3]-initial[:3])>.001
    assert planner.observations[1].timestamp>planner.observations[0].timestamp
    assert not np.array_equal(planner.observations[-1].configuration,planner.observations[0].configuration)
    assert np.allclose(planner.observations[-1].start_pva[1,:3],planner.observations[-1].velocity[:3])
    assert not sim.collision_detected
    task.close()


def test_goal_settling_uses_actual_velocity_and_does_not_request_more_plans():
    sim,task,planner=setup(0.)
    for _ in range(200):
        task.step();sim.step()
        if task.done:break
    assert task.state=='REACHED' and task.accepted_count==0 and not planner.observations
    task.close()


def test_worker_close_does_not_wait_for_overdue_inference():
    from threading import Event
    from uav_ac.tasks.whole_body_reaching import PlannerWorker
    entered=Event();release=Event();worker=PlannerWorker()
    def blocked():entered.set();release.wait();return 1
    future=worker.submit(blocked)
    assert entered.wait(1.)
    worker.shutdown(wait=False)
    assert not future.done()
    release.set();assert future.result(timeout=1.)==1


def test_pending_inference_keeps_controller_running_and_reports_timeout(capsys):
    sim,task,_=setup(.5)
    pending=Future();pending.set_running_or_notify_cancel()
    task.future=pending;task.submit_wall=0.
    task.step()
    assert task.state=='FAILED' and task.failure_reason=='planning_timeout'
    assert 'state=FAILED reason=planning_timeout' in capsys.readouterr().out
    task.close()


def test_bamboo_benchmark_has_a_collision_free_whole_body_witness():
    from dataclasses import replace
    from uav_ac.yopo.whole_body.training.oracle import PrivilegedEvaluator
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
    sim=MujocoSimulation('uav_ac/simulation/models/yopo_bamboo.xml',planning_path_capacity=0,record_actual_trajectory=False)
    states=np.zeros((11,8))
    states[:,0]=[.1,1.3,4,6.5,9,11.5,14,16.5,19,21.5,23]
    states[:,1]=[0,.35,1.1,0,-1.1,0,1.1,0,-1.1,-.5,0];states[:,2]=-1.9
    states[1:-1,4:]=[0,-1.6,1.9,-.3];states[-1,4:]=[0,.15,-.2,.1]
    from uav_ac.runners.whole_body_config import benchmark_endpoints
    start,goal,_,_,_=benchmark_endpoints('uav_ac/simulation/models/yopo_bamboo.xml')
    states[0]=start;states[-1]=goal
    durations=np.full(10,4.)
    coefficients=[]
    for i,dt in enumerate(durations):
        head=np.vstack((states[i],np.zeros((2,8))));tail=np.vstack((states[i+1],np.zeros((2,8))))
        coeff,_=MINCOQuintic(head,tail,1).solve(np.empty((0,8)),np.array([dt]))
        coefficients.append(coeff.reshape(6,8))
    trajectory=AerialManipulatorTrajectory(durations,np.array(coefficients),states,0.,0,False,'offline geometry witness')
    config=replace(AerialManipulatorMINCOConfig(),esdf_resolution=.06,esdf_discretization_margin=.105,self_clearance=0.)
    evaluator=PrivilegedEvaluator.from_simulation(sim,config=config).evaluator
    passed,_,violation,_=evaluator.dense_validate(trajectory)
    assert passed and violation==0 and not evaluator.last_validation_metrics['collision_detected'], (violation,evaluator.last_validation_metrics)


def test_outdoor_ceiling_can_be_disabled_without_disabling_other_bounds():
    from dataclasses import replace
    for enforce,height,x,expected_failure in (
        (True,4.,0.,True), (False,4.,0.,False),
        (False,.7,0.,True), (False,1.65,5.,True),
    ):
        sim,task,planner=setup(.5)
        task.settings=replace(task.settings,enforce_height_ceiling=enforce)
        state=np.array([x,0.,-height,0.,0.,0.,0.,0.])
        q,v=recover_state(np.vstack((state,np.zeros((3,8)))),.06)
        sim.robot.reset(q.copy(),v.copy())
        task.command_reference=task._hold(sim.robot.configuration)
        task.step()
        if expected_failure:
            assert task.failure_reason=='actual_workspace_violation'
        else:
            assert not task.done and len(planner.observations)==1
        task.close()
