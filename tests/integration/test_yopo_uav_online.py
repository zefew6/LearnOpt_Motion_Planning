from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.control import CascadedController
from uav_ac.yopo.vendor.yopo_minco.policy.poly_solver import MincoTraj


class ImmediateExecutor:
    def submit(self,fn,*args):
        f=Future()
        try:f.set_result(fn(*args))
        except Exception as e:f.set_exception(e)
        return f
    def shutdown(self,**kwargs):pass


class Observer:
    def observe(self):return np.full((96,160),20.,dtype=np.float32),np.zeros((96,160),bool)


class Planner:
    def __init__(self):self.states=[]
    def plan(self,depth,state,acc,goal):
        self.states.append(state.copy());d=np.diag([1.,-1.,-1.])
        head=np.stack((d@state[:3],d@state[7:10],d@acc));tail=np.stack((d@goal,np.zeros(3),np.zeros(3)));middle=(head[0]+tail[0])/2
        t=MincoTraj().solve(head,tail,middle,np.array([1.,1.]))
        return SimpleNamespace(trajectory=t,brake=False)


def test_uav_online_tracks_physics_and_replans_without_state_resets():
    from uav_ac.tasks.yopo_navigation import UavNavigation
    from uav_ac.runners.yopo_navigation import UavSettings
    sim=MujocoSimulation('uav_ac/simulation/models/uav_yopo_bamboo.xml',planning_path_capacity=0,record_actual_trajectory=False)
    initial=sim.quad.X.copy();hover=np.full(4,np.sqrt(sim.quad.m*sim.quad.g/(4*sim.quad.kf)))
    sim.reset(initial,hover);sim.goal_position=initial[:3]+[.5,0,0]
    planner=Planner();controller=CascadedController(sim.quad.g,sim.quad.dt)
    task=UavNavigation(sim,controller,planner,Observer(),UavSettings(replan_dt=.04,goal_tolerance=.05),.02,2.,executor=ImmediateExecutor())
    sim.reset=lambda *a,**k:(_ for _ in ()).throw(AssertionError('online reset forbidden'))
    for _ in range(700):
        before=sim.quad.X.copy();task.step();np.testing.assert_array_equal(sim.quad.X,before);sim.step()
    assert task.accepted>2 and len(planner.states)>2
    assert np.linalg.norm(sim.quad.position-initial[:3])>.001 and not sim.collision_detected
    assert not np.array_equal(planner.states[0],planner.states[-1])
    task.close()


def test_goal_settling_discards_pending_snapshot_and_old_plan():
    from uav_ac.tasks.yopo_navigation import UavNavigation
    from uav_ac.runners.yopo_navigation import UavSettings
    sim=MujocoSimulation('uav_ac/simulation/models/uav_yopo_bamboo.xml',planning_path_capacity=0,record_actual_trajectory=False)
    sim.goal_position=sim.quad.position.copy()
    task=UavNavigation(sim,CascadedController(sim.quad.g,sim.quad.dt),Planner(),Observer(),UavSettings(),.02,2.,executor=ImmediateExecutor())
    task.future=Future();task.future.set_running_or_notify_cancel();task.plan=object()
    task.step()
    assert task.state=='SETTLING' and task.future is None and task.plan is None
    task.close()


def test_uav_depth_can_exclude_robot_without_changing_live_geometry():
    from uav_ac.simulation.depth_observation import IdealDepthObserver
    from uav_ac.yopo.whole_body.contracts import DepthSpec
    sim=MujocoSimulation('uav_ac/simulation/models/uav_yopo_bamboo.xml',planning_path_capacity=0,record_actual_trajectory=False)
    groups=sim.model.geom_group.copy();state=sim.quad.X.copy()
    spec=DepthSpec(height=90,vfov_deg=float(np.rad2deg(2*np.arctan(45/80))),far_m=20.)
    with IdealDepthObserver(sim.model,sim.data,spec=spec,exclude_robot=True) as observer:
        depth,valid=observer.observe()
        assert depth.shape==(90,160) and depth[valid].min()>1.
    np.testing.assert_array_equal(sim.model.geom_group,groups)
    np.testing.assert_array_equal(sim.quad.X,state)
