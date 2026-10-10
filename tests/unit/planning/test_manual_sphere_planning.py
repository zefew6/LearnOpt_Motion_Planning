import numpy as np
import pytest
from types import SimpleNamespace
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import AerialManipulatorTrajectoryEvaluator, TaskWaypoint
from uav_ac.planning.trajectory.aerial_manipulator_minco.planner import AerialManipulatorMINCO
from uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory

@pytest.fixture
def context():
    sim=MujocoSimulation('uav_ac/simulation/models/aerial_manipulator_workcell.xml',record_actual_trajectory=False)
    esdf=ESDF.from_axis_aligned_boxes(np.empty((0,6)),[-3.,-3.,-3.],[3.,3.,1.],.2,ground_height=0.)
    cfg=AerialManipulatorMINCOConfig(self_clearance=0.,integral_resolution=2)
    return sim,esdf,cfg

def test_robot_authored_spheres_and_structural_pair_mask(context):
    sim,esdf,cfg=context
    spheres=sim.robot.planning_collision_spheres()
    assert len(spheres)==21
    assert not spheres[0]['center'].flags.writeable
    for loaded,n in [(False,21),(True,22)]:
        e=AerialManipulatorTrajectoryEvaluator(sim.robot,esdf,sim.quad,cfg,sim.space_limits,.06,loaded)
        assert len(e.sphere_radii)==n
        assert len(e.self_pairs)==170
        assert (6,8) in map(tuple,e.self_pairs)  # adjacent remote spheres retained
        assert (6,7) not in map(tuple,e.self_pairs)  # shared connection centre
        assert len(e.payload_pairs)==(16 if loaded else 0)

def test_sphere_queries_have_analytic_gradients_and_never_call_exact_geometry(context,monkeypatch):
    sim,esdf,cfg=context
    e=AerialManipulatorTrajectoryEvaluator(sim.robot,esdf,sim.quad,cfg,sim.space_limits,.06,True)
    monkeypatch.setattr(sim.robot,'exact_collision_distances',lambda *a,**k:pytest.fail('runtime exact query'))
    state=np.array([.113,.127,-.213,.3,.4,.65,-.8,.2]);acc=np.array([.4,.2,.1,0,0,0,0,0.])
    before=sim.data.qpos.copy()
    value,g,ga,*_=e.collision_cost_gradient(state,acc)
    for i in (0,2,3,4,5,7):
        plus=state.copy();minus=state.copy();plus[i]+=1e-6;minus[i]-=1e-6
        numeric=(e.collision_cost_gradient(plus,acc)[0]-e.collision_cost_gradient(minus,acc)[0])/2e-6
        np.testing.assert_allclose(g[i],numeric,rtol=3e-4,atol=.03)
    for i in range(3):
        plus=acc.copy();minus=acc.copy();plus[i]+=1e-6;minus[i]-=1e-6
        numeric=(e.collision_cost_gradient(state,plus)[0]-e.collision_cost_gradient(state,minus)[0])/2e-6
        np.testing.assert_allclose(ga[i],numeric,rtol=3e-4,atol=.03)
    e.collision_feasible_batch(state[None,:])
    np.testing.assert_array_equal(sim.data.qpos,before)

def test_task_has_one_solve_and_no_offline_validation_or_restart(monkeypatch):
    from uav_ac.planning.trajectory.aerial_manipulator_minco import optimization, search, constraints
    calls=[]
    trajectory=AerialManipulatorTrajectory(np.ones(1),np.zeros((1,6,8)),np.zeros((2,8)),0.,0,False,'limit')
    problem=SimpleNamespace(initial=np.zeros(1),objective=lambda x:(0.,x),
        equalities=lambda x:(np.zeros(1),None),extra_equalities=lambda x:(np.zeros(0),None),
        trajectories=lambda x,r:[trajectory,trajectory])
    monkeypatch.setattr(search,'task_initialization',lambda *a,**k:([None,None],[SimpleNamespace(metrics={})]*2))
    def evaluator(*a,**k):
        return SimpleNamespace(last_violation=-.1,minimum_clearance=.2,objective_samples=0,
            dense_validate=lambda *a:pytest.fail('offline validation called'))
    monkeypatch.setattr(constraints,'AerialManipulatorTrajectoryEvaluator',evaluator)
    monkeypatch.setattr(optimization,'JointTaskObjective',lambda *a,**k:problem)
    def solve(*a,**k):
        calls.append(1)
        return SimpleNamespace(x=np.zeros(1),iterations=1,converged=False,message='limit')
    monkeypatch.setattr(optimization,'scipy_lbfgs',solve)
    planner=AerialManipulatorMINCO()
    plans,_=planner.plan_task(np.zeros(8),[TaskWaypoint(np.zeros(3))]*2,[np.zeros(8)]*2,
        robot=None,esdf=None,quad=None,workspace_bounds=None,gaps=[.06,.025])
    assert len(calls)==1
    assert plans['pick'].validation_passed is None
    assert not plans['pick'].validation_performed
    assert not plans['pick'].optimizer_converged

def test_connected_elbow_can_bend_without_false_proxy_self_collision(context):
    sim,esdf,cfg=context
    e=AerialManipulatorTrajectoryEvaluator(sim.robot,esdf,sim.quad,cfg,sim.space_limits,.06,False)
    states=np.tile(np.r_[0.,0.,-1.,0.,0.,0.,0.,0.],(3,1))
    states[:,6]=[-.4,0.,.4]
    assert np.all(e.collision_feasible_batch(states))

def test_custom_robot_without_planning_sites_keeps_legacy_collision_api(context,monkeypatch):
    sim,esdf,cfg=context
    monkeypatch.setattr(sim.robot,'planning_collision_spheres',lambda:())
    e=AerialManipulatorTrajectoryEvaluator(sim.robot,esdf,sim.quad,cfg,sim.space_limits,.06,False)
    assert not e.manual_spheres and len(e.sphere_radii)>21
    value,gradient,*_=e.collision_cost_gradient(np.r_[0.,0.,-1.5,0.,0.,0.,0.,0.])
    assert np.isfinite(value) and np.all(np.isfinite(gradient))

def test_explicit_offline_status_is_distinct_from_unvalidated_result():
    from dataclasses import replace
    from uav_ac.tasks.aerial_pick_place import _plan_result_fields
    p=AerialManipulatorTrajectory(np.ones(1),np.zeros((1,6,8)),np.zeros((2,8)),0.,1,False,'limit')
    fields=_plan_result_fields({'pick':p,'place':p})
    assert fields['pick_plan_valid'] is None
    assert fields['pick_validation_performed'] is False
    checked=replace(p,validation_passed=False)
    fields=_plan_result_fields({'pick':checked,'place':checked})
    assert fields['pick_plan_valid'] is False
    assert fields['pick_validation_performed'] is True

def test_optimizer_penalizes_sphere_extents_outside_known_map(context):
    sim,_,cfg=context
    esdf=ESDF(np.full((16,16,16),10.),np.array([-1.5,-1.5,-3.]),.2)
    e=AerialManipulatorTrajectoryEvaluator(sim.robot,esdf,sim.quad,cfg,sim.space_limits,.06,False)
    state=np.r_[1.32,0.,-1.5,0.,0.,0.,0.,0.]
    assert not e.collision_feasible_batch(state[None,:])[0]
    value,g,_,v,clearance,*_=e.collision_cost_gradient(state)
    assert value > 0. and v > 0. and clearance < 0.
    assert g[0] > 0.
    plus=state.copy();minus=state.copy();plus[0]+=1e-6;minus[0]-=1e-6
    numeric=(e.collision_cost_gradient(plus)[0]-e.collision_cost_gradient(minus)[0])/2e-6
    np.testing.assert_allclose(g[0],numeric,rtol=3e-4,atol=.01)
