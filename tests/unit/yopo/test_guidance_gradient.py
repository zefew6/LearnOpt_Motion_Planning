from tests.yopo_scenes import scene_path
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.yopo.contracts import PlannerObservation
from uav_ac.yopo.trajectory.legacy import InteriorMap
from uav_ac.yopo.training.legacy import PrivilegedEvaluator
from uav_ac.yopo.training.losses import guidance_loss, score_loss, calibrate_cost_scale


def setup_case():
    sim=MujocoSimulation(scene_path('free'),planning_path_capacity=0,record_actual_trajectory=False)
    q=sim.robot.configuration.copy();q[:3]=[0.,0.,-1.5];q[7:11]=0.;q[11]=.06
    sim.robot.reset(q,np.zeros(11))
    y=np.r_[q[:3],0.,q[7:11]];head=np.vstack((y,np.zeros((2,8))))
    goal=y.copy();goal[0]=.8;goal[1]=.1;goal[3]=.1
    obs=PlannerObservation(np.full((96,160),6.),np.zeros((96,160),bool),q,np.zeros(11),head,goal)
    from uav_ac.yopo.model.encoding import LocalFrame
    frame=LocalFrame.from_observation(obs)
    free=InteriorMap(frame.local_pva(head),np.vstack((frame.local_goal(goal),np.zeros((2,8))))).seed()
    return sim,obs,free


def test_oracle_gradient_equivalence_and_live_state_unchanged():
    sim,obs,free=setup_case();oracle=PrivilegedEvaluator.from_simulation(sim)
    before=(sim.data.qpos.copy(),sim.data.qvel.copy(),sim.data.time)
    result=oracle.evaluate(obs,free[None])
    assert np.isfinite(result.costs).all() and np.isfinite(result.free_gradients).all()
    assert result.costs[0]==pytest.approx(sum(v[0] for v in result.components.values()),rel=1e-9)
    rng=np.random.default_rng(9)
    for _ in range(10):
        d=rng.normal(size=20);d/=np.linalg.norm(d)
        fd=(oracle.evaluate(obs,(free+1e-6*d)[None]).costs[0]-oracle.evaluate(obs,(free-1e-6*d)[None]).costs[0])/2e-6
        ad=result.free_gradients[0]@d
        assert abs(fd-ad)/max(1.,abs(fd),abs(ad))<1e-4
    assert np.array_equal(before[0],sim.data.qpos) and np.array_equal(before[1],sim.data.qvel)
    assert sim.data.time==before[2]
    cost=guidance_loss(torch.tensor(free[None],requires_grad=True),obs,oracle,scale=10.)
    cost.backward()


def test_score_target_detach_and_cost_calibration():
    target=torch.tensor([2.,3.],requires_grad=True);pred=torch.tensor([1.,2.],requires_grad=True)
    score_loss(pred,target).backward()
    assert target.grad is None and pred.grad is not None
    assert calibrate_cost_scale([0.,2.,4.])==2.
    with pytest.raises(ValueError): calibrate_cost_scale([np.nan])


def test_reference_objective_and_backward_are_numerically_identical():
    from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective
    sim,obs,free=setup_case();oracle=PrivilegedEvaluator.from_simulation(sim)
    from uav_ac.yopo.model.encoding import LocalFrame
    frame=LocalFrame.from_observation(obs)
    mapping=InteriorMap(frame.local_pva(obs.start_pva),np.vstack((frame.local_goal(obs.goal),np.zeros((2,8)))))
    world_minco=InteriorMap(obs.start_pva,np.vstack((obs.goal,np.zeros((2,8))))).minco
    result=oracle.evaluate(obs,free[None])
    value,gp,_,_=evaluate_minco_objective(world_minco,frame.world_points8(mapping.decode(free)),mapping.durations,
        terms=(oracle.evaluator.integrated_penalty,),energy_weights=np.asarray(oracle.config.jerk_weights))
    assert value==pytest.approx(result.costs[0],rel=1e-10,abs=1e-8)
    assert np.allclose(mapping.pullback(frame.pullback_points8(gp)),result.free_gradients[0],rtol=1e-9,atol=1e-7)
    tensor=torch.tensor(free[None],dtype=torch.float64,requires_grad=True)
    guidance_loss(tensor,obs,oracle,scale=10.).backward()
    assert np.allclose(tensor.grad.numpy(),result.free_gradients/10.,rtol=1e-10)


def test_collision_and_nonzero_yaw_pullback_gradients():
    from scipy.spatial.transform import Rotation
    from uav_ac.planning.geometry.esdf import ESDF
    sim,obs,_=setup_case()
    q=obs.configuration.copy();q[3:7]=[np.cos(.3),0,0,np.sin(.3)]
    head=obs.start_pva.copy();head[0,3]=.6
    goal=obs.goal.copy();goal[3]=.7
    obs=PlannerObservation(obs.depth_m,obs.depth_valid,q,obs.velocity,head,goal)
    shape=(140,140,140);origin=np.array([-1.,-1.,-3.])
    x=origin[0]+.04*np.arange(shape[0])
    values=np.broadcast_to((.55-x)[:,None,None],shape).copy()
    oracle=PrivilegedEvaluator(sim.robot,ESDF(values,origin,.04),sim.quad,sim.space_limits)
    from uav_ac.yopo.model.encoding import LocalFrame
    frame=LocalFrame.from_observation(obs)
    free=InteriorMap(frame.local_pva(head),np.vstack((frame.local_goal(goal),np.zeros((2,8))))).seed()
    result=oracle.evaluate(obs,free[None]);assert result.components['environment'][0]>0
    rng=np.random.default_rng(77)
    for _ in range(10):
        d=rng.normal(size=20);d/=np.linalg.norm(d)
        fd=(oracle.evaluate(obs,(free+1e-6*d)[None]).costs[0]-oracle.evaluate(obs,(free-1e-6*d)[None]).costs[0])/2e-6
        ad=result.free_gradients[0]@d
        assert abs(fd-ad)/max(1.,abs(fd),abs(ad))<1e-4


def test_oracle_rejects_stationary_pva_with_tilted_actual_start():
    sim,obs,free=setup_case();q=obs.configuration.copy();q[3:7]=[np.cos(.15),0,np.sin(.15),0]
    changed=PlannerObservation(obs.depth_m,obs.depth_valid,q,obs.velocity,obs.start_pva,obs.goal)
    oracle=PrivilegedEvaluator.from_simulation(sim)
    with pytest.raises(ValueError,match='attitude'):oracle.evaluate(changed,free[None])
