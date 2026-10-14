from tests.yopo_scenes import scene_path
from dataclasses import replace
import numpy as np
import pytest
import torch


def test_goal_matches_official_parallel_perpendicular_huber_and_collision_detach():
    from uav_ac.YOPO.minco_guidance import goal_cost_gradient, MincoGuidanceConfig
    cfg=MincoGuidanceConfig();head=np.zeros(8);goal=np.array([2.,1.,-.5,.3,.2,-.3,.4,-.1]);tail=np.array([.6,.7,-.3,-.1,.1,.2,-.2,.2])
    value,gradient=goal_cost_gradient(head,tail,goal,cfg)
    x=torch.tensor(tail,dtype=torch.float64,requires_grad=True);g=torch.tensor(goal);direction=g[:3]/torch.linalg.vector_norm(g[:3]);e=x[:3]-g[:3]
    parallel=torch.dot(e,direction)*direction;perpendicular=e-parallel
    huber=lambda a:torch.nn.functional.smooth_l1_loss(a,torch.zeros_like(a),reduction='sum')
    reference=cfg.goal_weight*(huber(parallel)+cfg.goal_perpendicular_weight*huber(perpendicular)+cfg.goal_pose_weight*(huber(torch.atan2(torch.sin(x[3]-g[3]),torch.cos(x[3]-g[3]))/.7)+huber((x[4:]-g[4:])/.8)))
    reference.backward();assert value==pytest.approx(reference.item());assert np.allclose(gradient,x.grad.numpy())
    v,blocked=goal_cost_gradient(head,tail,goal,cfg,collided=True)
    assert v==value and np.array_equal(blocked,np.zeros(8))


def test_focused_safety_uses_last_safe_sample_and_margin_cost_is_not_collision():
    from uav_ac.YOPO.minco_guidance import focused_safety
    cost=np.array([.2,.5,100.,200.]);contacts=np.array([False,False,True,True])
    value,weight,collided=focused_safety(cost,contacts,beta=4.)
    assert collided and value==.5 and np.array_equal(weight,[0,1,0,0])
    value,weight,collided=focused_safety(cost[:2],np.zeros(2,dtype=bool),beta=4.)
    assert not collided and .2<value<.5 and weight[1]>weight[0] and weight.sum()==pytest.approx(1.)


def test_curvature_cap_detaches_geometry_but_full_derivative_remains_auditable():
    from uav_ac.YOPO.minco_guidance import curvature_cost_gradient
    v=np.array([2.,.1,0]);a=np.array([.2,3.,.1])
    value,guide,detached=curvature_cost_gradient(v,a,3.,3.,.8)
    assert value>0 and np.array_equal(guide[1],np.zeros(3))
    assert np.dot(guide[0],v)>0
    for i in range(6):
        d=np.eye(6)[i]*1e-6;plus=np.r_[v,a]+d;minus=np.r_[v,a]-d
        fd=(curvature_cost_gradient(plus[:3],plus[3:],3.,3.,.8)[0]-curvature_cost_gradient(minus[:3],minus[3:],3.,3.,.8)[0])/2e-6
        assert fd==pytest.approx((guide+detached).ravel()[i],rel=1e-5,abs=1e-7)


def case(scene='free'):
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    from uav_ac.planning.end2end.contracts import PlannerObservation
    from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
    sim=MujocoSimulation(scene_path(scene),planning_path_capacity=0,record_actual_trajectory=False)
    pva=np.zeros((3,8));pva[0,:3]=[0,0,-1.8];q,v=recover_state(np.vstack((pva,np.zeros((1,8)))),.06);sim.robot.reset(q,v)
    goal=pva[0].copy();goal[0]=3.
    obs=PlannerObservation(np.full((96,160),6.),np.zeros((96,160),bool),q,v,pva,goal)
    z=np.r_[[1.5,0,0,0,0,0,0,0],[3,0,0,0,0,0,0,0],np.zeros(16),2.5,2.5]
    return sim,obs,z


def test_whole_body_collision_blocks_goal_without_stopping_or_gyro_matching():
    from uav_ac.YOPO.minco_guidance import MincoGuidanceEvaluator,MincoGuidanceConfig
    sim,o,z=case('wall');z[8]=2.7;settings=MincoGuidanceConfig(curvature_weight=0.)
    oracle=MincoGuidanceEvaluator.from_simulation(sim,guidance=settings)
    before=(sim.robot.configuration.copy(),sim.robot.velocity.copy(),sim.data.time)
    r=oracle.evaluate(o,z[None]);assert r.collided[0] and r.components['environment'][0]>0
    oracle.guidance=replace(settings,goal_weight=0.)
    no_goal=oracle.evaluate(o,z[None]);assert r.costs[0]>no_goal.costs[0]
    assert np.allclose(r.free_gradients,no_goal.free_gradients)
    assert np.linalg.norm(r.detached_gradients)>0
    assert 'goal_stop' not in r.components and 'continuity_and_peak' not in r.components
    oracle.guidance=settings
    velocity=o.velocity.copy();velocity[3:6]+=[.5,-.2,.1]
    changed=oracle.evaluate(replace(o,velocity=velocity),z[None])
    assert np.allclose(r.costs,changed.costs) and np.allclose(r.free_gradients,changed.free_gradients)
    assert np.array_equal(before[0],sim.robot.configuration) and np.array_equal(before[1],sim.robot.velocity) and before[2]==sim.data.time


def test_minco_guidance_all_34_parameter_pullbacks_match_finite_difference():
    from uav_ac.YOPO.minco_guidance import MincoGuidanceEvaluator,MincoGuidanceConfig
    sim,o,z=case();z=z.copy();z[:3]=[1.4,.06,-.02];z[8:11]=[2.4,.08,.03];z[3:8]=[.02,.03,-.06,.1,-.03];z[11:16]=[.02,.03,-.06,.1,-.03]
    oracle=MincoGuidanceEvaluator.from_simulation(sim,guidance=MincoGuidanceConfig(curvature_weight=0.))
    r=oracle.evaluate(o,z[None]);assert not r.collided[0]
    assert r.costs[0]==pytest.approx(sum(v[0] for v in r.components.values()))
    for i in range(34):
        d=np.eye(34)[i]*1e-5;fd=(oracle.evaluate(o,(z+d)[None]).costs[0]-oracle.evaluate(o,(z-d)[None]).costs[0])/2e-5
        assert fd==pytest.approx(r.free_gradients[0,i],rel=2e-4,abs=2e-4)


def test_new_recipe_rejects_stopping_and_duplicate_peak_and_unknown_guidance():
    from uav_ac.YOPO.training import TrainingConfig
    TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry')
    for kwargs in ({'goal_stop_weight':1.},{'peak_weight':1.},{'guidance':{'not_a_weight':1.}}):
        with pytest.raises(ValueError):TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry',**kwargs)


def test_curved_minco_full_cost_fd_and_surrogate_backward_are_distinct_and_correct():
    from uav_ac.YOPO.minco_guidance import MincoGuidanceEvaluator,MincoGuidanceConfig
    from uav_ac.YOPO.audit import audit_candidates
    sim,o,z=case();z[:3]=[1.1,.7,.03];z[8:11]=[2.4,.6,.02];z[16:19]=[2.4,-.3,.1];z[24:27]=[.1,.3,.1];z[-2:]=[.7,.8]
    oracle=MincoGuidanceEvaluator.from_simulation(sim,guidance=MincoGuidanceConfig())
    result=oracle.evaluate(o,z[None]);assert result.components['curvature'][0]>0
    assert np.linalg.norm(result.detached_gradients)>0
    report=audit_candidates(o,oracle,z[None],[0],epsilon=1e-5,tolerance=5e-4)
    assert report['passed'],report['candidates'][0]['max_relative_gradient_error']
    from uav_ac.YOPO.losses import guidance_costs
    tensor=torch.tensor(z[None],requires_grad=True);guidance_costs(tensor,o,oracle).sum().backward()
    assert np.allclose(tensor.grad.numpy(),result.free_gradients)


def test_zero_feasibility_weight_does_not_hide_physical_violations_in_metrics():
    from uav_ac.YOPO.minco_guidance import MincoGuidanceEvaluator,MincoGuidanceConfig
    from uav_ac.YOPO.training import evaluate
    from uav_ac.planning.end2end.network import NetworkOutput
    from uav_ac.planning.end2end.encoding import Normalization
    sim,o,z=case();z[-2:]=.4
    oracle=MincoGuidanceEvaluator.from_simulation(sim,guidance=MincoGuidanceConfig(feasibility_weight=0.))
    r=oracle.evaluate(o,z[None]);assert not r.collided[0] and r.components['physical'][0]==0 and r.violations[0]>0
    class FixedPrediction(torch.nn.Module):
        profile='direct_v3';score_mode='ranking'
        normalization=Normalization(tuple(sim.robot.limits.joint_lower),tuple(sim.robot.limits.joint_upper))
        def forward(self,*args):return NetworkOutput(torch.tensor(z[None,None],dtype=torch.float64),torch.ones(1,1))
    class Cache:
        def get(self,priv):return oracle
    metrics=evaluate(FixedPrediction(),[(o,None)],Cache(),1,'cpu')
    assert metrics['selected_valid_fraction']==0.
    assert metrics['selected_violation_fraction']['physical']==1.


def test_acceleration_energy_is_exact_polynomial_integral():
    from uav_ac.YOPO.minco_guidance import MincoGuidanceEvaluator,MincoGuidanceConfig
    from uav_ac.planning.end2end.direct import world_problem
    sim,o,z=case();z[:3]=[1.1,.2,.03];z[8:11]=[2.4,.3,.02];z[-2:]=[.9,1.1]
    oracle=MincoGuidanceEvaluator.from_simulation(sim,guidance=MincoGuidanceConfig())
    result=oracle.evaluate(o,z[None]);minco,points,times,_,_=world_problem(z,o);coeff,_=minco.solve(points,times);coeff=coeff.reshape(2,6,8)
    expected=0.;scales=np.r_[np.full(3,.5),.7,np.full(4,.8)]
    for c,t in zip(coeff,times):
        for d in range(8):
            a=np.polynomial.polynomial.polyder(c[:,d],2)
            integral=np.polynomial.polynomial.polyint(np.polynomial.polynomial.polymul(a,a))
            expected+=.002/scales[d]**2*np.polynomial.polynomial.polyval(t,integral)
    assert result.components['acceleration'][0]==pytest.approx(expected,rel=1e-10,abs=1e-10)
