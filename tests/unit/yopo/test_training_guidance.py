import runpy
import numpy as np
import pytest
import torch
from uav_ac.YOPO.learning import DirectEvaluator, ranking_loss, augment_observation
from uav_ac.YOPO.losses import guidance_costs
from uav_ac.planning.end2end.anchors import make_anchors
from uav_ac.planning.end2end.encoding import Normalization
from uav_ac.simulation.mujoco_sim import MujocoSimulation


@pytest.mark.parametrize('peak_weight',[0.,1.])
def test_direct_guidance_time_and_boundary_gradients(peak_weight):
    o=runpy.run_path('tests/unit/planning/end2end/test_rolling.py')['moving_observation']()
    sim=MujocoSimulation('uav_ac/simulation/models/yopo_free.xml',planning_path_capacity=0,record_actual_trajectory=False)
    n=Normalization(tuple(sim.robot.limits.joint_lower),tuple(sim.robot.limits.joint_upper))
    oracle=DirectEvaluator.from_simulation(sim);z=make_anchors(o,n,count=1)[0]
    oracle.peak_weight=peak_weight
    result=oracle.evaluate(o,z[None]);assert result.free_gradients.shape==(1,34)
    rng=np.random.default_rng(22)
    for _ in range(4):
        d=rng.normal(size=34);d/=np.linalg.norm(d)
        fd=(oracle.evaluate(o,(z+1e-6*d)[None]).costs[0]-oracle.evaluate(o,(z-1e-6*d)[None]).costs[0])/2e-6
        assert abs(fd-result.free_gradients[0]@d)/max(1,abs(fd))<1e-4
    tensor=torch.tensor(z[None],requires_grad=True);cost=guidance_costs(tensor,o,oracle);cost.mean().backward()
    assert np.allclose(tensor.grad.numpy(),result.free_gradients)


def test_ranking_prefers_safe_candidate_and_detaches_labels():
    scores=torch.tensor([[0.,0.,0.]],requires_grad=True)
    costs=torch.tensor([[.1,2.,3.]],requires_grad=True);unsafe=torch.tensor([[True,False,False]])
    loss,target=ranking_loss(scores,costs,unsafe);assert target.item()==1
    loss.backward();assert costs.grad is None and scores.grad is not None
    _,target=ranking_loss(scores,costs,torch.ones_like(unsafe));assert target.item()==0


def test_augmentation_preserves_observed_image_and_robot_pose():
    o=runpy.run_path('tests/unit/planning/end2end/test_rolling.py')['moving_observation']()
    n=Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4))
    changed=augment_observation(o,np.random.default_rng(2),n,goal_std=.1,velocity_std=.05)
    assert np.array_equal(o.depth_m,changed.depth_m) and np.array_equal(o.configuration,changed.configuration)
    assert not np.array_equal(o.velocity,changed.velocity)
    assert np.allclose(changed.start_pva[1,:3],changed.velocity[:3])


def test_training_proxy_clearance_allows_authored_tangent_connections():
    from dataclasses import replace
    from uav_ac.YOPO.training import TrainingConfig
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
    case=runpy.run_path('tests/unit/yopo/test_guidance_gradient.py')
    sim,o,_=case['setup_case']();n=Normalization(tuple(sim.robot.limits.joint_lower),tuple(sim.robot.limits.joint_upper))
    cfg=TrainingConfig()
    assert cfg.self_clearance==0.
    evaluator=DirectEvaluator.from_simulation(sim,config=replace(AerialManipulatorMINCOConfig(),self_clearance=cfg.self_clearance))
    result=evaluator.evaluate(o,make_anchors(o,n,count=1))
    assert result.components['self'][0]<1e-6


def test_checkpoint_selection_prioritizes_constraint_validity():
    from uav_ac.YOPO.training import checkpoint_priority
    safe={'selected_valid_fraction':1.,'selected_cost':1000.}
    unsafe={'selected_valid_fraction':0.,'selected_cost':10.}
    assert checkpoint_priority(safe)<checkpoint_priority(unsafe)


def test_curriculum_family_selection_and_training_configuration():
    from uav_ac.YOPO.training import TrainingConfig,subset
    class Data:
        def __len__(self):return 4
        def family_at(self,i):return ('free','obstacle','low','wall')[i]
        def __getitem__(self,i):return i
    assert subset(Data(),10,families=('free','low'))==[0,2]
    assert TrainingConfig(families=('free',)).families==('free',)
    with pytest.raises(ValueError,match='famil'):TrainingConfig(families=('bogus',))


def test_near_goal_stopping_cost_and_gradient():
    from dataclasses import replace
    o=runpy.run_path('tests/unit/planning/end2end/test_rolling.py')['moving_observation']()
    goal=o.goal.copy();goal[0]=o.configuration[0]+.5;o=replace(o,goal=goal)
    sim=MujocoSimulation('uav_ac/simulation/models/yopo_free.xml',planning_path_capacity=0,record_actual_trajectory=False)
    n=Normalization(tuple(sim.robot.limits.joint_lower),tuple(sim.robot.limits.joint_upper));z=make_anchors(o,n,count=1)[0]
    oracle=DirectEvaluator.from_simulation(sim,goal_stop_weight=50.,goal_stop_distance=1.5)
    result=oracle.evaluate(o,z[None]);assert result.components['goal_stop'][0]>0
    d=np.zeros(34);d[16]=1.
    fd=(oracle.evaluate(o,(z+1e-6*d)[None]).costs[0]-oracle.evaluate(o,(z-1e-6*d)[None]).costs[0])/2e-6
    assert abs(fd-result.free_gradients[0]@d)/max(1.,abs(fd))<1e-4


def test_exact_training_geometry_agrees_with_validated_folded_pose():
    from dataclasses import replace
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
    sim=MujocoSimulation('uav_ac/simulation/models/yopo_stage_wall.xml',planning_path_capacity=0,record_actual_trajectory=False)
    cfg=replace(AerialManipulatorMINCOConfig(),self_clearance=0.)
    sigma=np.array([1.5,0,-1.8,0,0,-1.6,1.9,-.3]);a=np.zeros(8)
    oracle=DirectEvaluator.from_simulation(sim,config=cfg,collision_model='exact_geometry')
    q=oracle.evaluator._configuration(sigma,a)
    assert not sim.robot.check_collision(q,clearance=.05,self_clearance=0.)['pairs']
    assert oracle._parts['self'].collision_cost_gradient(sigma,a)[0]<1e-8
    assert oracle._parts['environment'].collision_cost_gradient(sigma,a)[0]<1e-8


def test_exact_environment_collision_gradient_at_penetrating_arm_pose():
    from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig
    sim=MujocoSimulation('uav_ac/simulation/models/yopo_stage_wall.xml',planning_path_capacity=0,record_actual_trajectory=False)
    oracle=DirectEvaluator.from_simulation(sim,config=AerialManipulatorMINCOConfig(self_clearance=0.),collision_model='exact_geometry')
    # Avoid the symmetric, nondifferentiable gripper/box contact orientation.
    sigma=np.array([1.451,.023,-1.763,.031,.027,-.81,1.03,-.17])
    acceleration=np.array([.02,-.01,.03,0,0,0,0,0])
    evaluate=oracle._parts['environment'].collision_cost_gradient
    value=evaluate(sigma,acceleration);assert value[0]>0
    for dimension in range(8):
        step=np.eye(8)[dimension]*1e-6
        numeric=(evaluate(sigma+step,acceleration)[0]-evaluate(sigma-step,acceleration)[0])/2e-6
        analytic=value[1][dimension]
        assert abs(numeric-analytic)/max(1.,abs(numeric),abs(analytic))<1e-4
