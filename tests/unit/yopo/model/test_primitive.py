from tests.yopo_scenes import scene_path
import runpy
import numpy as np
import torch
from uav_ac.yopo.model.encoding import Normalization
from uav_ac.yopo.model.primitives import PrimitiveSpec, primitive_anchors, encode_primitive, primitive_state
from uav_ac.yopo.model.network import PrimitivePlannerNet
from uav_ac.yopo.trajectory.minco import decode_direct


def observation():return runpy.run_path('tests/unit/yopo/model/test_rolling.py')['moving_observation']()
def normalization():return Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4))


def test_grid_rotations_and_state_alignment_and_goal_clipping():
    o=observation();n=normalization();spec=PrimitiveSpec();e=encode_primitive(o,n,spec)
    model=PrimitivePlannerNet(n,width=4,depth_spec=o.depth_spec,spec=spec)
    rotations=model.primitive_rotations.detach().numpy()
    assert rotations.shape==(9,3,3)
    assert np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-6)
    assert np.allclose(np.linalg.det(rotations),1,atol=1e-6)
    state=torch.tensor(e.state);aligned=primitive_state(state,model.primitive_rotations)
    assert aligned.shape==(1,9,58)
    assert torch.allclose(aligned[0,:,45:49],state[0,45:49].expand(9,4))
    assert np.linalg.norm(e.state[0,24:27])<=1.000001
    assert np.allclose(aligned[0,:,42:45].numpy(),np.einsum('nij,j->ni',rotations.transpose(0,2,1),e.state[0,42:45]),atol=1e-6)


def test_network_frustum_parameters_independent_tail_gradients_and_actual_head():
    torch.set_num_threads(2);o=observation();n=normalization();spec=PrimitiveSpec()
    model=PrimitivePlannerNet(n,width=4,depth_spec=o.depth_spec,spec=spec).eval()
    e=encode_primitive(o,n,spec);anchors=primitive_anchors(o,n,spec)
    output=model(torch.tensor(e.depth),torch.tensor(e.state),torch.tensor(anchors[None],dtype=torch.float32))
    assert output.parameters.shape==(1,27,34) and output.scores.shape==(1,27)
    assert torch.isfinite(output.parameters).all()
    assert len(np.unique(np.round(anchors[:,:3],6),axis=0))==9
    selected=decode_direct(output.parameters[0,0].detach().numpy(),o)
    for d in range(3):assert np.allclose(selected.evaluate(0,d),o.start_pva[d],atol=1e-7)
    output.parameters.sum().backward()
    assert model.parameter_head.weight.grad is not None
    assert torch.isfinite(model.parameter_head.weight.grad).all()


def test_new_and_legacy_checkpoints_roundtrip_and_primitive_contract(tmp_path):
    import json,pytest
    from uav_ac.yopo.model.checkpoint import save_checkpoint, load_checkpoint
    from uav_ac.yopo.deployment import YopoPlanner
    from uav_ac.yopo.model.network import WholeBodyPlannerNet
    o=observation();n=normalization();meta={'depth_spec':o.depth_spec.as_mapping(),'candidate_count':27,'gripper_gap':o.configuration[11]}
    new=PrimitivePlannerNet(n,width=4,depth_spec=o.depth_spec)
    save_checkpoint(tmp_path/'new',new,meta)
    loaded,info=load_checkpoint(tmp_path/'new')
    assert info['profile']=='primitive_v4' and info['schema_version']==2
    assert isinstance(loaded,PrimitivePlannerNet)
    plan=YopoPlanner(tmp_path/'new').plan(o)
    for d in range(3):assert np.allclose(plan.trajectory.evaluate(0,d),o.start_pva[d],atol=1e-7)
    save_checkpoint(tmp_path/'old',WholeBodyPlannerNet(n,width=4),meta)
    old,legacy=load_checkpoint(tmp_path/'old')
    assert legacy['profile']=='direct_v3' and isinstance(old,WholeBodyPlannerNet)
    path=tmp_path/'new'/'metadata.json';info['primitive_spec']['overlap']=1.2;path.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='geometry|contract'):load_checkpoint(tmp_path/'new')


def test_independent_terminal_and_primitive_derivative_decode():
    o=observation();n=normalization();model=PrimitivePlannerNet(n,width=4,depth_spec=o.depth_spec)
    e=encode_primitive(o,n);state=torch.tensor(e.state);anchors=torch.tensor(primitive_anchors(o,n)[None],dtype=torch.float32)
    raw=torch.zeros(1,27,34);initial=model.decode(raw,state,anchors)
    raw[0,0,8:11]=torch.tensor([.6,-.3,2.]);raw[0,0,16]=.2
    changed=model.decode(raw,state,anchors)
    assert torch.equal(initial[0,1:],changed[0,1:])
    assert torch.linalg.norm(initial[0,0,8:11]-changed[0,0,8:11])>.25
    assert torch.equal(initial[0,0,:8],changed[0,0,:8])
    from uav_ac.yopo.model.primitives import local_body_rotation
    expected=local_body_rotation(state)[0]@model.primitive_rotations[0]@torch.tensor([np.tanh(.2)*3,0,0],dtype=torch.float32)
    assert torch.allclose(changed[0,0,16:19],expected,atol=1e-6)
    # Extreme offsets stay in their angular region, with overlap and camera-edge clipping.
    raw[:,:,:2]=10.;points=model.decode(raw,state,anchors)[0,:,:3]
    body=local_body_rotation(state)[0]
    optical=((points@body)-model.camera_offset)@model.camera_rotation
    angles=torch.atan(optical[:,:2]/optical[:,2:3])
    center=model.bin_angles.repeat_interleave(3,0);half=torch.atan(model.fov_tangent)
    assert ((angles-center).abs()<=half/3*model.spec.overlap+1e-6).all()


def test_guidance_gradient_through_primitive_decode_matches_finite_difference():
    from uav_ac.yopo.training.legacy import DirectEvaluator
    from uav_ac.yopo.training.losses import guidance_costs
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    o=observation();n=normalization();model=PrimitivePlannerNet(n,width=4).double()
    e=encode_primitive(o,n);state=torch.tensor(e.state,dtype=torch.float64)
    anchors=torch.tensor(primitive_anchors(o,n)[None],dtype=torch.float64)
    raw=torch.full((1,27,34),.03,dtype=torch.float64,requires_grad=True)
    sim=MujocoSimulation(scene_path('free'),planning_path_capacity=0,record_actual_trajectory=False)
    oracle=DirectEvaluator.from_simulation(sim)
    z=model.decode(raw,state,anchors)[0,12:13]
    guidance_costs(z,o,oracle).sum().backward()
    direction=torch.zeros_like(raw);direction[0,12]=torch.linspace(-.1,.1,34,dtype=torch.float64)
    def value(x):return oracle.evaluate(o,model.decode(x,state,anchors)[0,12:13].detach().numpy()).costs[0]
    fd=(value(raw.detach()+1e-6*direction)-value(raw.detach()-1e-6*direction))/2e-6
    ad=(raw.grad*direction).sum().item()
    assert abs(fd-ad)/max(1,abs(fd),abs(ad))<1e-4


def test_camera_cells_and_zero_head_match_physical_anchor_initialization():
    from dataclasses import replace
    o=observation();goal=o.goal.copy();goal[3]+=.4;o=replace(o,goal=goal);n=normalization();model=PrimitivePlannerNet(n,width=4,depth_spec=o.depth_spec)
    e=encode_primitive(o,n);anchors=primitive_anchors(o,n)
    state=torch.tensor(e.state);a=torch.tensor(anchors[None],dtype=torch.float32)
    decoded=model.decode(torch.zeros(1,27,34),state,a)[0].detach().numpy()
    assert np.allclose(decoded,anchors,atol=1e-6)
    from uav_ac.yopo.model.primitives import local_body_rotation
    body=local_body_rotation(state)[0].numpy()
    points=(anchors[::3,:3]@body-model.camera_offset.numpy())@model.camera_rotation.numpy()
    xy=points[:,:2]/points[:,2:3]/model.fov_tangent.numpy()
    assert np.allclose(xy,model.sampling_xy.numpy(),atol=1e-6)
    assert np.all(np.diff(xy.reshape(3,3,2)[:,:,0],axis=1)>0)
    assert np.all(np.diff(xy.reshape(3,3,2)[:,:,1],axis=0)>0)


def test_prediction_head_does_not_force_identical_zero_terminal_outputs():
    torch.manual_seed(23);o=observation();n=normalization()
    model=PrimitivePlannerNet(n,width=4).eval()
    assert torch.count_nonzero(model.parameter_head.weight)>0
    e=encode_primitive(o,n);a=primitive_anchors(o,n)
    output=model(torch.tensor(e.depth),torch.tensor(e.state),torch.tensor(a[None],dtype=torch.float32))
    endpoints=output.parameters[0,:,8:11]
    assert torch.linalg.vector_norm(endpoints[:,None]-endpoints[None],dim=-1).max()>1e-5


def test_whole_body_goal_terminal_can_hold_and_move_behind_camera():
    from uav_ac.yopo.model.network import GoalPrimitivePlannerNet
    o=observation();n=normalization();net=GoalPrimitivePlannerNet(n,width=4)
    e=encode_primitive(o,n,net.spec);state=torch.tensor(e.state);a=torch.tensor(primitive_anchors(o,n,net.spec)[None],dtype=torch.float32)
    raw=torch.zeros(1,27,34)
    assert torch.equal(net.decode(raw,state,a)[:,:,8:11],torch.zeros(1,27,3))
    raw[:,:,8]=-1.
    tail=net.decode(raw,state,a)[:,:,8:11]
    from uav_ac.yopo.model.primitives import local_body_rotation
    assert (torch.einsum('bni,bij->bnj',tail,local_body_rotation(state))[:,:,0]<0).all()
    assert (torch.linalg.vector_norm(tail,dim=-1)<net.spec.tail_max).all()


def test_goal_profile_serialization_preserves_v4_semantics(tmp_path):
    from uav_ac.yopo.model.network import GoalPrimitivePlannerNet
    from uav_ac.yopo.model.checkpoint import save_checkpoint, load_checkpoint
    from uav_ac.yopo.deployment import YopoPlanner
    o=observation();n=normalization();model=GoalPrimitivePlannerNet(n,width=4)
    save_checkpoint(tmp_path/'goal',model,{'candidate_count':27,'depth_spec':o.depth_spec.as_mapping(),'gripper_gap':.06})
    loaded,info=load_checkpoint(tmp_path/'goal')
    assert info['profile']=='primitive_goal_v5' and info['schema_version']==3
    assert isinstance(loaded,GoalPrimitivePlannerNet)
    plan=YopoPlanner(tmp_path/'goal').plan(o)
    for d in range(3):assert np.allclose(plan.trajectory.evaluate(0,d),o.start_pva[d],atol=1e-7)
