import runpy
import numpy as np
import pytest
import torch
from uav_ac.yopo.model.primitives import make_anchors, parameter_scales
from uav_ac.yopo.model.network import WholeBodyPlannerNet
from uav_ac.yopo.trajectory.minco import decode_direct
from uav_ac.yopo.model.encoding import Normalization, encode_observation
from uav_ac.yopo.model.checkpoint import save_checkpoint, load_checkpoint


def obs():return runpy.run_path('tests/unit/yopo/model/test_rolling.py')['moving_observation']()

def norm():return Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4))


def test_anchors_and_network_have_fixed_semantics_and_equivariant_candidate_order():
    torch.set_num_threads(2);o=obs();n=norm();anchors=make_anchors(o,n,count=27)
    assert anchors.shape==(27,34)
    assert len(np.unique(anchors[:,:3],axis=0))==9
    assert len(np.unique(anchors[:,4:8],axis=0))==3
    e=encode_observation(o,n);net=WholeBodyPlannerNet(n,score_mode='ranking')
    net.eval();depth=torch.tensor(e.depth);state=torch.tensor(e.state);a=torch.tensor(anchors[None],dtype=torch.float32)
    result=net(depth,state,a);assert result.parameters.shape==(1,27,34)
    assert result.scores.shape==(1,27) and torch.isfinite(result.parameters).all()
    p=torch.randperm(27);permuted=net(depth,state,a[:,p])
    assert torch.allclose(result.parameters[:,p],permuted.parameters,atol=1e-6)
    assert torch.allclose(result.scores[:,p],permuted.scores,atol=1e-6)
    assert (result.parameters[:,:,-2:]>=.1).all() and (result.parameters[:,:,-2:]<=3.).all()


def test_direct_decoder_accepts_actual_state_model_residual():
    from dataclasses import replace
    o=obs();v=o.velocity.copy();v[3]+=.15;o=replace(o,velocity=v)
    z=make_anchors(o,norm(),count=1)[0];tr=decode_direct(z,o)
    assert tr.coefficients.shape==(2,6,8)
    for d in range(3):assert np.allclose(tr.evaluate(0,d),o.start_pva[d],atol=1e-8)


def test_checkpoint_fresh_load_and_metadata_contract(tmp_path):
    o=obs();n=norm();net=WholeBodyPlannerNet(n)
    metadata={'robot_hash':'test-model','depth_spec':o.depth_spec.as_mapping(),'candidate_count':1,'cost_scale':1.}
    save_checkpoint(tmp_path/'run',net,metadata)
    loaded,info=load_checkpoint(tmp_path/'run',expected_robot_hash='test-model')
    assert info['profile']=='direct_v3' and info['parameter_dim']==34
    for name,value in net.state_dict().items():assert torch.equal(value,loaded.state_dict()[name])
    with pytest.raises(ValueError,match='robot'):load_checkpoint(tmp_path/'run',expected_robot_hash='wrong')
    with pytest.raises(FileExistsError):save_checkpoint(tmp_path/'run',net,metadata)


def test_checkpoint_rejects_conflicting_geometry_buffers(tmp_path):
    import json
    o=obs();net=WholeBodyPlannerNet(norm())
    save_checkpoint(tmp_path/'run',net,{'depth_spec':o.depth_spec.as_mapping(),'candidate_count':1})
    path=tmp_path/'run'/'metadata.json';info=json.loads(path.read_text())
    info['normalization']['joint_lower'][0]-=.1;path.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='geometry'):load_checkpoint(tmp_path/'run')


def test_deployment_import_has_no_training_or_map_dependencies():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import sys; from uav_ac.yopo.deployment import YopoPlanner; assert 'uav_ac.yopo.training.runner' not in sys.modules; assert 'uav_ac.yopo.training.legacy' not in sys.modules; assert 'uav_ac.yopo.training.legacy' not in sys.modules"],check=True)


def test_camera_rays_sample_stride32_feature_centers():
    from uav_ac.yopo.model.network import camera_feature_grid
    # Pixel-index 64,32 is the center of output feature index 2,1.
    xy=torch.tensor([[[2*(64+.5)/160-1,2*(32+.5)/96-1],[0.,0.]]])
    grid=camera_feature_grid(xy,(96,160),(3,5))
    assert torch.allclose(grid[0,0],torch.zeros(2),atol=1e-6)
    assert torch.allclose(grid[0,1],torch.tensor([.2421875,.484375]),atol=1e-6)


def test_fixed_gripper_policy_accepts_small_measured_servo_error(tmp_path):
    from dataclasses import replace
    from uav_ac.yopo.deployment import YopoPlanner
    o=obs();net=WholeBodyPlannerNet(norm())
    save_checkpoint(tmp_path/'policy',net,{'depth_spec':o.depth_spec.as_mapping(),'candidate_count':1,'gripper_gap':float(o.configuration[11])})
    changed=o.configuration.copy();changed[11]+=.0001
    prediction=YopoPlanner(tmp_path/'policy').plan(replace(o,configuration=changed))
    assert prediction.trajectory is not None


def test_speed_scaling_keeps_actual_head_and_candidate_selection(tmp_path):
    from uav_ac.yopo.deployment import YopoPlanner
    o=obs();net=WholeBodyPlannerNet(norm(),score_mode='ranking')
    path=tmp_path/'policy'
    save_checkpoint(path,net,{'depth_spec':o.depth_spec.as_mapping(),'candidate_count':27,'gripper_gap':float(o.configuration[11])})
    normal=YopoPlanner(path).plan(o);faster=YopoPlanner(path,speed_scale=1.3).plan(o)
    assert normal.selected_index==faster.selected_index
    assert np.allclose(normal.parameters[:,:16],faster.parameters[:,:16])
    assert faster.trajectory.total_time==pytest.approx(normal.trajectory.total_time/1.3)
    for d in range(3):assert np.allclose(faster.trajectory.evaluate(0,d),o.start_pva[d],atol=1e-7)
    assert faster.observation is o
    for value in (True,0.,2.1,float('nan')):
        with pytest.raises(ValueError,match='speed_scale'):YopoPlanner(path,speed_scale=value)
