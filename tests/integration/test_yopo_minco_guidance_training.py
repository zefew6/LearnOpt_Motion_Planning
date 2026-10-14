from dataclasses import replace
import numpy as np
import pytest
from uav_ac.yopo.whole_body.training.runner import TrainingConfig, train, OracleCache
from uav_ac.yopo.whole_body.data.config import DatasetConfig
from uav_ac.yopo.whole_body.data.generate import generate_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from uav_ac.yopo.whole_body.model.checkpoint import load_checkpoint
from uav_ac.yopo.whole_body.deployment import YopoPlanner


def test_new_guidance_training_checkpoint_resume_and_actual_state_decode(tmp_path):
    generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=1,split_counts=(1,1,1)),tmp_path/'data')
    dataset=tmp_path/'data/dataset'
    config=TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry',profile='primitive_v4',width=4,candidate_count=27,steps=1,train_samples=1,validation_samples=1,batch_size=1)
    report=train(config,dataset,tmp_path/'train');assert np.isfinite(report['final_train_cost'])
    checkpoint=tmp_path/'train/last';model,metadata=load_checkpoint(checkpoint)
    assert metadata['training_config']['guidance_profile']=='yopo_minco_v1'
    assert metadata['training_config']['goal_stop_weight']==0.
    assert metadata['training_config']['guidance']==config.guidance
    data=PlanningDataset(dataset,split='validation');obs,priv=data[0];plan=YopoPlanner(checkpoint).plan(obs)
    assert plan.selected_index==np.argmax(plan.scores)
    for d in range(3):assert np.allclose(plan.trajectory.evaluate(0,d),obs.start_pva[d])
    cache=OracleCache(data,0.,guidance_profile='yopo_minco_v1',guidance=config.guidance)
    evaluation=cache.get(priv).evaluate(obs,plan.parameters)
    assert evaluation.collided.shape==(27,)
    assert 'goal_stop' not in evaluation.components and 'continuity_and_peak' not in evaluation.components
    resumed=train(replace(config,steps=2),dataset,tmp_path/'resume',resume=checkpoint)
    assert resumed['resumed_from_step']==1
    with pytest.raises(ValueError,match='resume training setting mismatch'):
        train(replace(config,steps=2,guidance={'safety_beta':3.}),dataset,tmp_path/'bad',resume=checkpoint)


def test_parallel_guidance_matches_serial_with_coarse_auxiliary_map(tmp_path):
    import json
    import torch
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from uav_ac.yopo.whole_body.training.runner import _initialize_guidance_worker, _evaluate_guidance_worker
    from uav_ac.yopo.whole_body.training.losses import guidance_costs_from_result
    generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=1,split_counts=(1,1,1)),tmp_path/'data')
    data=PlanningDataset(tmp_path/'data/dataset',split='train')
    obs,priv=data[0]
    config=TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry',oracle_map_resolution=.2,guidance_workers=2)
    from uav_ac.yopo.whole_body.model.primitives import make_anchors
    from uav_ac.yopo.whole_body.model.encoding import Normalization
    oracle=OracleCache(data,0.,guidance_profile=config.guidance_profile,guidance=config.guidance).get(priv)
    parameters=make_anchors(obs,Normalization(tuple(oracle.robot.limits.joint_lower),tuple(oracle.robot.limits.joint_upper)),count=1)
    fine=OracleCache(data,0.,guidance_profile=config.guidance_profile,guidance=config.guidance).get(priv).evaluate(obs,parameters)
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize_guidance_worker,initargs=(data.manifest,config)) as pool:
        coarse=list(pool.map(_evaluate_guidance_worker,[(obs,priv,parameters)]))[0]
    assert np.allclose(coarse.costs,fine.costs,rtol=1e-10,atol=1e-10)
    assert np.allclose(coarse.free_gradients,fine.free_gradients,rtol=1e-10,atol=1e-10)
    p=torch.tensor(parameters,requires_grad=True)
    guidance_costs_from_result(p,coarse,scale=2.).sum().backward()
    assert np.allclose(p.grad.numpy(),fine.free_gradients/2)
    c=replace(config,steps=2,candidate_count=27,profile='primitive_goal_v5',width=4,train_samples=0,
        validation_samples=1,batch_size=2,checkpoint_interval=1,checkpoint_keep=1,train_evaluation_samples=1,sampling='shuffle')
    report=train(c,data.path,tmp_path/'parallel')
    assert report['training_samples']==1
    assert (tmp_path/'parallel/latest/training_state.pt').exists()
    assert not (tmp_path/'parallel/step_000001').exists()
    resumed=train(replace(c,steps=3),data.path,tmp_path/'resumed',resume=tmp_path/'parallel/latest')
    assert resumed['resumed_from_step']==2


def test_torch_esdf_cuda_train_resume_and_deploy(tmp_path):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=2,split_counts=(1,1,1)),tmp_path/'data')
    data=tmp_path/'data/dataset'
    cfg=TrainingConfig(profile='primitive_goal_v5',guidance_profile='yopo_minco_v1',guidance_backend='torch_esdf',collision_model='esdf_spheres',
        device='cuda:0',steps=2,width=8,candidate_count=27,batch_size=2,train_samples=0,train_evaluation_samples=1,validation_samples=1,oracle_map_resolution=.1,checkpoint_interval=1,sampling='shuffle')
    report=train(cfg,data,tmp_path/'cuda')
    assert np.isfinite(report['final_train_cost'])
    model,metadata=load_checkpoint(tmp_path/'cuda/latest',device='cuda:0')
    assert metadata['training_config']['collision_model']=='esdf_spheres'
    assert metadata['training_config']['guidance_backend']=='torch_esdf'
    obs,_=PlanningDataset(data,split='validation')[0]
    plan=YopoPlanner(tmp_path/'cuda/latest',device='cuda:0').plan(obs)
    for d in range(3):assert np.allclose(plan.trajectory.evaluate(0,d),obs.start_pva[d])
    resumed=train(replace(cfg,steps=3),data,tmp_path/'resume-cuda',resume=tmp_path/'cuda/latest')
    assert resumed['resumed_from_step']==2
