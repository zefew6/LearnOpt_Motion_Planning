"""The official architecture extended to whole-body inputs/outputs."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from tests.yopo_dataset import write_dataset


def test_official_whole_body_architecture_train_save_resume_deploy(tmp_path,monkeypatch):
    from uav_ac.yopo.whole_body.training.runner import TrainingConfig,train
    from uav_ac.yopo.whole_body.model.checkpoint import load_checkpoint
    from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
    from uav_ac.yopo.whole_body.deployment import YopoPlanner
    from uav_ac.yopo.vendor.yopo_minco.policy.models.resnet import ResNet
    from uav_ac.yopo.vendor.yopo_minco.policy.models.head import YopoHead
    from uav_ac.yopo.whole_body.training import runner
    original=runner.candidate_loss;batch_shapes=[]
    def candidate_loss(costs,*args,**kwargs):
        batch_shapes.append(costs.shape)
        return original(costs,*args,**kwargs)
    monkeypatch.setattr(runner,'candidate_loss',candidate_loss)
    write_dataset(tmp_path/'data',samples_per_scene=2)
    cfg=TrainingConfig(profile='official_whole_body_v1',guidance_profile='yopo_minco_v1',
        guidance_backend='torch_esdf',collision_model='esdf_spheres',width=64,candidate_count=27,
        steps=1,batch_size=2,train_samples=2,validation_samples=1,oracle_map_resolution=.2)
    report=train(cfg,tmp_path/'data/dataset',tmp_path/'train')
    assert np.isfinite(report['final_train_cost'])
    assert batch_shapes==[torch.Size([2,27])]
    model,metadata=load_checkpoint(tmp_path/'train/last')
    assert isinstance(model.backbone.cnn,ResNet) and isinstance(model.head,YopoHead)
    assert model.backbone.cnn.conv1.in_channels==2
    assert metadata['schema_version']==4 and metadata['profile']=='official_whole_body_v1'
    assert metadata['training_source']['commit']=='d5e1e60f83cc991602f142d9b6b96204a24d6b9f'
    entry=json.loads((tmp_path/'train/metrics.jsonl').read_text().splitlines()[0])
    assert set(entry['official_losses'])=={'generation','ranking','inner_regularizer','radius_nll'}
    state=torch.load(tmp_path/'train/last/training_state.pt',weights_only=True)
    names=list(dict(model.named_parameters()))
    head_id=state['optimizer']['param_groups'][0]['params'][names.index('head.model.4.weight')]
    momentum=state['optimizer']['state'][head_id]['exp_avg']
    # Real trajectory gradients reach arm, score and clearance channels.
    assert momentum[4:8].abs().sum()>0 and momentum[12:16].abs().sum()>0
    assert momentum[34:35].abs().sum()>0 and momentum[35:].abs().sum()>0
    obs,_=PlanningDataset(tmp_path/'data/dataset',split='validation')[0]
    result=YopoPlanner(tmp_path/'train/last').plan(obs)
    assert result.candidate_count==27 and result.parameters.shape==(27,34)
    for d in range(3):np.testing.assert_allclose(result.trajectory.evaluate(0,d),obs.start_pva[d],atol=1e-6)
    resumed=train(replace(cfg,steps=2),tmp_path/'data/dataset',tmp_path/'resume',resume=tmp_path/'train/last')
    assert resumed['resumed_from_step']==1
    train(replace(cfg,steps=2),tmp_path/'data/dataset',tmp_path/'continuous')
    a,_=load_checkpoint(tmp_path/'resume/last');b,_=load_checkpoint(tmp_path/'continuous/last')
    for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='profile'):
        train(replace(cfg,steps=2,profile='primitive_goal_v5'),tmp_path/'data/dataset',tmp_path/'wrong',resume=tmp_path/'train/last')
