"""Real official loss backward, checkpoint deployment and exact resume on CPU."""
from dataclasses import replace
import json

import torch
import numpy as np
import pytest

from tests.unit.yopo_uav.test_official_training import write_official_dataset


def test_official_uav_train_resume_and_deployment_checkpoint(tmp_path,monkeypatch):
    from uav_ac.yopo.uav.training import runner
    from uav_ac.yopo.uav.checkpoint import load_checkpoint
    root=write_official_dataset(tmp_path/'dataset')
    monkeypatch.setattr(runner,'RUNS_ROOT',tmp_path)
    cfg=runner.TrainingConfig(epochs=1,batch_size=2,device='cpu',workers=0,threads=2,val_ratio=.25)
    report=runner.train(cfg,root,tmp_path/'train')
    assert report['completed_epochs']==1 and report['optimizer_steps']==2
    assert np.isfinite(report['validation']['mean_traj_loss'])
    core,meta=load_checkpoint(tmp_path/'train/last')
    assert meta['training_source']['commit']==core.config.snapshot().get('upstream_commit',meta['upstream_commit'])
    assert (tmp_path/'train/last/training_state.pt').exists()
    with torch.no_grad():
        outputs=core.inference(torch.ones(1,1,96,160),torch.tensor([[1.,0.,0.,0.,0.,0.,5.,0.,0.]]))
    assert outputs[0].shape==(1,15,3) and all(torch.isfinite(x).all() for x in outputs)
    resumed=runner.train(replace(cfg,epochs=2),root,tmp_path/'resume',resume=tmp_path/'train/last')
    assert resumed['resumed_from_epoch']==1 and resumed['optimizer_steps']==4
    runner.train(replace(cfg,epochs=2),root,tmp_path/'continuous')
    a,_=load_checkpoint(tmp_path/'resume/last');b,_=load_checkpoint(tmp_path/'continuous/last')
    for name,value in a.network.state_dict().items():torch.testing.assert_close(value,b.network.state_dict()[name],rtol=0,atol=0)
    with pytest.raises(FileExistsError):runner.train(cfg,root,tmp_path/'train')
    with pytest.raises(ValueError,match='resume'):
        runner.train(replace(cfg,epochs=2,batch_size=3),root,tmp_path/'bad',resume=tmp_path/'train/last')


def test_prepare_only_checks_data_without_constructing_network(tmp_path,monkeypatch):
    from uav_ac.yopo.uav.training import runner
    root=write_official_dataset(tmp_path/'dataset')
    monkeypatch.setattr(runner,'RUNS_ROOT',tmp_path)
    def fail(*a,**k):raise AssertionError('prepare-only must not build a network or ESDF')
    monkeypatch.setattr(runner,'YopoTrainer',fail)
    result=runner.train(runner.TrainingConfig(device='cpu'),root,tmp_path/'prepare',prepare_only=True)
    assert result['training_samples']==3 and result['validation_samples']==1
    assert not (tmp_path/'prepare/last').exists()
