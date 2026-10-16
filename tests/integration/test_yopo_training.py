import json
import numpy as np

from tests.yopo_dataset import write_dataset
from uav_ac.yopo.whole_body.training.runner import TrainingConfig, train
from uav_ac.yopo.whole_body.model.checkpoint import load_checkpoint
from uav_ac.yopo.whole_body.deployment import YopoPlanner
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset


def test_train_save_resume_and_map_free_prediction(tmp_path):
    root=tmp_path/'data';write_dataset(root, samples_per_scene=2)
    dataset=root/'dataset';config=TrainingConfig(steps=2,train_samples=2,validation_samples=1,batch_size=1,width=4)
    result=train(config,dataset,tmp_path/'train')
    assert np.isfinite(result['final_train_cost'])
    checkpoint=tmp_path/'train'/'last';model,metadata=load_checkpoint(checkpoint)
    assert metadata['candidate_count']==1
    from uav_ac.yopo.whole_body.validation.evaluation import compare
    metrics=compare(dataset,[checkpoint],samples=1)['models'][str(checkpoint)]
    assert metrics['selected_cost']==metrics['random_cost']
    observation,_=PlanningDataset(dataset,split='train')[0]
    planner=YopoPlanner(checkpoint);prediction=planner.plan(observation)
    assert prediction.trajectory is not None and prediction.candidate_count==1
    resumed=train(TrainingConfig(steps=3,train_samples=2,validation_samples=1,batch_size=1,width=4),dataset,
                  tmp_path/'resume',resume=checkpoint)
    assert resumed['completed_steps']==3
    assert resumed['resumed_from_step']==2
    import pytest
    import torch
    with pytest.raises(ValueError,match='resume steps'):
        train(config,dataset,tmp_path/'backward',resume=checkpoint)
    train(TrainingConfig(steps=3,train_samples=2,validation_samples=1,batch_size=1,width=4),dataset,tmp_path/'continuous')
    continuous,_=load_checkpoint(tmp_path/'continuous'/'last')
    resumed_model,_=load_checkpoint(tmp_path/'resume'/'last')
    for name,value in continuous.state_dict().items():
        assert torch.equal(value,resumed_model.state_dict()[name])


def test_ranking_training_and_gripper_contract(tmp_path):
    from dataclasses import replace
    import pytest
    root=tmp_path/'data';write_dataset(root, samples_per_scene=1)
    dataset=root/'dataset';config=TrainingConfig(steps=1,train_samples=1,validation_samples=1,batch_size=1,width=4,candidate_count=27)
    result=train(config,dataset,tmp_path/'train')
    assert result['candidate_count']==27 and np.isfinite(result['validation']['selected_cost'])
    observation,_=PlanningDataset(dataset,split='validation')[0]
    planner=YopoPlanner(tmp_path/'train'/'last')
    prediction=planner.plan(observation)
    assert prediction.candidate_count==27
    assert prediction.selected_index==int(np.argmax(prediction.scores))
    changed=observation.configuration.copy();changed[11]+=.01
    with pytest.raises(ValueError,match='gripper'):planner.plan(replace(observation,configuration=changed))
    with pytest.raises(ValueError,match='resume'):
        train(replace(config,steps=2,peak_weight=1.),dataset,tmp_path/'incompatible',resume=tmp_path/'train'/'last')


def test_primitive_training_save_load_deploy_and_profile_resume_guard(tmp_path):
    import pytest
    root=tmp_path/'data'
    write_dataset(root, samples_per_scene=1)
    config=TrainingConfig(profile='primitive_v4',steps=1,train_samples=1,validation_samples=1,batch_size=1,width=4,candidate_count=27)
    report=train(config,root/'dataset',tmp_path/'primitive')
    assert np.isfinite(report['final_train_cost'])
    model,meta=load_checkpoint(tmp_path/'primitive'/'last')
    assert meta['profile']=='primitive_v4'
    o,_=PlanningDataset(root/'dataset',split='validation')[0]
    output=YopoPlanner(tmp_path/'primitive'/'last').plan(o)
    assert output.candidate_count==27 and np.isfinite(output.parameters).all()
    from dataclasses import replace
    with pytest.raises(ValueError,match='resume profile'):
        train(replace(config,profile='direct_v3',steps=2),root/'dataset',tmp_path/'wrong',resume=tmp_path/'primitive'/'last')


def test_primitive_resume_keeps_saved_observation_normalization(tmp_path,monkeypatch):
    from dataclasses import replace
    import shutil
    import torch
    from uav_ac.yopo.whole_body.training import runner as training
    from uav_ac.yopo.whole_body.model.network import PrimitivePlannerNet
    from uav_ac.yopo.whole_body.model.checkpoint import save_checkpoint
    root=tmp_path/'data'
    write_dataset(root, samples_per_scene=1)
    config=TrainingConfig(profile='primitive_v4',steps=1,train_samples=1,validation_samples=1,batch_size=1,width=4,candidate_count=27)
    train(config,root/'dataset',tmp_path/'initial')
    old,meta=load_checkpoint(tmp_path/'initial'/'last')
    custom=PrimitivePlannerNet(old.normalization,width=4,depth_spec=old.depth_spec,spec=replace(old.spec,goal_clip=5.,max_speed=1.5))
    old_params=dict(old.named_parameters())
    with torch.no_grad():
        for name,value in custom.named_parameters():value.copy_(old_params[name])
    save_checkpoint(tmp_path/'custom',custom,meta)
    shutil.copyfile(tmp_path/'initial'/'last'/'training_state.pt',tmp_path/'custom'/'training_state.pt')
    original=training.tensors;seen=[]
    def tensors(*args,**kwargs):
        if torch.is_grad_enabled():
            spec=kwargs.get('primitive_spec',args[5] if len(args)>5 else None)
            seen.append(spec)
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'tensors',tensors)
    train(replace(config,steps=2),root/'dataset',tmp_path/'resumed',resume=tmp_path/'custom')
    assert seen and all(spec is not None and spec.goal_clip==5. and spec.max_speed==1.5 for spec in seen)
