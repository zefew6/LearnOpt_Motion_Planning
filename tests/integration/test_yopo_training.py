import json
import numpy as np
from uav_ac.YOPO.config import DatasetConfig
from uav_ac.YOPO.generate import generate_dataset
from uav_ac.YOPO.training import TrainingConfig,train
from uav_ac.planning.end2end.checkpoint import load_checkpoint
from uav_ac.planning.end2end.planner import End2EndPlanner
from uav_ac.YOPO.dataset import PlanningDataset


def test_train_save_resume_and_map_free_prediction(tmp_path):
    root=tmp_path/'data';generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=2,split_counts=(1,1,1)),root)
    dataset=root/'dataset';config=TrainingConfig(steps=2,train_samples=2,validation_samples=1,batch_size=1,width=4)
    result=train(config,dataset,tmp_path/'train')
    assert np.isfinite(result['final_train_cost'])
    checkpoint=tmp_path/'train'/'last';model,metadata=load_checkpoint(checkpoint)
    assert metadata['candidate_count']==1
    from uav_ac.YOPO.evaluation import compare
    metrics=compare(dataset,[checkpoint],samples=1)['models'][str(checkpoint)]
    assert metrics['selected_cost']==metrics['random_cost']
    observation,_=PlanningDataset(dataset,split='train')[0]
    planner=End2EndPlanner(checkpoint);prediction=planner.plan(observation)
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
    root=tmp_path/'data';generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=1,split_counts=(1,1,1)),root)
    dataset=root/'dataset';config=TrainingConfig(steps=1,train_samples=1,validation_samples=1,batch_size=1,width=4,candidate_count=27)
    result=train(config,dataset,tmp_path/'train')
    assert result['candidate_count']==27 and np.isfinite(result['validation']['selected_cost'])
    observation,_=PlanningDataset(dataset,split='validation')[0]
    planner=End2EndPlanner(tmp_path/'train'/'last')
    prediction=planner.plan(observation)
    assert prediction.candidate_count==27
    assert prediction.selected_index==int(np.argmax(prediction.scores))
    changed=observation.configuration.copy();changed[11]+=.01
    with pytest.raises(ValueError,match='gripper'):planner.plan(replace(observation,configuration=changed))
    with pytest.raises(ValueError,match='resume'):
        train(replace(config,steps=2,peak_weight=1.),dataset,tmp_path/'incompatible',resume=tmp_path/'train'/'last')
