from dataclasses import replace
import numpy as np
import pytest
from uav_ac.YOPO.training import TrainingConfig,train,OracleCache
from uav_ac.YOPO.config import DatasetConfig
from uav_ac.YOPO.generate import generate_dataset
from uav_ac.YOPO.dataset import PlanningDataset
from uav_ac.planning.end2end.checkpoint import load_checkpoint
from uav_ac.planning.end2end.planner import End2EndPlanner


def test_new_guidance_training_checkpoint_resume_and_actual_state_decode(tmp_path):
    generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=1,split_counts=(1,1,1)),tmp_path/'data')
    dataset=tmp_path/'data/dataset'
    config=TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry',profile='primitive_v4',width=4,candidate_count=27,steps=1,train_samples=1,validation_samples=1,batch_size=1)
    report=train(config,dataset,tmp_path/'train');assert np.isfinite(report['final_train_cost'])
    checkpoint=tmp_path/'train/last';model,metadata=load_checkpoint(checkpoint)
    assert metadata['training_config']['guidance_profile']=='yopo_minco_v1'
    assert metadata['training_config']['goal_stop_weight']==0.
    assert metadata['training_config']['guidance']==config.guidance
    data=PlanningDataset(dataset,split='validation');obs,priv=data[0];plan=End2EndPlanner(checkpoint).plan(obs)
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
