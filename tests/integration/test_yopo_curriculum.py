import numpy as np
from uav_ac.yopo.whole_body.data.config import DatasetConfig
from uav_ac.yopo.whole_body.data.generate import generate_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset


def test_curriculum_is_scene_disjoint_independent_current_state_data(tmp_path):
    config=DatasetConfig(profile='curriculum_v5',scene_count=12,split_counts=(4,4,4),samples_per_scene=2)
    path=generate_dataset(config,tmp_path/'data')
    data=PlanningDataset(path.parent,split='train')
    assert set(data.family_at(i) for i in range(len(data)))=={'free','obstacle','low','wall'}
    assert data.manifest['derivative_provenance'].startswith('independent')
    for o,_ in data:
        assert o.derivative_source=='augmentation'
        assert np.allclose(o.start_pva[1,:3],o.velocity[:3])
    assert all(s['path_validated'] for s in data.manifest['scenes'])


def test_warm_start_rejects_opposite_scoring_semantics(tmp_path):
    import pytest
    from uav_ac.yopo.whole_body.training.runner import TrainingConfig, train
    from uav_ac.yopo.whole_body.model.encoding import Normalization
    from uav_ac.yopo.whole_body.model.network import PrimitivePlannerNet
    from uav_ac.yopo.whole_body.model.checkpoint import save_checkpoint
    path=generate_dataset(DatasetConfig(profile='curriculum_v5',scene_count=12,split_counts=(4,4,4),samples_per_scene=1),tmp_path/'data')
    data=PlanningDataset(path.parent,split='train')
    n=Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4))
    model=PrimitivePlannerNet(n,width=4,score_mode='cost',depth_spec=data.spec)
    checkpoint=tmp_path/'cost';save_checkpoint(checkpoint,model,{'candidate_count':27,'depth_spec':data.spec.as_mapping(),'robot_hash':data.manifest['robot_sha256'],'gripper_gap':.06})
    config=TrainingConfig(profile='primitive_v4',families=('free',),candidate_count=27,width=4,steps=1,train_samples=1,validation_samples=1,batch_size=1)
    with pytest.raises(ValueError,match='warm-start.*score'):train(config,path.parent,tmp_path/'wrong',warm_start=checkpoint)
