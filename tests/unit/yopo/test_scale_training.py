from dataclasses import replace
import numpy as np
import pytest
from uav_ac.yopo.whole_body.training.runner import TrainingConfig, subset


def test_scaling_config_and_lazy_full_subset():
    c=TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry',train_samples=0,train_evaluation_samples=16,checkpoint_interval=50,checkpoint_keep=3,
        warmup_steps=10,final_learning_rate_ratio=.1,sampling='shuffle',guidance_workers=2,
        oracle_map_resolution=.2)
    class Dataset:
        reads=0
        def __len__(self):return 21
        def family_at(self,i):return 'free'
        def __getitem__(self,i):self.reads+=1;return i
    d=Dataset();samples=subset(d,c.train_samples)
    assert len(samples)==21 and d.reads==0
    assert samples[17]==17 and d.reads==1
    from uav_ac.yopo.whole_body.training.runner import learning_rate_at, batch_indices
    assert learning_rate_at(c,0)==pytest.approx(c.learning_rate/10)
    assert learning_rate_at(c,c.steps-1)==pytest.approx(c.learning_rate*.1)
    rng=np.random.default_rng(4)
    batches=[batch_indices(21,8,k,'shuffle',4,rng) for k in range(3)]
    assert sorted(np.concatenate(batches).tolist())==list(range(21))
    assert len(batch_indices(21,8,3,'shuffle',4,rng))==8
    with pytest.raises(ValueError):replace(c,guidance_workers=-1)
    with pytest.raises(ValueError):replace(c,final_learning_rate_ratio=2.)


def test_cuda_esdf_backend_has_explicit_collision_representation():
    c=TrainingConfig(guidance_profile='yopo_minco_v1',guidance_backend='torch_esdf',collision_model='esdf_spheres')
    assert c.guidance_workers==0
    with pytest.raises(ValueError):replace(c,collision_model='exact_geometry')
    with pytest.raises(ValueError):replace(c,guidance_workers=2)
