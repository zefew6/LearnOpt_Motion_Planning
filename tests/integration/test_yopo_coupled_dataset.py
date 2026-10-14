import json
import numpy as np
from uav_ac.yopo.data.config import DatasetConfig
from uav_ac.yopo.data.generate import generate_dataset
from uav_ac.yopo.data.dataset import PlanningDataset
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state


def test_coupled_dataset_retains_validated_sources_and_compatible_observations(tmp_path):
    config=DatasetConfig(profile='coupled_v3',scene_count=6,samples_per_scene=6,split_counts=(2,2,2))
    path=generate_dataset(config,tmp_path/'coupled')
    ds=PlanningDataset(path.parent);assert len(ds)==36
    assert ds.manifest['schema_version']==2 and ds.manifest['profile']=='coupled_v3'
    assert set(s['family'] for s in ds.manifest['scenes'])=={'coupled_corridor','coupled_bamboo'}
    for scene in ds.manifest['scenes']:
        assert scene['path_validated'] and scene['source_count']==2
        source=path.parent/scene['source_diagnostics']
        diagnostics=json.loads(source.read_text())
        assert all(d['validation_passed'] and d['frozen_arm_collision_frames']>0 and d['base_collision_frames']==0 for d in diagnostics['sources'])
    for obs,_ in (ds[i] for i in range(len(ds))):
        q,v=recover_state(np.vstack((obs.start_pva,obs.start_jerk)),.06)
        assert np.allclose(q,obs.configuration,atol=1e-7)
        assert np.allclose(v,obs.velocity,atol=1e-7)
        assert obs.goal.shape==(8,) and obs.depth_m.shape==(96,160)
    # New observations feed the existing direct_v3 analytic guidance interface.
    import torch
    from uav_ac.yopo.training.runner import OracleCache
    from uav_ac.yopo.training.losses import guidance_costs
    from uav_ac.yopo.model.primitives import make_anchors
    from uav_ac.yopo.model.encoding import Normalization
    observation,privileged=ds[1];oracle=OracleCache(ds,0.).get(privileged)
    normalization=Normalization(tuple(oracle.robot.limits.joint_lower),tuple(oracle.robot.limits.joint_upper))
    parameters=torch.tensor(make_anchors(observation,normalization,count=1),requires_grad=True)
    cost=guidance_costs(parameters,observation,oracle);cost.mean().backward()
    assert torch.isfinite(cost).all() and torch.isfinite(parameters.grad).all()
    import pytest
    source_numeric=path.parent/ds.manifest['scenes'][0]['source_numeric']
    source_numeric.write_bytes(source_numeric.read_bytes()+b'corrupted witness')
    with pytest.raises(ValueError,match='source.*checksum'):PlanningDataset(path.parent)
