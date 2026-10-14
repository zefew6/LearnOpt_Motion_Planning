import numpy as np
from uav_ac.yopo.whole_body.data.config import DatasetConfig
from uav_ac.yopo.whole_body.data.generate import generate_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset


def test_fixed_dense_independent_states_and_bounded_shards(tmp_path):
    cfg=DatasetConfig(profile='fixed_dense_v6',scene_count=3,split_counts=(1,1,1),samples_per_scene=3)
    generate_dataset(cfg,tmp_path/'data')
    data=PlanningDataset(tmp_path/'data/dataset',max_cached_shards=1)
    obs=[data[i][0] for i in (0,1,3,6)]
    assert len(data._shards)==1
    assert np.allclose(obs[0].goal,obs[1].goal)
    assert not np.allclose(obs[0].configuration,obs[1].configuration)
    assert obs[0].derivative_source=='augmentation'
    assert data.manifest['derivative_provenance']=='independent current-state PVA; no rollout or reference sampling'
    assert not data.manifest['scenes'][0]['path_validated']
    assert not np.allclose(obs[0].depth_m,obs[1].depth_m)


def test_cylinder_bounds_are_finite_and_pruning_preserves_guidance(tmp_path):
    from uav_ac.yopo.whole_body.training.runner import OracleCache, TrainingConfig, tensors
    from uav_ac.yopo.whole_body.model.network import GoalPrimitivePlannerNet
    from uav_ac.yopo.whole_body.model.encoding import Normalization
    import torch
    generate_dataset(DatasetConfig(profile='fixed_dense_v6',scene_count=3,split_counts=(1,1,1),samples_per_scene=1),tmp_path/'data')
    data=PlanningDataset(tmp_path/'data/dataset',split='train');obs,priv=data[0]
    cfg=TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry')
    oracle=OracleCache(data,0.,guidance_profile=cfg.guidance_profile,guidance=cfg.guidance,map_resolution=.2).get(priv)
    bounds=oracle.robot.collision_environment_aabbs()
    assert np.isfinite(bounds['obstacle_bamboo_0']).all()
    norm=Normalization(tuple(oracle.robot.limits.joint_lower),tuple(oracle.robot.limits.joint_upper))
    torch.manual_seed(11);model=GoalPrimitivePlannerNet(norm,width=4)
    with torch.inference_mode():parameters=model(*tensors([obs],norm,27,'cpu',model.profile,model.spec)).parameters[0].numpy()[::9]
    pruned=oracle.evaluate(obs,parameters)
    oracle.evaluator.world_aabb_lower[:]=-np.inf;oracle.evaluator.world_aabb_upper[:]=np.inf
    exhaustive=oracle.evaluate(obs,parameters)
    assert np.allclose(pruned.costs,exhaustive.costs,rtol=1e-9,atol=1e-9)
    assert np.allclose(pruned.free_gradients,exhaustive.free_gradients,rtol=1e-9,atol=1e-9)
    assert np.array_equal(pruned.collided,exhaustive.collided)
