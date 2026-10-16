import numpy as np
from tests.yopo_dataset import write_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset


def test_cylinder_bounds_are_finite_and_pruning_preserves_guidance(tmp_path):
    from uav_ac.yopo.whole_body.training.runner import OracleCache, TrainingConfig, tensors
    from uav_ac.yopo.whole_body.model.network import GoalPrimitivePlannerNet
    from uav_ac.yopo.whole_body.model.encoding import Normalization
    import torch
    write_dataset(tmp_path/'data', samples_per_scene=1, cylinder=True)
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
