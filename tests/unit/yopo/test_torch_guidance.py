import numpy as np
import torch
import pytest
from uav_ac.YOPO.training import OracleCache,TrainingConfig,tensors
from uav_ac.YOPO.dataset import PlanningDataset
from uav_ac.planning.end2end.goal_network import GoalPrimitivePlannerNet
from uav_ac.planning.end2end.encoding import Normalization


def test_torch_physical_terms_match_cpu_and_esdf_backward(tmp_path):
    from uav_ac.YOPO.torch_guidance import TorchGuidance
    from uav_ac.YOPO.generate import generate_dataset
    from uav_ac.YOPO.config import DatasetConfig
    generate_dataset(DatasetConfig(profile='rolling_v2',scene_count=3,split_counts=(1,1,1),samples_per_scene=1),tmp_path/'data')
    data=PlanningDataset(tmp_path/'data/dataset',split='train');obs,priv=data[0]
    cfg=TrainingConfig(guidance_profile='yopo_minco_v1',collision_model='exact_geometry')
    oracle=OracleCache(data,0.,guidance_profile=cfg.guidance_profile,guidance=cfg.guidance,map_resolution=.1).get(priv)
    backend=TorchGuidance(oracle,'cpu',dtype=torch.float64)
    values=torch.tensor(np.vstack((obs.start_pva,np.zeros((1,8)))),dtype=torch.float64)[None]
    peak=backend.physical_residuals(values)
    expected,_=oracle._physical_residuals(values[0].numpy())
    assert np.allclose(peak.detach().numpy()[0],expected,atol=1e-8)
    norm=Normalization(tuple(oracle.robot.limits.joint_lower),tuple(oracle.robot.limits.joint_upper));torch.manual_seed(41)
    model=GoalPrimitivePlannerNet(norm,width=4)
    with torch.inference_mode():z=model(*tensors([obs],norm,27,'cpu',model.profile,model.spec)).parameters[:,:1].double()
    z=z.clone().requires_grad_()
    result=backend.evaluate(z,[obs]);assert torch.isfinite(result.costs).all()
    gradient=torch.autograd.grad(result.costs.sum(),z)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    # Compare the actual full cost derivative, including deliberately detached
    # goal/curvature contributions exposed separately for FD auditing.
    assert torch.autograd.gradcheck(lambda p:backend.evaluate(p,[obs],detach_guidance=False).costs,(z,),eps=1e-5,atol=2e-3,rtol=2e-3)
    if torch.cuda.is_available():
        cuda=TorchGuidance(oracle,'cuda:0',dtype=torch.float64)
        zg=z.detach().to('cuda:0').requires_grad_();rg=cuda.evaluate(zg,[obs]);gg=torch.autograd.grad(rg.costs.sum(),zg)[0]
        assert np.allclose(rg.costs.detach().cpu(),result.costs.detach(),rtol=1e-6,atol=1e-6)
        assert np.allclose(gg.cpu(),gradient,rtol=1e-5,atol=1e-5)
