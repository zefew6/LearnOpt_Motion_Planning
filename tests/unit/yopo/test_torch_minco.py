import numpy as np
import torch
from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic


def test_batched_torch_minco_matches_coefficients_and_gradient():
    from uav_ac.YOPO.torch_minco import solve_minco,states,energy
    rng=np.random.default_rng(6)
    head=torch.tensor(rng.normal(size=(2,3,8)),dtype=torch.float64)
    tail=torch.tensor(rng.normal(size=(2,3,8)),dtype=torch.float64)
    middle=torch.tensor(rng.normal(size=(2,8)),dtype=torch.float64,requires_grad=True)
    times=torch.tensor([[.7,1.3],[1.5,.9]],dtype=torch.float64,requires_grad=True)
    coeff=solve_minco(head,tail,middle,times)
    for i in range(2):
        reference,_=MINCOQuintic(head[i].numpy(),tail[i].numpy(),2).solve(middle[i:i+1].detach().numpy(),times[i].detach().numpy())
        assert np.allclose(coeff[i].detach().numpy().reshape(12,8),reference,rtol=1e-9,atol=1e-9)
    loss=energy(coeff,times,3,torch.ones(8,dtype=torch.float64)).sum()
    gm,gt=torch.autograd.grad(loss,(middle,times))
    assert torch.autograd.gradcheck(lambda m,t:energy(solve_minco(head,tail,m,t),t,3,torch.ones(8,dtype=torch.float64)),(middle,times),atol=1e-4,rtol=1e-4)
    sampled=states(coeff,times,17)
    assert sampled.shape==(2,34,4,8)
    assert torch.allclose(sampled[:,0,:3],head)
    assert torch.allclose(sampled[:,-1,:3],tail)
