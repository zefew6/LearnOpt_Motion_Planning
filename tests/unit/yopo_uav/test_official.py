import numpy as np
import torch


def test_official_network_contract_and_instance_isolation():
    from uav_ac.yopo.uav.core import OfficialCore
    torch.set_num_threads(2)
    first=OfficialCore(device='cpu',velocity=6.)
    second=OfficialCore(device='cpu',velocity=2.)
    assert first.network.state_transform.lattice_primitive is not second.network.state_transform.lattice_primitive
    assert first.network.state_transform.lattice_primitive.vel_max==6.
    assert second.network.state_transform.lattice_primitive.vel_max==2.
    depth=torch.ones(1,1,96,160);state=torch.tensor([[1.,0.,0.,0.,0.,0.,5.,0.,0.]])
    original=state.clone()
    with torch.no_grad():inner,tail,times,score,offset,radius=first.inference(depth,state)
    assert inner.shape==(1,15,3) and tail.shape==(1,15,3,3)
    assert times.shape==(1,15,2) and score.shape==(1,3,5) and radius.shape==(1,15,20)
    assert torch.equal(state,original)
    assert torch.isfinite(tail).all() and torch.all(times>=.1)


def test_official_numpy_minco_matches_existing_reference():
    from uav_ac.yopo.vendor.yopo_minco.policy.poly_solver import MincoTraj
    from tests.fixtures.yopo_minco_reference import build_A
    times=np.array([[.8,1.1],[1.2,.7]])
    np.testing.assert_array_equal(MincoTraj._build_A(*times.T),build_A(*times.T))
    rng=np.random.default_rng(7);head=rng.normal(size=(2,3,3));tail=rng.normal(size=(2,3,3));middle=rng.normal(size=(2,3))
    t=MincoTraj().solve(head,tail,middle,times)
    np.testing.assert_allclose(t.position(0),head[:,0],atol=1e-8)
    np.testing.assert_allclose(t.velocity(0),head[:,1],atol=1e-8)


def test_pinned_vendor_hashes_and_incompatible_checkpoint_rejection(tmp_path):
    import hashlib,json
    from pathlib import Path
    import pytest
    from uav_ac.yopo.vendor import yopo_minco
    from uav_ac.yopo.uav.checkpoint import load_checkpoint
    root=Path(yopo_minco.__file__).parent
    manifest=json.loads((root/'provenance.json').read_text())
    for name,record in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==record['local_sha256']
    simulator=root.parent/'yopo_simulator'
    sources=json.loads((simulator/'provenance.json').read_text())
    for name,record in sources['files'].items():
        assert hashlib.sha256((simulator/name).read_bytes()).hexdigest()==record['local_sha256']
    (tmp_path/'metadata.json').write_text(json.dumps({'profile':'primitive_goal_v5'}))
    with pytest.raises(ValueError,match='profile/version'):load_checkpoint(tmp_path)
    (tmp_path/'metadata.json').write_text(json.dumps({'profile':'uav_official_minco_v1','upstream_commit':manifest['commit'],'weights_sha256':'invalid'}))
    (tmp_path/'weights.pth').write_bytes(b'invalid')
    with pytest.raises(ValueError,match='checksum'):load_checkpoint(tmp_path)
