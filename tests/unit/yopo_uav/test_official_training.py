"""Pinned upstream training math and explicit local data/runner contracts."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def write_official_dataset(root, *, maps=1, frames=4):
    root.mkdir(parents=True)
    for mid in range(maps):
        folder=root/str(mid);folder.mkdir()
        poses=np.tile([0.,0.,1.,1.,0.,0.,0.],(frames,1))
        poses[:,0]=np.arange(frames)*.1
        np.savetxt(root/f'pose-{mid}.csv',poses,delimiter=',',header='x,y,z,qw,qx,qy,qz',comments='')
        for i in range(frames):
            pixels=(np.arange(90*160).reshape(90,160)+1000+i).astype(np.uint16)
            Image.fromarray(pixels).save(folder/f'img_{i}.png')
        points=np.array([[-2,-2,0],[12,2,4],[4,0,1],[4,.2,1]])
        (root/f'pointcloud-{mid}.ply').write_text(
            'ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+
            '\n'.join(' '.join(map(str,p)) for p in points)+'\n')
    return root


def test_candidate_loss_matches_official_formula_and_detached_paths():
    from uav_ac.yopo.vendor.yopo_minco.loss.selection_loss import candidate_loss, radius_loss
    costs=torch.tensor([[1.,3.,2.],[8.,6.,5.]],requires_grad=True)
    scores=torch.tensor([[.3,.6,.2],[.4,.1,.5]],requires_grad=True)
    collided=torch.tensor([[True,False,False],[True,True,True]])
    inner=torch.tensor([[.1,.2,.3],[-.1,-.2,-.3]],requires_grad=True)
    normalized_goal=torch.tensor([.5,1.])
    loss,rank,reg=candidate_loss(costs,scores,collided,inner,normalized_goal,rank_weight=1.,inner_reg_weight=3.)
    target=torch.tensor([2,2])
    expected_rank=torch.nn.functional.cross_entropy(scores,target)
    expected_reg=((inner-(normalized_goal-1)[:,None])**2)[~collided].mean()
    expected_gen=((costs/costs.detach().mean(1,keepdim=True))*scores.detach().clamp(min=.001)).mean()
    torch.testing.assert_close(loss,expected_gen+expected_rank+3*expected_reg)
    torch.testing.assert_close(rank,expected_rank);torch.testing.assert_close(reg,expected_reg)
    loss.backward()
    torch.testing.assert_close(costs.grad,scores.detach().clamp(min=.001)/costs.detach().mean(1,keepdim=True)/6)
    assert torch.isfinite(scores.grad).all()
    pred=torch.tensor([[.2,.7,.1,.2]],requires_grad=True)
    dist=torch.tensor([[1.,2.]],requires_grad=True)
    radius_loss(pred,dist.detach(),warp_lambda=2.,over_weight=3.).backward()
    assert torch.isfinite(pred.grad).all() and dist.grad is None


def test_official_loader_matches_split_resize_and_resamples_states(tmp_path):
    from uav_ac.yopo.uav.training.data import load_datasets
    root=write_official_dataset(tmp_path/'data')
    train,valid=load_datasets(root,val_ratio=.25)
    assert len(train)==3 and len(valid)==1
    # sklearn ShuffleSplit(random_state=0): validation takes the permutation's first row.
    order=np.random.RandomState(0).permutation(4)
    assert [Path(p).stem for p in train.img_list]==[f'img_{i}' for i in order[1:]]
    assert not set(train.img_list)&set(valid.img_list)
    np.random.seed(9);first=train[0];second=train[0]
    assert first[0].shape==(1,96,160) and first[3].shape==(9,)
    raw=np.asarray(Image.open(train.img_list[0]))
    expected=raw[(np.arange(96)*90//96)[:,None],np.arange(160)[None,:]]/65535.
    np.testing.assert_allclose(first[0][0],expected,rtol=1e-7)
    assert not np.array_equal(first[3],second[3])
    assert np.linalg.norm(first[3][:3])<7.2 and first[4]==0
    (root/'pose-0.csv').write_text('x,y,z,qw,qx,qy,qz\n0,0,1,1,0,0,0\n')
    with pytest.raises(ValueError,match='pose|image'):load_datasets(root)


def test_training_config_rejects_unknown_and_invalid_settings(tmp_path):
    from uav_ac.yopo.uav.training.runner import TrainingConfig,load_training_config
    for kwargs in ({'epochs':0},{'batch_size':True},{'val_ratio':0.},{'device':'auto'},{'learning_rate':float('nan')}):
        with pytest.raises(ValueError):TrainingConfig(**kwargs)
    file=tmp_path/'bad.yaml';file.write_text('epochs: 1\nepochs: 2\n')
    with pytest.raises(ValueError):load_training_config(file)
    file.write_text('not_official: 4\n')
    with pytest.raises(ValueError):load_training_config(file)


def test_complete_loss_matches_unmodified_pinned_reference(tmp_path):
    from uav_ac.yopo.vendor.yopo_minco.config.config import Config,configuration
    from uav_ac.yopo.vendor.yopo_minco.loss.loss_function import YOPOLoss
    reference=json.loads(Path('tests/fixtures/yopo_uav_training_reference.json').read_text())
    root=write_official_dataset(tmp_path/'data')
    config=Config(train=True);config['dataset_path']=str(root)
    with configuration(config):loss=YOPOLoss()
    def batch(name,requires_grad=False):
        value=torch.tensor(reference[name],dtype=torch.float32)
        return value.repeat(15,*([1]*value.ndim)).requires_grad_(requires_grad)
    head=batch('head');tail=batch('tail',True);inner=batch('inner',True);times=batch('durations',True)
    costs,aux=loss(head,tail,inner,batch('goal'),torch.tensor([0]),times)
    gradients=torch.autograd.grad(sum(costs.values()).sum(),(tail,inner,times))
    for key,values in costs.items():
        torch.testing.assert_close(values,torch.full_like(values,reference['costs'][key]),rtol=1e-6,atol=1e-6)
    for key,value in zip(('tail','inner','durations'),gradients):
        expected=torch.tensor(reference['gradients'][key]).expand_as(value)
        torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-6)
    assert bool(aux['collided'][0])==reference['collided']
    torch.testing.assert_close(aux['radius_dist'][0],torch.tensor(reference['radius_dist']),rtol=0,atol=0)


def test_canonical_recipes_and_uav_cli():
    import subprocess,sys
    from dataclasses import replace
    from uav_ac.yopo.uav.training.runner import load_training_config
    from uav_ac.yopo.whole_body.training.runner import load_training_config as load_body
    assert load_training_config('configs/yopo/train/yopo_uav.yaml').epochs==50
    config=load_body('configs/yopo/train/yopo_manipulator.yaml')
    assert config.profile=='official_whole_body_v1' and config.candidate_count==27
    with pytest.raises(ValueError,match='official whole-body'):replace(config,width=8)
    with pytest.raises(ValueError):replace(config,guidance_backend='exact_cpu',collision_model='exact_geometry')
    result=subprocess.run([sys.executable,'-m','uav_ac.yopo.uav.training','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--prepare-only' in result.stdout and '--resume' in result.stdout


def test_unavailable_cuda_and_run_path_fail_before_writes(tmp_path,monkeypatch):
    from uav_ac.yopo.uav.training import runner
    monkeypatch.setattr(runner,'RUNS_ROOT',tmp_path/'runs')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    target=tmp_path/'runs/train'
    with pytest.raises(ValueError,match='CUDA'):
        runner.train(runner.TrainingConfig(),tmp_path/'unused',target)
    assert not target.exists()
    with pytest.raises(ValueError,match='beneath runs'):
        runner.train(runner.TrainingConfig(device='cpu'),tmp_path/'unused',tmp_path/'outside')
