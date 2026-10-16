"""Train the pinned official pure-UAV model on native PNG/pose/point-cloud data."""
import argparse
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from uav_ac.yopo.common.collection import RUNS_ROOT
from uav_ac.yopo.common.config import _UniqueLoader
from uav_ac.yopo.vendor.yopo_minco.config.config import Config,configuration
from uav_ac.yopo.vendor.yopo_minco.policy.yopo_trainer import YopoTrainer
from uav_ac.yopo.uav.core import UPSTREAM_COMMIT
from uav_ac.yopo.uav.checkpoint import load_checkpoint,save_checkpoint
from .data import load_datasets,dataset_fingerprint


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1.5e-4
    device: str = 'cuda:0'
    seed: int = 0
    workers: int = 0
    threads: int = 2
    val_ratio: float = .1
    checkpoint_interval: int = 5

    def __post_init__(self):
        for name in ('epochs','batch_size','threads','checkpoint_interval','seed','workers'):
            value=getattr(self,name);lower=0 if name in ('seed','workers') else 1
            if isinstance(value,bool) or not isinstance(value,int) or value<lower:raise ValueError(f'invalid {name}')
        for name in ('learning_rate','val_ratio'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:raise ValueError(f'invalid {name}')
        if self.val_ratio>=1:raise ValueError('val_ratio must lie in (0,1)')
        if self.device!='cpu' and (not isinstance(self.device,str) or not self.device.startswith('cuda:') or not self.device[5:].isdigit()):
            raise ValueError('device must be cpu or cuda:<index>')


def load_training_config(path):
    values=yaml.load(Path(path).read_text(),Loader=_UniqueLoader)
    if not isinstance(values,dict) or set(values)-set(TrainingConfig.__dataclass_fields__):raise ValueError('unknown official UAV training settings')
    return TrainingConfig(**values)


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def _rng_state():
    state=np.random.get_state()
    return dict(torch_rng=torch.get_rng_state(),numpy_rng=(state[0],state[1].tolist(),state[2],state[3],state[4]),
        python_rng=random.getstate(),**({'cuda_rng':torch.cuda.get_rng_state_all()} if torch.cuda.is_available() else {}))


def _restore_rng(state):
    torch.set_rng_state(state['torch_rng']);value=state['numpy_rng']
    np.random.set_state((value[0],np.asarray(value[1],dtype=np.uint32),*value[2:]))
    random.setstate(state['python_rng'])
    if 'cuda_rng' in state and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])


def _epoch(trainer,loader,optimizer=None):
    training=optimizer is not None;trainer.policy.train(training)
    totals={};count=0;updates=0
    with torch.set_grad_enabled(training):
        for batch in loader:
            # Unlike upstream's fixed-env skip, retain the final partial batch.
            trainer.batch_size=len(batch[0])
            if training:optimizer.zero_grad(set_to_none=True)
            loss,metrics,_=trainer.forward_and_compute_loss(*batch)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite official YOPO loss')
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainer.policy.parameters()):
                    raise FloatingPointError('nonfinite official YOPO gradient')
                # Upstream defines max_grad_norm but does not apply clipping; retain its update.
                optimizer.step();updates+=1
            for key,value in {'loss':loss,**metrics}.items():
                totals[key]=totals.get(key,0.)+float(value.detach())*trainer.batch_size
            count+=trainer.batch_size
    if not count:raise ValueError('empty official dataset split')
    return {key:value/count for key,value in totals.items()},updates


def train(config,dataset_path,run_dir,*,resume=None,warm_start=None,prepare_only=False):
    if resume is not None and warm_start is not None:raise ValueError('resume and warm-start are mutually exclusive')
    run=Path(run_dir).resolve();runs=Path(RUNS_ROOT).resolve()
    if not run.is_relative_to(runs) or run==runs:raise ValueError('training run must be a new directory beneath runs/')
    if run.exists():raise FileExistsError(run)
    if not prepare_only and config.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('requested CUDA is unavailable')
    train_data,valid_data=load_datasets(dataset_path,val_ratio=config.val_ratio)
    fingerprint=dataset_fingerprint(dataset_path,train_data,valid_data)
    report=dict(training_samples=len(train_data),validation_samples=len(valid_data),
        dataset_path=str(Path(dataset_path).resolve()),dataset_fingerprint=fingerprint)
    if prepare_only:
        run.mkdir(parents=True);(run/'training.yaml').write_text(yaml.safe_dump(asdict(config)))
        (run/'prepared.json').write_text(json.dumps(report,indent=2)+'\n');return report
    torch.set_num_threads(config.threads);random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    if config.device.startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    official=Config(device=config.device,train=True);official['dataset_path']=report['dataset_path']
    with configuration(official):trainer=YopoTrainer(config.batch_size)
    optimizer=torch.optim.AdamW(trainer.policy.parameters(),lr=config.learning_rate,fused=config.device.startswith('cuda'))
    start=0;updates=0
    if warm_start is not None or resume is not None:
        core,metadata=load_checkpoint(resume or warm_start,device=config.device)
        trainer.policy.load_state_dict(core.network.state_dict());del core
        if resume is not None:
            previous=metadata.get('training_config',{})
            for key,value in asdict(config).items():
                if key not in ('epochs','threads','device','checkpoint_interval') and previous.get(key)!=value:
                    raise ValueError(f'resume training setting mismatch: {key}')
            if metadata.get('dataset_fingerprint')!=fingerprint or metadata.get('dataset_path')!=report['dataset_path']:
                raise ValueError('resume dataset mismatch')
            state=torch.load(Path(resume)/'training_state.pt',map_location='cpu',weights_only=True)
            start=state['epoch'];updates=state['optimizer_steps']
            if start>=config.epochs:raise ValueError('resume epochs must extend checkpoint')
            optimizer.load_state_dict(state['optimizer']);_restore_rng(state)
    run.mkdir(parents=True);(run/'training.yaml').write_text(yaml.safe_dump(asdict(config)))
    metadata={**report,'training_config':asdict(config),'training_source':{
        'repository':'https://github.com/TJU-Aerial-Robotics/YOPO','commit':UPSTREAM_COMMIT,
        'trainer':'YOPO/policy/yopo_trainer.py','entrypoint':'YOPO/train_yopo.py'}}
    loader_options=dict(batch_size=config.batch_size,num_workers=config.workers,pin_memory=config.device.startswith('cuda'),worker_init_fn=seed_worker)
    training=DataLoader(train_data,shuffle=True,**loader_options);validation=DataLoader(valid_data,shuffle=False,**loader_options)
    def save(name,epoch):
        save_checkpoint(run/name,trainer.policy,{**metadata,'completed_epoch':epoch,'optimizer_steps':updates})
        torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,optimizer_steps=updates,**_rng_state()),run/name/'training_state.pt')
    started=time.perf_counter()
    with SummaryWriter(log_dir=str(run/'tensorboard')) as writer:
        for epoch in range(start,config.epochs):
            train_metrics,n=_epoch(trainer,training,optimizer);updates+=n
            valid_metrics,_=_epoch(trainer,validation)
            entry=dict(epoch=epoch+1,optimizer_steps=updates,train=train_metrics,validation=valid_metrics)
            with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
            for split,metrics in (('Train',train_metrics),('Eval',valid_metrics)):
                for key,value in metrics.items():writer.add_scalar(f'{split}/{key}',value,epoch+1)
            if (epoch+1)%config.checkpoint_interval==0:save(f'epoch_{epoch+1:04d}',epoch+1)
        save('last',config.epochs)
    report.update(completed_epochs=config.epochs,resumed_from_epoch=start,optimizer_steps=updates,
        train=train_metrics,validation=valid_metrics,seconds=time.perf_counter()-started)
    (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--resume');parser.add_argument('--warm-start');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(train(load_training_config(args.config),args.dataset,args.run_dir,resume=args.resume,
        warm_start=args.warm_start,prepare_only=args.prepare_only),indent=2))
