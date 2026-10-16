"""Independent-sample guidance training; no online physics or expert labels."""
import argparse
from dataclasses import dataclass, asdict, replace, field
from pathlib import Path
import json
import random
import time
import math
import shutil
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from uav_ac.yopo.common.config import _UniqueLoader
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from uav_ac.yopo.whole_body.training.oracle import DirectEvaluator
from uav_ac.yopo.whole_body.training.losses import ranking_loss, augment_observation, guidance_costs, guidance_costs_from_result, score_loss, calibrate_cost_scale
from uav_ac.yopo.whole_body.training.guidance import MincoGuidanceEvaluator, MincoGuidanceConfig
from uav_ac.yopo.whole_body.model.network import WholeBodyPlannerNet, PrimitivePlannerNet, GoalPrimitivePlannerNet
from uav_ac.yopo.whole_body.model.primitives import PrimitiveSpec, primitive_anchors, encode_primitive, make_anchors
from uav_ac.yopo.whole_body.model.encoding import Normalization, encode_observation
from uav_ac.yopo.whole_body.model.checkpoint import save_checkpoint, load_checkpoint
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig

@dataclass(frozen=True)
class TrainingConfig:
    profile: str = 'direct_v3'
    guidance_profile: str = 'legacy_direct_v3'
    guidance: dict = field(default_factory=dict)
    collision_model: str = 'planning_spheres'
    guidance_backend: str = 'exact_cpu'
    families: tuple = ()
    steps: int = 100
    train_samples: int = 32
    validation_samples: int = 8
    batch_size: int = 4
    candidate_count: int = 1
    width: int = 16
    seed: int = 23
    learning_rate: float = .001
    device: str = 'cpu'
    goal_std: float = 0.
    goal_stop_weight: float = 0.
    goal_stop_distance: float = 1.5
    velocity_std: float = 0.
    peak_weight: float = 0.
    self_clearance: float = 0.
    evaluation_interval: int = 25
    threads: int = 2
    train_evaluation_samples: int = 0
    checkpoint_interval: int = 0
    checkpoint_keep: int = 3
    warmup_steps: int = 0
    final_learning_rate_ratio: float = 1.
    sampling: str = 'random'
    guidance_workers: int = 0
    oracle_map_resolution: float = .02

    def __post_init__(self):
        if self.guidance_profile not in ('legacy_direct_v3','yopo_minco_v1'):raise ValueError('unknown guidance profile')
        if self.guidance_profile=='yopo_minco_v1':
            object.__setattr__(self,'guidance',asdict(MincoGuidanceConfig.from_mapping(self.guidance)))
            if self.collision_model not in ('exact_geometry','esdf_spheres'):raise ValueError('new guidance requires an explicit whole-body collision representation')
            if self.goal_stop_weight or self.peak_weight:raise ValueError('new guidance excludes stopping and duplicate peak loss')
        elif self.guidance:raise ValueError('guidance settings require yopo_minco_v1')
        if self.collision_model not in ('planning_spheres','exact_geometry','esdf_spheres'):raise ValueError('unknown collision model')
        if not isinstance(self.families,(tuple,list)) or any(f not in ('free','obstacle','low','wall') for f in self.families) or len(set(self.families))!=len(self.families):raise ValueError('invalid curriculum families')
        object.__setattr__(self,'families',tuple(self.families))
        if self.profile not in ('direct_v3','primitive_v4','primitive_goal_v5'):raise ValueError('unknown training profile')
        if self.profile in ('primitive_v4','primitive_goal_v5') and self.candidate_count!=27:raise ValueError('primitive_v4 requires 27 candidates')
        for name in ('steps','validation_samples','batch_size','width','evaluation_interval','threads','checkpoint_keep'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'invalid {name}')
        for name in ('train_samples','train_evaluation_samples','checkpoint_interval','warmup_steps','guidance_workers'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError(f'invalid {name}')
        if self.warmup_steps>=self.steps:raise ValueError('warmup must be shorter than training')
        if not np.isfinite(self.final_learning_rate_ratio) or not 0<self.final_learning_rate_ratio<=1:raise ValueError('invalid final learning rate ratio')
        if self.sampling not in ('random','shuffle'):raise ValueError('invalid sampling')
        if not np.isfinite(self.oracle_map_resolution) or self.oracle_map_resolution<=0:raise ValueError('invalid oracle map resolution')
        if self.guidance_backend not in ('exact_cpu','torch_esdf'):raise ValueError('invalid guidance backend')
        if (self.guidance_backend=='torch_esdf')!=(self.collision_model=='esdf_spheres'):raise ValueError('torch_esdf requires explicit esdf_spheres collision representation')
        if self.guidance_backend=='torch_esdf' and (self.guidance_workers or self.guidance_profile!='yopo_minco_v1'):raise ValueError('torch_esdf uses batched torch guidance without CPU workers')
        if self.guidance_workers and self.guidance_profile!='yopo_minco_v1':raise ValueError('parallel guidance requires yopo_minco_v1')
        if self.candidate_count not in (1,27) or isinstance(self.candidate_count,bool):raise ValueError('candidate_count must be 1 or 27')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:raise ValueError('invalid seed')
        if self.learning_rate<=0 or not np.isfinite(self.learning_rate):raise ValueError('invalid learning rate')
        for name in ('goal_std','velocity_std','peak_weight','self_clearance','goal_stop_weight'):
            if getattr(self,name)<0 or not np.isfinite(getattr(self,name)):raise ValueError(f'invalid {name}')
        if not np.isfinite(self.goal_stop_distance) or self.goal_stop_distance<=0:raise ValueError('invalid goal_stop_distance')
        if self.width<4:raise ValueError('width must be >=4')
        if self.device!='cpu' and not self.device.startswith('cuda'):raise ValueError('device must be cpu or cuda')


def load_training_config(path):
    with Path(path).open() as stream:values=yaml.load(stream,Loader=_UniqueLoader)
    if not isinstance(values,dict) or set(values)-set(TrainingConfig.__dataclass_fields__):raise ValueError('unknown training settings')
    return TrainingConfig(**values)


class OracleCache:
    def __init__(self,dataset,peak_weight,self_clearance=0.,goal_stop_weight=0.,goal_stop_distance=1.5,collision_model='planning_spheres',guidance_profile='legacy_direct_v3',guidance=None,*,map_resolution=.02,max_entries=3,guidance_backend='exact_cpu'):
        self.backend=guidance_backend;self.dataset=dataset;self.entries=OrderedDict();self.map_resolution=map_resolution;self.max_entries=max_entries;self.peak_weight=peak_weight;self.self_clearance=self_clearance;self.goal_stop_weight=goal_stop_weight;self.goal_stop_distance=goal_stop_distance;self.collision_model=collision_model;self.guidance_profile=guidance_profile;self.guidance=guidance or {}
    def get(self,privileged):
        if privileged.scene_id not in self.entries:
            sim=MujocoSimulation(privileged.scene_path,planning_path_capacity=0,record_actual_trajectory=False)
            if self.guidance_profile=='yopo_minco_v1':
                oracle=MincoGuidanceEvaluator.from_simulation(sim,gripper_gap=self.dataset.manifest['gripper_gap'],config=replace(AerialManipulatorMINCOConfig(),self_clearance=self.self_clearance,esdf_resolution=self.map_resolution,esdf_discretization_margin=max(.04,np.sqrt(3.)*self.map_resolution+1e-8)),guidance=MincoGuidanceConfig.from_mapping(self.guidance))
                self.entries[privileged.scene_id]=(sim,oracle)
                self._trim()
                return oracle
            self.entries[privileged.scene_id]=(sim,DirectEvaluator.from_simulation(sim,
                gripper_gap=self.dataset.manifest['gripper_gap'],config=replace(
                    AerialManipulatorMINCOConfig(),
                    self_clearance=self.self_clearance),goal_stop_weight=self.goal_stop_weight,goal_stop_distance=self.goal_stop_distance,collision_model=self.collision_model))
            self.entries[privileged.scene_id][1].peak_weight=self.peak_weight
        self.entries.move_to_end(privileged.scene_id);self._trim()
        return self.entries[privileged.scene_id][1]

    def get_torch(self,privileged,device):
        from uav_ac.yopo.whole_body.training.cuda import TorchGuidance
        oracle=self.get(privileged)
        if not hasattr(oracle,'torch_backend'):
            with torch.inference_mode(False):oracle.torch_backend=TorchGuidance(oracle,device)
        return oracle.torch_backend

    def _trim(self):
        while len(self.entries)>self.max_entries:self.entries.popitem(last=False)


class _Subset(Sequence):
    """Retain indices, not thousands of copied depth observations."""
    def __init__(self,dataset,indices):self.dataset=dataset;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        if isinstance(index,slice):return _Subset(self.dataset,self.indices[index])
        return self.dataset[self.indices[index]]


def subset(dataset,count,*,families=()):
    indices=[i for i in range(len(dataset)) if not families or dataset.family_at(i) in families]
    if not indices:raise ValueError('empty dataset split/family selection')
    if count:indices=[indices[int(i)] for i in np.linspace(0,len(indices)-1,min(count,len(indices)),dtype=int)]
    return _Subset(dataset,indices)


def learning_rate_at(config,step):
    if config.warmup_steps and step<config.warmup_steps:return config.learning_rate*(step+1)/config.warmup_steps
    progress=(step-config.warmup_steps)/max(1,config.steps-1-config.warmup_steps)
    return config.learning_rate*(config.final_learning_rate_ratio+(1-config.final_learning_rate_ratio)*.5*(1+math.cos(math.pi*progress)))


def batch_indices(count,batch_size,step,sampling,seed,rng):
    if sampling=='random':return rng.integers(0,count,size=min(batch_size,count))
    batches=math.ceil(count/batch_size);epoch,index=divmod(step,batches)
    order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)
    return order[index*batch_size:min((index+1)*batch_size,count)]


_worker_cache=None


def _initialize_guidance_worker(manifest,config):
    global _worker_cache
    torch.set_num_threads(1)
    _worker_cache=OracleCache(SimpleNamespace(manifest=manifest),config.peak_weight,config.self_clearance,
        guidance_profile=config.guidance_profile,guidance=config.guidance,map_resolution=config.oracle_map_resolution,max_entries=8)


def _evaluate_guidance_worker(task):
    obs,priv,parameters=task
    return _worker_cache.get(priv).evaluate(obs,parameters)


def tensors(observations,normalization,count,device,profile='direct_v3',primitive_spec=None):
    if profile=='primitive_goal_v5' and primitive_spec is None:primitive_spec=PrimitiveSpec(inner_min=0.,tail_min=0.)
    encoded=[encode_primitive(o,normalization,primitive_spec) if profile in ('primitive_v4','primitive_goal_v5') else encode_observation(o,normalization) for o in observations]
    anchors=np.stack([primitive_anchors(o,normalization,primitive_spec) if profile in ('primitive_v4','primitive_goal_v5') else make_anchors(o,normalization,count=count) for o in observations])
    return (torch.as_tensor(np.concatenate([e.depth for e in encoded]),device=device),
            torch.as_tensor(np.concatenate([e.state for e in encoded]),device=device),
            torch.as_tensor(anchors,dtype=torch.float32,device=device))


def checkpoint_priority(metrics):
    return (-metrics['selected_valid_fraction'],metrics['selected_cost'])


def evaluate(model,samples,cache,count,device):
    model.eval();costs=[];selected=[];best=[];valid=[];random_cost=[];regret=[];selected_valid=[];ranks=[]
    selected_violations={name:[] for name in ('environment','self','physical')}
    with torch.inference_mode():
        for obs,priv in samples:
            output=model(*tensors([obs],model.normalization,count,device,getattr(model,'profile','direct_v3'),getattr(model,'spec',None)))
            result=cache.get_torch(priv,device).evaluate(output.parameters,[obs]).single(0).numpy() if getattr(cache,'backend','exact_cpu')=='torch_esdf' else cache.get(priv).evaluate(obs,output.parameters[0].cpu().numpy())
            score=output.scores[0].cpu().numpy();index=int(np.argmin(score) if model.score_mode=='cost' else np.argmax(score))
            unsafe=result.collided if result.collided is not None else (result.components['environment']>1e-6)|(result.components['self']>1e-6)|(result.components['physical']>1e-6)
            physical_bad=result.violations>1e-8 if result.collided is not None else result.components['physical']>1e-6
            for name,values in selected_violations.items():
                values.append(float(physical_bad[index]) if name=='physical' else float(result.collision_components[name][index]) if result.collision_components is not None and name in result.collision_components else float(result.components[name][index]>1e-6))
            tier_order=np.lexsort((result.costs,unsafe.astype(int)));oracle_index=int(tier_order[0])
            costs.append(float(result.costs.mean()));selected.append(float(result.costs[index]));best.append(float(result.costs[oracle_index]))
            feasible=~unsafe & ~physical_bad
            valid.append(float(np.mean(feasible)));selected_valid.append(float(feasible[index]));ranks.append(int(np.flatnonzero(tier_order==index)[0]));random_cost.append(float(result.costs.mean()))
            regret.append(float(result.costs[index]-result.costs[oracle_index]))
    return dict(mean_cost=float(np.mean(costs)),selected_cost=float(np.mean(selected)),oracle_best_cost=float(np.mean(best)),
                candidate_valid_fraction=float(np.mean(valid)),selected_valid_fraction=float(np.mean(selected_valid)),
                selected_rank=float(np.mean(ranks)),random_cost=float(np.mean(random_cost)),cost_gap_vs_safe_oracle=float(np.mean(regret)),
                selected_violation_fraction={name:float(np.mean(values)) for name,values in selected_violations.items()})


def train(config,dataset_path,run_dir,*,resume=None,warm_start=None):
    if resume is not None and warm_start is not None:raise ValueError('resume and warm_start are mutually exclusive')
    if config.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('requested CUDA is unavailable')
    if config.device.startswith('cuda'):torch.backends.cudnn.benchmark=True
    run=Path(run_dir);run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.threads);torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed)
    data=PlanningDataset(dataset_path,split='train',max_cached_shards=8);validation=PlanningDataset(dataset_path,split='validation',max_cached_shards=2)
    samples=subset(data,config.train_samples,families=config.families);val_samples=subset(validation,config.validation_samples,families=config.families)
    cache=OracleCache(data,config.peak_weight,config.self_clearance,config.goal_stop_weight,config.goal_stop_distance,config.collision_model,config.guidance_profile,config.guidance,map_resolution=config.oracle_map_resolution,max_entries=8,guidance_backend=config.guidance_backend);val_cache=OracleCache(validation,config.peak_weight,config.self_clearance,config.goal_stop_weight,config.goal_stop_distance,config.collision_model,config.guidance_profile,config.guidance,map_resolution=config.oracle_map_resolution,guidance_backend=config.guidance_backend)
    first_oracle=cache.get(samples[0][1]);robot=first_oracle.robot
    normalization=Normalization(tuple(robot.limits.joint_lower),tuple(robot.limits.joint_upper))
    mode='cost' if config.candidate_count==1 else 'ranking'
    cls={'direct_v3':WholeBodyPlannerNet,'primitive_v4':PrimitivePlannerNet,'primitive_goal_v5':GoalPrimitivePlannerNet}[config.profile]
    model=cls(normalization,score_mode=mode,width=config.width,depth_spec=data.spec).to(config.device)
    if warm_start is not None:
        model,metadata=load_checkpoint(warm_start,device=config.device,expected_robot_hash=data.manifest['robot_sha256'],expected_depth_spec=data.spec)
        if metadata['profile']!=config.profile or metadata['candidate_count']!=config.candidate_count or model.width!=config.width or model.score_mode!=mode:raise ValueError('warm-start architecture/score mismatch')
        if metadata.get('gripper_gap')!=data.manifest['gripper_gap']:raise ValueError('warm-start gripper mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.0001)
    start_step=0;rng=np.random.default_rng(config.seed)
    if resume is not None:
        model,metadata=load_checkpoint(resume,device=config.device,expected_robot_hash=data.manifest['robot_sha256'],expected_depth_spec=data.spec)
        if metadata['profile']!=config.profile:raise ValueError('resume profile mismatch')
        if metadata['candidate_count']!=config.candidate_count or model.width!=config.width:raise ValueError('resume architecture mismatch')
        previous=metadata['training_config']
        for name,value in asdict(config).items():
            previous_value=tuple(previous.get(name,())) if name=='families' else previous.get(name,{'guidance_backend':'exact_cpu','checkpoint_keep':3,'train_evaluation_samples':0,'checkpoint_interval':0,'warmup_steps':0,'final_learning_rate_ratio':1.,'sampling':'random','guidance_workers':0,'oracle_map_resolution':.02,'profile':'direct_v3','goal_stop_weight':0.,'goal_stop_distance':1.5,'collision_model':'planning_spheres','guidance_profile':'legacy_direct_v3','guidance':{}}.get(name))
            if name not in ('steps','evaluation_interval','threads','device','guidance_workers','checkpoint_interval','checkpoint_keep') and previous_value!=value:
                raise ValueError(f'resume training setting mismatch: {name}')
        if metadata.get('dataset_path')!=str(Path(dataset_path).resolve()) or metadata.get('gripper_gap')!=data.manifest['gripper_gap']:
            raise ValueError('resume dataset/gripper mismatch')
        state=torch.load(Path(resume)/'training_state.pt',map_location='cpu',weights_only=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.0001);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);start_step=int(state['step']);rng.bit_generator.state=state['numpy_rng']
        if start_step>=config.steps:raise ValueError('resume steps must extend checkpoint')
        if config.device.startswith('cuda') and 'cuda_rng' in state:torch.cuda.set_rng_state_all(state['cuda_rng'])
        cost_scale=float(metadata['cost_scale'])
    else:
        calibration=[]
        for obs,priv in samples[:8]:
            model.eval()
            with torch.inference_mode():
                baseline=model(*tensors([obs],normalization,config.candidate_count,config.device,config.profile,getattr(model,'spec',None))).parameters
                values=cache.get_torch(priv,config.device).evaluate(baseline,[obs]).costs[0].cpu().numpy() if cache.backend=='torch_esdf' else cache.get(priv).evaluate(obs,baseline[0].cpu().numpy()).costs
            calibration.extend(values)
        cost_scale=calibrate_cost_scale(calibration)
    train_eval=samples if not config.train_evaluation_samples else _Subset(samples,np.linspace(0,len(samples)-1,min(len(samples),config.train_evaluation_samples),dtype=int).tolist())
    initial=evaluate(model,train_eval,cache,config.candidate_count,config.device)
    started=time.perf_counter();history=[];best_validation=(float('inf'),float('inf'))
    metadata={'robot_hash':data.manifest['robot_sha256'],'depth_spec':data.spec.as_mapping(),'candidate_count':config.candidate_count,
              'cost_scale':cost_scale,'training_config':asdict(config),'dataset_path':str(Path(dataset_path).resolve()),
              'gripper_gap':data.manifest['gripper_gap'],
              'collision_representation':{'backend':config.guidance_backend,'model':config.collision_model,'esdf_resolution_m':config.oracle_map_resolution,
                  'voxel_margin_affects':'safety_cost_only','version':1}}
    def save(name,step):
        target=run/name;save_checkpoint(target,model,{**metadata,'completed_step':step})
        torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                    'numpy_rng':rng.bit_generator.state,'step':step,
                    **({'cuda_rng':torch.cuda.get_rng_state_all()} if config.device.startswith('cuda') else {})},target/'training_state.pt')
    executor=ProcessPoolExecutor(max_workers=config.guidance_workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=_initialize_guidance_worker,initargs=(data.manifest,config)) if config.guidance_workers else None
    try:
        for step in range(start_step,config.steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            step_started=time.perf_counter();lr=learning_rate_at(config,step)
            for group in optimizer.param_groups:group['lr']=lr
            indices=batch_indices(len(samples),config.batch_size,step,config.sampling,config.seed,rng)
            records=[samples[int(i)] for i in indices]
            observations=[augment_observation(o,rng,normalization,goal_std=config.goal_std,velocity_std=config.velocity_std) for o,_ in records]
            output=model(*tensors(observations,normalization,config.candidate_count,config.device,config.profile,getattr(model,'spec',None)))
            guidance_started=time.perf_counter()
            results=None
            if cache.backend=='torch_esdf':
                groups={};results=[None]*len(records)
                for i,(_,priv) in enumerate(records):groups.setdefault(priv.scene_id,[]).append(i)
                for indices in groups.values():
                    selected=torch.tensor(indices,device=config.device)
                    result=cache.get_torch(records[indices[0]][1],config.device).evaluate(output.parameters[selected],[observations[i] for i in indices])
                    for j,i in enumerate(indices):results[i]=result.single(j)
            elif executor:
                physical=output.parameters.detach().cpu().numpy()
                results=list(executor.map(_evaluate_guidance_worker,[(o,r[1],physical[i]) for i,(o,r) in enumerate(zip(observations,records))]))
            loss=output.parameters.new_zeros(());raw_cost=[]
            for i,(observation,(_,priv)) in enumerate(zip(observations,records)):
                if results is None:
                    oracle=cache.get(priv);costs=guidance_costs(output.parameters[i],observation,oracle,scale=cost_scale);result=oracle.last_result
                elif cache.backend=='torch_esdf':
                    result=results[i];costs=result.costs/cost_scale
                else:
                    result=results[i];costs=guidance_costs_from_result(output.parameters[i],result,scale=cost_scale)
                raw_cost.append(float(result.costs.detach().mean()) if isinstance(result.costs,torch.Tensor) else float(result.costs.mean()))
                if mode=='cost':loss=loss+costs.mean()+score_loss(output.scores[i],costs)
                else:
                    unsafe=torch.as_tensor(result.collided if result.collided is not None else (result.components['environment']>1e-6)|(result.components['self']>1e-6)|(result.components['physical']>1e-6),device=config.device)
                    weights=torch.nn.functional.softplus(output.scores[i].detach()).clamp(min=.001)
                    scores=torch.nn.functional.softplus(output.scores[i:i+1]) if config.guidance_profile=='yopo_minco_v1' else output.scores[i:i+1]
                    rank,_=ranking_loss(scores,costs[None],unsafe[None])
                    normalized=costs/costs.detach().mean().clamp(min=.001)
                    loss=loss+(normalized*weights).mean()+rank
            guidance_seconds=time.perf_counter()-guidance_started
            loss=loss/len(records)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient_norm):raise FloatingPointError('nonfinite training gradient')
            optimizer.step()
            entry={'step':step+1,'loss':float(loss.detach()),'mean_cost':float(np.mean(raw_cost)),'gradient_norm':float(gradient_norm),'learning_rate':lr,'batch_samples':len(records),'step_seconds':time.perf_counter()-step_started,'guidance_seconds':guidance_seconds}
            if (step+1)%config.evaluation_interval==0 or step+1==config.steps:
                metrics=evaluate(model,val_samples,val_cache,config.candidate_count,config.device);entry['validation']=metrics
                if checkpoint_priority(metrics)<best_validation:
                    best_validation=checkpoint_priority(metrics);save('best_'+str(step+1),step+1)
            if config.checkpoint_interval and ((step+1)%config.checkpoint_interval==0 or step==start_step):
                name=f'step_{step+1:06d}';save(name,step+1)
                link=run/'.latest-next';link.symlink_to(name,target_is_directory=True);link.replace(run/'latest')
                for old in sorted(run.glob('step_[0-9]*'))[:-config.checkpoint_keep]:shutil.rmtree(old)
            history.append(entry)
            with (run/'metrics.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
    finally:
        if executor:executor.shutdown(wait=True,cancel_futures=True)
    final=evaluate(model,train_eval,cache,config.candidate_count,config.device)
    validation_metrics=evaluate(model,val_samples,val_cache,config.candidate_count,config.device)
    save('last',config.steps)
    report={'completed_steps':config.steps,'resumed_from_step':start_step,'candidate_count':config.candidate_count,
            'training_samples':len(samples),'train_evaluation_samples':len(train_eval),'validation_samples':len(val_samples),'initial_train_cost':initial['mean_cost'],'final_train_cost':final['mean_cost'],
            'relative_cost_reduction':(initial['mean_cost']-final['mean_cost'])/max(1.,initial['mean_cost']),
            'validation':validation_metrics,'train_metrics':final,'seconds':time.perf_counter()-started,
            'interpretation':'offline independent-sample learning; sampled proxy constraints, no flight guarantee'}
    (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--resume');parser.add_argument('--warm-start');args=parser.parse_args()
    print(json.dumps(train(load_training_config(args.config),args.dataset,args.run_dir,resume=args.resume,warm_start=args.warm_start),indent=2))

if __name__=='__main__':main()
