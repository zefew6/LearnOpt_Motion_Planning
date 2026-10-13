"""Independent-sample guidance training; no online physics or expert labels."""
import argparse
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import json,random,time
import numpy as np
import torch
import yaml
from .config import _UniqueLoader
from .dataset import PlanningDataset
from .learning import DirectEvaluator,ranking_loss,augment_observation
from .losses import guidance_costs,score_loss,calibrate_cost_scale
from uav_ac.planning.end2end.network import WholeBodyPlannerNet
from uav_ac.planning.end2end.primitive_network import PrimitivePlannerNet
from uav_ac.planning.end2end.goal_network import GoalPrimitivePlannerNet
from uav_ac.planning.end2end.primitive import PrimitiveSpec
from uav_ac.planning.end2end.primitive import primitive_anchors,encode_primitive
from uav_ac.planning.end2end.anchors import make_anchors
from uav_ac.planning.end2end.encoding import Normalization,encode_observation
from uav_ac.planning.end2end.checkpoint import save_checkpoint,load_checkpoint
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig


@dataclass(frozen=True)
class TrainingConfig:
    profile: str = 'direct_v3'
    collision_model: str = 'planning_spheres'
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

    def __post_init__(self):
        if self.collision_model not in ('planning_spheres','exact_geometry'):raise ValueError('unknown collision model')
        if not isinstance(self.families,(tuple,list)) or any(f not in ('free','obstacle','low','wall') for f in self.families) or len(set(self.families))!=len(self.families):raise ValueError('invalid curriculum families')
        object.__setattr__(self,'families',tuple(self.families))
        if self.profile not in ('direct_v3','primitive_v4','primitive_goal_v5'):raise ValueError('unknown training profile')
        if self.profile in ('primitive_v4','primitive_goal_v5') and self.candidate_count!=27:raise ValueError('primitive_v4 requires 27 candidates')
        for name in ('steps','train_samples','validation_samples','batch_size','width','evaluation_interval','threads'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'invalid {name}')
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
    def __init__(self,dataset,peak_weight,self_clearance=0.,goal_stop_weight=0.,goal_stop_distance=1.5,collision_model='planning_spheres'):
        self.dataset=dataset;self.entries={};self.peak_weight=peak_weight;self.self_clearance=self_clearance;self.goal_stop_weight=goal_stop_weight;self.goal_stop_distance=goal_stop_distance;self.collision_model=collision_model
    def get(self,privileged):
        if privileged.scene_id not in self.entries:
            sim=MujocoSimulation(privileged.scene_path,planning_path_capacity=0,record_actual_trajectory=False)
            self.entries[privileged.scene_id]=(sim,DirectEvaluator.from_simulation(sim,
                gripper_gap=self.dataset.manifest['gripper_gap'],config=replace(
                    AerialManipulatorMINCOConfig(),
                    self_clearance=self.self_clearance),goal_stop_weight=self.goal_stop_weight,goal_stop_distance=self.goal_stop_distance,collision_model=self.collision_model))
            self.entries[privileged.scene_id][1].peak_weight=self.peak_weight
        return self.entries[privileged.scene_id][1]


def subset(dataset,count,*,families=()):
    indices=[i for i in range(len(dataset)) if not families or dataset.family_at(i) in families]
    if not indices:raise ValueError('empty dataset split/family selection')
    return [dataset[indices[int(i)]] for i in np.linspace(0,len(indices)-1,min(count,len(indices)),dtype=int)]


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
            result=cache.get(priv).evaluate(obs,output.parameters[0].cpu().numpy())
            score=output.scores[0].cpu().numpy();index=int(np.argmin(score) if model.score_mode=='cost' else np.argmax(score))
            unsafe=(result.components['environment']>1e-6)|(result.components['self']>1e-6)|(result.components['physical']>1e-6)
            for name,values in selected_violations.items():values.append(float(result.components[name][index]>1e-6))
            tier_order=np.lexsort((result.costs,unsafe.astype(int)));oracle_index=int(tier_order[0])
            costs.append(float(result.costs.mean()));selected.append(float(result.costs[index]));best.append(float(result.costs[oracle_index]))
            valid.append(float(np.mean(~unsafe)));selected_valid.append(float(not unsafe[index]));ranks.append(int(np.flatnonzero(tier_order==index)[0]));random_cost.append(float(result.costs.mean()))
            regret.append(float(result.costs[index]-result.costs[oracle_index]))
    return dict(mean_cost=float(np.mean(costs)),selected_cost=float(np.mean(selected)),oracle_best_cost=float(np.mean(best)),
                candidate_valid_fraction=float(np.mean(valid)),selected_valid_fraction=float(np.mean(selected_valid)),
                selected_rank=float(np.mean(ranks)),random_cost=float(np.mean(random_cost)),cost_gap_vs_safe_oracle=float(np.mean(regret)),
                selected_violation_fraction={name:float(np.mean(values)) for name,values in selected_violations.items()})


def train(config,dataset_path,run_dir,*,resume=None,warm_start=None):
    if resume is not None and warm_start is not None:raise ValueError('resume and warm_start are mutually exclusive')
    if config.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('requested CUDA is unavailable')
    run=Path(run_dir);run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.threads);torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed)
    data=PlanningDataset(dataset_path,split='train');validation=PlanningDataset(dataset_path,split='validation')
    samples=subset(data,config.train_samples,families=config.families);val_samples=subset(validation,config.validation_samples,families=config.families)
    cache=OracleCache(data,config.peak_weight,config.self_clearance,config.goal_stop_weight,config.goal_stop_distance,config.collision_model);val_cache=OracleCache(validation,config.peak_weight,config.self_clearance,config.goal_stop_weight,config.goal_stop_distance,config.collision_model)
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
            previous_value=tuple(previous.get(name,())) if name=='families' else previous.get(name,{'profile':'direct_v3','goal_stop_weight':0.,'goal_stop_distance':1.5,'collision_model':'planning_spheres'}.get(name))
            if name not in ('steps','evaluation_interval','threads','device') and previous_value!=value:
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
            with torch.inference_mode():baseline=model(*tensors([obs],normalization,config.candidate_count,config.device,config.profile,getattr(model,'spec',None))).parameters[0].cpu().numpy()
            calibration.extend(cache.get(priv).evaluate(obs,baseline).costs)
        cost_scale=calibrate_cost_scale(calibration)
    initial=evaluate(model,samples,cache,config.candidate_count,config.device)
    started=time.perf_counter();history=[];best_validation=(float('inf'),float('inf'))
    metadata={'robot_hash':data.manifest['robot_sha256'],'depth_spec':data.spec.as_mapping(),'candidate_count':config.candidate_count,
              'cost_scale':cost_scale,'training_config':asdict(config),'dataset_path':str(Path(dataset_path).resolve()),
              'gripper_gap':data.manifest['gripper_gap']}
    def save(name,step):
        target=run/name;save_checkpoint(target,model,{**metadata,'completed_step':step})
        torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                    'numpy_rng':rng.bit_generator.state,'step':step,
                    **({'cuda_rng':torch.cuda.get_rng_state_all()} if config.device.startswith('cuda') else {})},target/'training_state.pt')
    for step in range(start_step,config.steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        indices=rng.integers(0,len(samples),size=min(config.batch_size,len(samples)))
        records=[samples[int(i)] for i in indices]
        observations=[augment_observation(o,rng,normalization,goal_std=config.goal_std,velocity_std=config.velocity_std) for o,_ in records]
        output=model(*tensors(observations,normalization,config.candidate_count,config.device,config.profile,getattr(model,'spec',None)))
        loss=output.parameters.new_zeros(());raw_cost=[]
        for i,(observation,(_,priv)) in enumerate(zip(observations,records)):
            oracle=cache.get(priv);costs=guidance_costs(output.parameters[i],observation,oracle,scale=cost_scale)
            result=oracle.last_result;raw_cost.append(float(result.costs.mean()))
            if mode=='cost':loss=loss+costs.mean()+score_loss(output.scores[i],costs)
            else:
                unsafe=torch.as_tensor((result.components['environment']>1e-6)|(result.components['self']>1e-6)|(result.components['physical']>1e-6),device=config.device)
                rank,_=ranking_loss(output.scores[i:i+1],costs[None],unsafe[None])
                weights=torch.nn.functional.softplus(output.scores[i].detach()).clamp(min=.001)
                normalized=costs/costs.detach().mean().clamp(min=.001)
                loss=loss+(normalized*weights).mean()+rank
        loss=loss/len(records)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(gradient_norm):raise FloatingPointError('nonfinite training gradient')
        optimizer.step()
        entry={'step':step+1,'loss':float(loss.detach()),'mean_cost':float(np.mean(raw_cost)),'gradient_norm':float(gradient_norm)}
        if (step+1)%config.evaluation_interval==0 or step+1==config.steps:
            metrics=evaluate(model,val_samples,val_cache,config.candidate_count,config.device);entry['validation']=metrics
            if checkpoint_priority(metrics)<best_validation:
                best_validation=checkpoint_priority(metrics);save('best_'+str(step+1),step+1)
        history.append(entry)
        with (run/'metrics.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
    final=evaluate(model,samples,cache,config.candidate_count,config.device)
    validation_metrics=evaluate(model,val_samples,val_cache,config.candidate_count,config.device)
    save('last',config.steps)
    report={'completed_steps':config.steps,'resumed_from_step':start_step,'candidate_count':config.candidate_count,
            'initial_train_cost':initial['mean_cost'],'final_train_cost':final['mean_cost'],
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
