"""Stage-2 promotion gate: held-out dense validation and actual controller reaching."""
import argparse
import json
from pathlib import Path
import tempfile
import yaml
import numpy as np
import torch
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from uav_ac.yopo.whole_body.training.runner import OracleCache, subset
from uav_ac.yopo.whole_body.deployment import YopoPlanner
from uav_ac.runners.config import load_config
from uav_ac.runners.whole_body_planning import run_whole_body_planning

def gate(result,*,minimum_valid=.9):
    if not result['families'] or set(result['families'])!=set(result['online']):return False
    dense=all(value.get('samples',0)>0 and value['selected_dense_valid_fraction']>=minimum_valid for value in result['families'].values())
    online=all(value['success'] and not value['collision'] for value in result['online'].values())
    return bool(result['families'] and result['online'] and dense and online)


def online_config(checkpoint,family,duration):
    root=Path(__file__).resolve().parents[4]
    values=yaml.safe_load((root/'configs/flight/yopo.yaml').read_text())
    values.update(visualize=False,duration=duration)
    # Small acceptance scenes retain their bounded workspace contract.
    values['whole_body']['enforce_height_ceiling']=True
    values['yopo']['checkpoint']=str(Path(checkpoint).resolve())
    with tempfile.NamedTemporaryFile(mode='w+',suffix='.yaml') as stream:
        yaml.safe_dump(values,stream);stream.flush()
        config=load_config(stream.name)
        from uav_ac.yopo.whole_body.validation.scenes import write_debug_scene
        from uav_ac.runners.whole_body_config import validate_whole_body
        artifacts=root/'runs/yopo_whole_body/acceptance_scenes';artifacts.mkdir(parents=True,exist_ok=True)
        directory=Path(tempfile.mkdtemp(prefix='check_',dir=artifacts))
        config['scene']=str(write_debug_scene(directory,family))
        # Batch acceptance owns its generated XML; interactive scene selectors
        # continue to resolve only the named benchmark in simulation/models.
        validate_whole_body(config,stream.name)
        return config


def check(dataset_path,checkpoint,*,families=('free',),samples_per_family=16,online=True,duration=30.):
    torch.set_num_threads(2);data=PlanningDataset(dataset_path,split='validation');cache=OracleCache(data,1.,0.)
    planner=YopoPlanner(checkpoint,expected_robot_hash=data.manifest['robot_sha256'],gripper_tolerance=.002)
    group={};details=[]
    for family in families:
        records=subset(data,samples_per_family,families=(family,));passed=[]
        for obs,priv in records:
            prediction=planner.plan(obs)
            evaluator=cache.get(priv).evaluator
            ok,clearance,violation,dt=evaluator.dense_validate(prediction.trajectory)
            passed.append(bool(ok));details.append({'family':family,'scene':priv.scene_id,'sample':priv.sample_index,'passed':bool(ok),
                'clearance':float(clearance),'violation':float(violation),'maximum_dt':float(dt),'selected':prediction.selected_index,**evaluator.last_validation_metrics})
        group[family]={'samples':len(records),'selected_dense_valid_fraction':float(np.mean(passed))}
    runs={}
    if online:
        for family in families:
            config=online_config(checkpoint,family,duration)
            runs[family]=run_whole_body_planning(config)
    result={'families':group,'dense_details':details,'online':runs,'minimum_dense_valid':.9,'acceptance_profile':'local_motion_v1',
        'interpretation':'held-out dense sampled whole-body feasibility plus actual controller reaching; no continuous safety guarantee'}
    result['passed']=gate(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--families',nargs='+',default=['free']);p.add_argument('--samples',type=int,default=16);p.add_argument('--output',required=True)
    args=p.parse_args();r=check(args.dataset,args.checkpoint,families=tuple(args.families),samples_per_family=args.samples)
    with Path(args.output).open('x') as s:json.dump(r,s,indent=2);s.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('dense_details','online')},indent=2))

if __name__=='__main__':main()
