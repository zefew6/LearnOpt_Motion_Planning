"""Held-out, common-objective candidate diagnostics; ESDF is evaluation-only."""
import argparse
import json
from pathlib import Path
import torch
from uav_ac.yopo.data.dataset import PlanningDataset
from uav_ac.yopo.training.runner import OracleCache, subset, evaluate
from uav_ac.yopo.model.checkpoint import load_checkpoint

def compare(dataset_path, checkpoints, *, split='validation', samples=8, peak_weight=0., self_clearance=0.,guidance_profile='legacy_direct_v3',guidance=None):
    if split not in ('validation','test') or samples<1:
        raise ValueError('use a nonempty held-out split')
    torch.set_num_threads(2)
    data=PlanningDataset(dataset_path,split=split)
    records=subset(data,samples)
    from uav_ac.yopo.training.runner import TrainingConfig
    settings=TrainingConfig(guidance_profile=guidance_profile,guidance=guidance or {},collision_model='exact_geometry' if guidance_profile=='yopo_minco_v1' else 'planning_spheres',peak_weight=peak_weight,self_clearance=self_clearance)
    cache=OracleCache(data,peak_weight,self_clearance,guidance_profile=guidance_profile,guidance=settings.guidance)
    results={}
    for checkpoint in checkpoints:
        model,metadata=load_checkpoint(checkpoint,expected_robot_hash=data.manifest['robot_sha256'],expected_depth_spec=data.spec)
        results[str(Path(checkpoint))]=evaluate(model,records,cache,metadata['candidate_count'],'cpu')
    return {'split':split,'samples':len(records),'peak_weight':peak_weight,'self_clearance':self_clearance,'guidance_profile':guidance_profile,'guidance':settings.guidance,
            'models':results,'interpretation':'same sampled proxy objective, held-out scenes; no flight or continuous safety certification'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--checkpoints',nargs='+',required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--peak-weight',type=float,default=0.)
    parser.add_argument('--self-clearance',type=float,default=0.)
    parser.add_argument('--guidance-profile',choices=['legacy_direct_v3','yopo_minco_v1'],default='legacy_direct_v3')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=compare(args.dataset,args.checkpoints,split=args.split,samples=args.samples,
                   peak_weight=args.peak_weight,self_clearance=args.self_clearance,guidance_profile=args.guidance_profile)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
