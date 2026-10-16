"""Stage-1 numerical audit of candidate gradients; never used by deployment."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from uav_ac.yopo.whole_body.training.runner import OracleCache, subset, tensors
from uav_ac.yopo.whole_body.training.losses import guidance_costs
from uav_ac.yopo.whole_body.model.encoding import Normalization
from uav_ac.yopo.whole_body.model.checkpoint import load_checkpoint
from uav_ac.yopo.whole_body.model.network import PrimitivePlannerNet

def audit_candidates(observation,oracle,parameters,indices,*,epsilon=1e-6,tolerance=1e-4):
    records=[]
    for index in indices:
        z=np.asarray(parameters[index],dtype=float)
        original=oracle.evaluate(observation,z[None]);fd=np.empty(34);components={name:np.empty(34) for name in original.components}
        for column in range(34):
            plus=z.copy();minus=z.copy();plus[column]+=epsilon;minus[column]-=epsilon
            a=oracle.evaluate(observation,plus[None]);b=oracle.evaluate(observation,minus[None])
            fd[column]=(a.costs[0]-b.costs[0])/(2*epsilon)
            for name in components:components[name][column]=(a.components[name][0]-b.components[name][0])/(2*epsilon)
        excluded=getattr(original,'detached_gradients',None)
        excluded=np.zeros(34) if excluded is None else excluded[0]
        analytic=original.free_gradients[0]+excluded
        error=np.abs(fd-analytic)/np.maximum(1.,np.maximum(abs(fd),abs(analytic)))
        records.append({'index':int(index),'cost':float(original.costs[0]),'max_relative_gradient_error':float(error.max()),
            'analytic_terminal_gradient_norm':float(np.linalg.norm(analytic[8:32])),
            'analytic_gradient_physical_34':analytic.tolist(),'fd_gradient_physical_34':fd.tolist(),'relative_gradient_error_by_column':error.tolist(),
            'guidance_gradient_physical_34':original.free_gradients[0].tolist(),'intentionally_detached_gradient_physical_34':excluded.tolist(),
            'component_costs':{name:float(value[0]) for name,value in original.components.items()},
            'component_terminal_gradient_norm':{name:float(np.linalg.norm(value[8:32])) for name,value in components.items()},
            'component_gradient_physical_34':{name:value.tolist() for name,value in components.items()},
            'finite':bool(np.isfinite(analytic).all() and np.isfinite(fd).all()),'passed':bool(np.isfinite(error).all() and error.max()<tolerance)})
    return {'passed':all(r['passed'] and r['finite'] for r in records),'epsilon':epsilon,'tolerance':tolerance,'candidates':records}


def audit_backward(output,observation,oracle,model,state,anchors,indices,*,epsilon=1e-6,tolerance=5e-4):
    output.parameters.retain_grad();output.raw_parameters.retain_grad()
    baseline=oracle.evaluate(observation,output.parameters[0].detach().numpy())
    expected=baseline.free_gradients/27
    excluded=getattr(baseline,'detached_gradients',None)
    model.zero_grad();guidance_costs(output.parameters[0],observation,oracle).mean().backward()
    actual=output.parameters.grad[0].detach().numpy()
    error=np.abs(actual-expected)/np.maximum(1.,np.maximum(abs(actual),abs(expected)))
    decoder=deepcopy(model).double().eval()
    raw=output.raw_parameters.detach().double();s=state.double();a=anchors.double();rows=[]
    for index in indices:
        errors=[];numeric=[];analytic=[]
        for column in (8,9,10):
            plus=raw.clone();minus=raw.clone();plus[0,index,column]+=epsilon;minus[0,index,column]-=epsilon
            za=decoder.decode(plus,s,a)[0,index:index+1].detach().numpy()
            zb=decoder.decode(minus,s,a)[0,index:index+1].detach().numpy()
            fd=float((oracle.evaluate(observation,za).costs[0]-oracle.evaluate(observation,zb).costs[0])/(2*epsilon*27))
            if excluded is not None:fd-=float(excluded[index]@((za[0]-zb[0])/(2*epsilon)))/27
            ad=float(output.raw_parameters.grad[0,index,column])
            numeric.append(fd);analytic.append(ad);errors.append(abs(fd-ad)/max(1.,abs(fd),abs(ad)))
        rows.append({'index':int(index),'raw_terminal_xyz_fd_gradient':numeric,'raw_terminal_xyz_autograd':analytic,'max_relative_error':max(errors)})
    norms={name:float(parameter.grad.norm()) for name,parameter in model.named_parameters() if parameter.grad is not None}
    finite=bool(np.isfinite(error).all() and np.isfinite(list(norms.values())).all() and all(np.isfinite(row['max_relative_error']) for row in rows))
    return {'passed':finite and float(error.max())<tolerance and all(row['max_relative_error']<tolerance for row in rows),
        'physical_output_backward_max_relative_error':float(error.max()),'raw_terminal_xyz_checks':rows,'tolerance':tolerance,
        'parameter_gradient_norms':norms,'finite':finite}


def audit(dataset_path,*,checkpoint=None,samples=3,seed=23,epsilon=1e-6):
    if samples<1 or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('audit requires positive sample count and epsilon')
    torch.set_num_threads(2);torch.manual_seed(seed)
    data=PlanningDataset(dataset_path,split='train');records=subset(data,samples)
    settings={}
    if checkpoint:
        settings=json.loads((Path(checkpoint)/'metadata.json').read_text()).get('training_config',{})
    objective={name:settings.get(name,default) for name,default in
        (('peak_weight',1.),('self_clearance',0.),('goal_stop_weight',0.),
         ('goal_stop_distance',1.5),('collision_model','planning_spheres'),('guidance_profile','legacy_direct_v3'),('guidance',{}))}
    objective['map_resolution']=settings.get('oracle_map_resolution',.02)
    objective['guidance_backend']=settings.get('guidance_backend','exact_cpu')
    cache=OracleCache(data,**objective)
    robot=cache.get(records[0][1]).robot
    normalization=Normalization(tuple(robot.limits.joint_lower),tuple(robot.limits.joint_upper))
    model=PrimitivePlannerNet(normalization,depth_spec=data.spec)
    if checkpoint:
        model,_=load_checkpoint(checkpoint,expected_robot_hash=data.manifest['robot_sha256'],expected_depth_spec=data.spec)
        if getattr(model,'profile',None) not in ('primitive_v4','primitive_goal_v5','official_whole_body_v1'):raise ValueError('audit requires primitive_v4')
    model.eval();reports=[]
    for obs,priv in records:
        output=model(*tensors([obs],model.normalization,27,'cpu',model.profile,model.spec))
        z=output.parameters[0].detach().numpy();selected=int(output.scores[0].argmax())
        oracle=cache.get(priv)
        if cache.backend=='torch_esdf':
            from uav_ac.yopo.whole_body.training.cuda import TorchGuidanceOracle
            oracle=TorchGuidanceOracle(oracle)
        numeric=audit_candidates(obs,oracle,z,sorted({selected,13,(selected+9)%27}),epsilon=epsilon)
        inputs=tensors([obs],model.normalization,27,'cpu',model.profile,model.spec)
        backward=audit_backward(output,obs,oracle,model,inputs[1],inputs[2],sorted({selected,13,(selected+9)%27}),epsilon=epsilon)
        norms=backward['parameter_gradient_norms']
        numeric.update({'scene_id':priv.scene_id,'sample_index':priv.sample_index,'selected':selected,
            'terminal_xyz_max_pair_distance_m':float(np.linalg.norm(z[:,None,8:11]-z[None,:,8:11],axis=-1).max()),
            'parameter_gradient_norms':norms,'gradient_finite':all(np.isfinite(list(norms.values()))),
            'terminal_head_weight_gradient_norm':float(model.parameter_head.weight.grad[8:32].norm())})
        numeric['neural_backward']=backward
        numeric['passed'] &= backward['passed']
        reports.append(numeric)
    return {'passed':all(r['passed'] for r in reports),'seed':seed,'samples':len(reports),'checkpoint':str(checkpoint) if checkpoint else None,'objective':objective,
        'reports':reports,'interpretation':'float64 reference for saved collision representation; physical-parameter FD and neural backward audit; no collision-feasibility or convergence claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True)
    parser.add_argument('--checkpoint');parser.add_argument('--samples',type=int,default=3);parser.add_argument('--output',required=True)
    parser.add_argument('--epsilon',type=float,default=1e-6,help='central finite-difference step; inspect step convergence for collision witnesses')
    args=parser.parse_args();result=audit(args.dataset,checkpoint=args.checkpoint,samples=args.samples,epsilon=args.epsilon)
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key!='reports'},indent=2))

if __name__=='__main__':main()
