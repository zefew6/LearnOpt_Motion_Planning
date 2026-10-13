"""Prepare a self-contained, scene-disjoint ideal-depth dataset."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
from .config import load_config
from .dataset import FIELDS,DYNAMIC_FIELDS,file_hash
from .scene_sampling import write_scene,sample_scene
from uav_ac.planning.end2end.encoding import STATE_LAYOUT


def generate_dataset(config,output_dir):
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';scenes_dir=dataset/'scenes';shards=dataset/'shards';shards.mkdir(parents=True)
    started=time.perf_counter();scene_entries=[];start_states=[];goal_states=[]
    split_names=['train']*config.split_counts[0]+['validation']*config.split_counts[1]+['test']*config.split_counts[2]
    for i,split in enumerate(split_names):
        seed=config.seed+i*1009
        coupled=config.profile in ('coupled_v3','portal_v4')
        source_metadata={}
        if coupled:
            from .coupled_scenes import write_coupled_scene
            from .coupled_sampling import sample_coupled_scene
            from uav_ac.simulation.mujoco_sim import MujocoSimulation
            if config.profile=='portal_v4':
                from .portal_scenes import write_portal_scene
                scene_id,family,path=write_portal_scene(scenes_dir,i,seed,config.depth)
            else:scene_id,family,path=write_coupled_scene(scenes_dir,i,seed,config.depth)
            sim=MujocoSimulation(path,planning_path_capacity=0,record_actual_trajectory=False)
            samples,sources,records=sample_coupled_scene(sim,config,seed+1)
            rejected={}
            witnesses=dataset/'sources';witnesses.mkdir(exist_ok=True)
            numeric=witnesses/f'{scene_id}.npz';diagnostics=witnesses/f'{scene_id}.json'
            np.savez_compressed(numeric,durations=np.stack([t.durations for t,_ in sources]),
                coefficients=np.stack([t.coefficients for t,_ in sources]),waypoints=np.stack([t.rrt_path for t,_ in sources]))
            diagnostics.write_text(json.dumps({'sources':[d for _,d in sources],'samples':records},indent=2)+'\n')
            source_metadata=dict(source_count=len(sources),source_numeric=str(numeric.relative_to(dataset)),
                source_numeric_sha256=file_hash(numeric),source_diagnostics=str(diagnostics.relative_to(dataset)),
                source_diagnostics_sha256=file_hash(diagnostics))
        else:
            scene_id,family,path=write_scene(scenes_dir,i,seed,config.depth,profile=config.profile)
            samples,rejected=sample_scene(path,config,seed+1)
        shard=shards/f'{scene_id}.npz'
        fields=FIELDS if config.profile=='stationary_v1' else DYNAMIC_FIELDS
        np.savez_compressed(shard,**{name:np.stack([getattr(obs,name) for obs in samples]) for name in fields})
        scene_entries.append(dict(scene_id=scene_id,family=family,split=split,seed=seed,
            xml=str(path.relative_to(dataset)),xml_sha256=file_hash(path),shard=str(shard.relative_to(dataset)),
            shard_sha256=file_hash(shard),count=len(samples),rejected=rejected,path_validated=coupled,**source_metadata))
        start_states.extend(obs.start_pva[0] for obs in samples);goal_states.extend(obs.goal for obs in samples)
    dynamic=config.profile!='stationary_v1'
    manifest=dict(schema_version=2 if dynamic else 1,profile=config.profile,sampling_config=config.as_mapping(),seed=config.seed,depth_spec=config.depth.as_mapping(),
        state_layout=list(STATE_LAYOUT),coordinates='NED/FRD; start-yaw local frame',
        trajectory_profile=('rolling two-piece MINCO; local terminal PVA and two variable durations'
                            if dynamic else 'stationary four-piece MINCO; durations [.75,.75,.75,.75]'),
        derivative_provenance=('sampled existing reference trajectory; no physics rollout' if dynamic else 'stationary'),
        gripper_gap=config.gripper_gap,robot_xml='scenes/model/aerial_manipulator.xml',
        robot_sha256=file_hash(scenes_dir/'model'/'aerial_manipulator.xml'),scenes=scene_entries)
    manifest_path=dataset/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if config.profile in ('coupled_v3','portal_v4'):
        manifest.update(source_profile='authored six-piece full-body MINCO; not optimized',
            derivative_provenance='densely validated reference snapshots; no physics rollout',
            trajectory_profile='training decoder unchanged: direct_v3 two-piece; source references are six-piece',
            interpretation='same-base frozen-arm collision witnesses; not proof that all fixed-arm routes are infeasible')
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    start_states=np.asarray(start_states);goal_states=np.asarray(goal_states)
    report=dict(samples=len(start_states),scenes=len(scene_entries),seconds=time.perf_counter()-started,
        starts_min=start_states.min(axis=0).tolist(),starts_max=start_states.max(axis=0).tolist(),
        goals_min=goal_states.min(axis=0).tolist(),goals_max=goal_states.max(axis=0).tolist(),
        rejected={k:sum(s['rejected'][k] for s in scene_entries) for k in scene_entries[0]['rejected']},
        path_feasibility=('all source paths densely exact-geometry/physical checked; not continuous certification'
                          if config.profile in ('coupled_v3','portal_v4') else 'unconfirmed; only endpoint limits/exact geometry checked'),
        observation='first-hit ideal depth; self-occlusion retained, hidden geometry unavailable to network')
    (output/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return manifest_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();print(generate_dataset(load_config(args.config),args.run_dir))

if __name__=='__main__':main()
