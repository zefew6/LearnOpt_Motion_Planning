"""Collect full-body states in official YOPO maps; no expert trajectories."""
import argparse
from pathlib import Path
import json
import subprocess

import numpy as np
import yaml

from uav_ac.yopo.common.collection import RUNS_ROOT, SIMULATOR
from uav_ac.yopo.common.depth import DepthSpec
from uav_ac.yopo.whole_body.contracts import PlannerObservation
from uav_ac.yopo.whole_body.data.config import load_config
from uav_ac.yopo.whole_body.data.dataset import DYNAMIC_FIELDS, file_hash
from uav_ac.yopo.whole_body.data.maps import read_points, write_scene
from uav_ac.yopo.whole_body.model.encoding import STATE_LAYOUT


def sample_scene(sim, config, seed):
    """Sample independent collision-free endpoints with consistent derivatives."""
    from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
    from uav_ac.simulation.depth_observation import IdealDepthObserver
    rng=np.random.default_rng(seed);bounds=np.asarray(config.bounds)
    lower,upper=sim.robot.limits.joint_lower,sim.robot.limits.joint_upper
    observations=[];rejected={'start_collision':0,'goal_collision':0,'invalid_state':0}
    def clear(q):
        return not sim.robot.check_collision(q,clearance=config.clearance,self_clearance=config.self_clearance)['pairs']
    with IdealDepthObserver(sim.model,sim.data,spec=DepthSpec()) as camera:
        for _ in range(config.samples_per_scene*config.max_attempts_per_sample):
            position=rng.uniform(*bounds)
            goal=np.r_[rng.uniform(*bounds),rng.uniform(-np.pi,np.pi),rng.uniform(lower,upper)]
            direction=goal[:3]-position
            if np.linalg.norm(direction)<.3:continue
            yaw=np.arctan2(direction[1],direction[0])+rng.uniform(-.8,.8)
            derivatives=np.zeros((4,8));derivatives[0]=np.r_[position,yaw,rng.uniform(lower,upper)]
            speed=0. if rng.random()<.2 else rng.uniform(.1,2.)
            derivatives[1,:3]=speed*direction/np.linalg.norm(direction)
            derivatives[1,3:]=rng.uniform(-.4,.4,5)
            derivatives[2,:3]=rng.uniform(-1.,1.,3)
            derivatives[2,3:]=rng.uniform(-.5,.5,5)
            derivatives[3,:3]=rng.uniform(-2.,2.,3)
            try:
                q,v=recover_state(derivatives,config.gripper_gap)
                qgoal,_=recover_state(np.vstack((goal,np.zeros((3,8)))),config.gripper_gap)
                if not clear(q):rejected['start_collision']+=1;continue
                if not clear(qgoal):rejected['goal_collision']+=1;continue
                sim.robot.reset(q,v);depth,mask=camera.observe()
                observations.append(PlannerObservation(depth,mask,sim.robot.configuration,sim.robot.velocity,
                    derivatives[:3],goal,DepthSpec(),0.,derivatives[3],'augmentation'))
            except ValueError:
                rejected['invalid_state']+=1;continue
            if len(observations)==config.samples_per_scene:return observations,rejected
    raise RuntimeError(f'whole-body sampling exhausted: {rejected}, accepted={len(observations)}')


def generate_dataset(config, run_dir, *, map_binary=None, prepare_only=False):
    """Prepare a new run or execute the official-map/full-body collection path."""
    run=Path(run_dir).resolve()
    if run==Path(RUNS_ROOT).resolve() or not run.is_relative_to(Path(RUNS_ROOT).resolve()):
        raise ValueError('collection run must be a new directory beneath runs/')
    binary=Path(map_binary) if map_binary is not None else Path(RUNS_ROOT)/'yopo/build/map_generator'
    if not prepare_only and not binary.is_file():raise FileNotFoundError('build official map_generator first')
    settings=config.official_settings()
    settings.update(save_path=str(run/'dataset/maps'),tree_file=str(SIMULATOR/'assets/tree.ply'))
    run.mkdir(parents=True,exist_ok=False)
    collection=run/'map_collection.yaml'
    collection.write_text(yaml.safe_dump(settings,sort_keys=False))
    (run/'sampling.yaml').write_text(yaml.safe_dump(config.as_mapping(),sort_keys=False))
    if prepare_only:return collection
    subprocess.run([str(binary.resolve()),str(collection)],check=True)
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    dataset=run/'dataset';shards=dataset/'shards';shards.mkdir()
    split_names=[name for name,count in zip(('train','validation','test'),config.split_counts) for _ in range(count)]
    entries=[]
    for i,split in enumerate(split_names):
        sid=f'scene_{i:03d}';cloud=dataset/f'maps/pointcloud-{i}.ply'
        scene,box_count,translation=write_scene(dataset/'scenes',sid,read_points(cloud),config)
        sim=MujocoSimulation(scene,planning_path_capacity=0,record_actual_trajectory=False)
        samples,rejected=sample_scene(sim,config,config.seed+i*1009)
        shard=shards/f'{sid}.npz'
        np.savez_compressed(shard,**{name:np.stack([getattr(o,name) for o in samples]).astype(np.float32)
            if name=='depth_m' else np.stack([getattr(o,name) for o in samples]) for name in DYNAMIC_FIELDS})
        entries.append(dict(scene_id=sid,family='official',split=split,seed=config.seed+i,
            xml=str(scene.relative_to(dataset)),xml_sha256=file_hash(scene),
            pointcloud=str(cloud.relative_to(dataset)),pointcloud_sha256=file_hash(cloud),
            pointcloud_translation_nwu=translation,
            shard=str(shard.relative_to(dataset)),shard_sha256=file_hash(shard),count=len(samples),
            obstacle_boxes=box_count,rejected=rejected,path_validated=False))
        print(f'{sid}: {len(samples)} {split} observations, {box_count} voxel boxes',flush=True)
    robot=dataset/'scenes/model/aerial_manipulator.xml'
    manifest=dict(schema_version=2,profile='official_maps_whole_body_v1',map_source='official_yopo',
        sampling_config=config.as_mapping(),depth_spec=DepthSpec().as_mapping(),
        state_layout=list(STATE_LAYOUT),coordinates='NED/FRD; raw PLY maps are NWU',
        gripper_gap=config.gripper_gap,robot_xml=str(robot.relative_to(dataset)),robot_sha256=file_hash(robot),
        geometry_representation='occupied point-cloud voxels, merged only along contiguous z; ground plane',
        voxel_resolution=config.voxel_resolution,derivative_provenance='independent flatness-consistent PVA/jerk augmentation',
        interpretation='variable 8-D goals; collision-free endpoints; global reachability unconfirmed',
        upstream_commit='d5e1e60f83cc991602f142d9b6b96204a24d6b9f',scenes=entries)
    path=dataset/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--map-binary',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    print(generate_dataset(load_config(args.config),args.run_dir,map_binary=args.map_binary,prepare_only=args.prepare_only))


if __name__=='__main__':main()
