"""Small real official-map → whole-body observation → training integration."""
import os
from pathlib import Path

import numpy as np
import pytest


def test_official_room_map_exports_with_pcl_metadata(tmp_path):
    import subprocess
    import yaml
    from uav_ac.yopo.whole_body.data.maps import read_points
    binary=Path(os.environ.get('YOPO_MAP_GENERATOR','runs/yopo/build/map_generator')).resolve()
    if not binary.is_file():pytest.skip('build the official map_generator first')
    values=yaml.safe_load(Path('configs/yopo/dataset/yopo_uav.yaml').read_text())
    output=tmp_path/'maps'
    values.update(maze_type=6,env_num=1,x_length=8,y_length=8,z_length=4,room_number=2,save_path=str(output))
    config=tmp_path/'room.yaml';config.write_text(yaml.safe_dump(values))
    result=subprocess.run([str(binary),str(config)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    points=read_points(output/'pointcloud-0.ply')
    assert len(points)>100 and np.isfinite(points).all()


def test_official_maps_collect_consistent_full_body_data_and_train(tmp_path, monkeypatch):
    from uav_ac.yopo.whole_body.data import generate
    from uav_ac.yopo.whole_body.data.config import CollectionConfig
    from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    from uav_ac.simulation.depth_observation import IdealDepthObserver
    from uav_ac.yopo.whole_body.training.runner import train, TrainingConfig
    from uav_ac.yopo.whole_body.deployment import YopoPlanner
    from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
    binary=Path(os.environ.get('YOPO_MAP_GENERATOR','runs/yopo/build/map_generator')).resolve()
    if not binary.is_file():pytest.skip('build the official map_generator first')
    monkeypatch.setattr(generate,'RUNS_ROOT',tmp_path)
    config=CollectionConfig(samples_per_scene=2,voxel_resolution=.2,bounds=((-2.,-2.,-2.8),(2.,2.,-1.5)),
        map_options=dict(maze_type=2,x_length=8,y_length=8,z_length=4,obstacle_number=2,width_min=.3,width_max=.6))
    root=tmp_path/'collection'
    generate.generate_dataset(config,root,map_binary=binary)
    data=PlanningDataset(root/'dataset')
    assert len(data)==6
    assert data.manifest['map_source']=='official_yopo'
    assert len(np.unique(np.stack([data[i][0].goal for i in range(6)]),axis=0))==6
    assert len({tuple(data[i][0].configuration[7:11]) for i in range(6)})==6
    assert len({r['xml_sha256'] for r in data.manifest['scenes']})==3
    for i in (0,1,2,4):
        obs,priv=data[i]
        sim=MujocoSimulation(priv.scene_path,planning_path_capacity=0,record_actual_trajectory=False)
        assert not sim.robot.check_collision(obs.configuration,clearance=config.clearance,self_clearance=config.self_clearance)['pairs']
        goal,_=recover_state(np.vstack((obs.goal,np.zeros((3,8)))),config.gripper_gap)
        assert not sim.robot.check_collision(goal,clearance=config.clearance,self_clearance=config.self_clearance)['pairs']
        q,v=recover_state(np.vstack((obs.start_pva,obs.start_jerk)),config.gripper_gap)
        assert abs(q[3:7]@obs.configuration[3:7])>1-1e-8
        np.testing.assert_allclose(v,obs.velocity,atol=1e-7)
        sim.robot.reset(obs.configuration.copy(),obs.velocity.copy())
        with IdealDepthObserver(sim.model,sim.data,spec=obs.depth_spec) as camera:
            depth,mask=camera.observe()
        assert np.array_equal(mask,obs.depth_valid)
        np.testing.assert_allclose(depth,obs.depth_m,atol=1e-6)
        assert obs.depth_valid.any() and np.linalg.norm(obs.velocity)>0
    result=train(TrainingConfig(profile='primitive_goal_v5',candidate_count=27,width=4,
        guidance_profile='yopo_minco_v1',collision_model='esdf_spheres',guidance_backend='torch_esdf',
        steps=1,batch_size=1,train_samples=1,validation_samples=1),root/'dataset',tmp_path/'train')
    assert np.isfinite(result['final_train_cost'])
    obs,_=data[0];plan=YopoPlanner(tmp_path/'train/last').plan(obs)
    np.testing.assert_allclose(plan.trajectory.evaluate(0),obs.start_pva[0],atol=1e-7)
    assert plan.candidate_count==27
    with pytest.raises(FileExistsError):generate.generate_dataset(config,root,map_binary=binary)
    # Original point clouds remain part of the verified dataset provenance.
    cloud=root/'dataset'/data.manifest['scenes'][0]['pointcloud']
    cloud.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='pointcloud'):PlanningDataset(root/'dataset')
