import numpy as np
from uav_ac.yopo.whole_body.data.config import DatasetConfig
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from uav_ac.yopo.whole_body.data.generate import generate_dataset
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state


def test_dynamic_dataset_reconstructs_consistent_motion_and_depth(tmp_path):
    config=DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=2,split_counts=(1,1,1))
    root=tmp_path/'dynamic';generate_dataset(config,root)
    ds=PlanningDataset(root/'dataset');assert ds.manifest['schema_version']==2
    for obs,priv in (ds[i] for i in range(len(ds))):
        assert obs.derivative_source=='reference' and obs.timestamp>0
        assert np.linalg.norm(obs.velocity)>1e-4
        q,v=recover_state(np.vstack((obs.start_pva,obs.start_jerk)),.06)
        assert abs(q[3:7]@obs.configuration[3:7])>1-1e-8
        assert np.allclose(v,obs.velocity,atol=1e-7)
        sim=MujocoSimulation(priv.scene_path,planning_path_capacity=0,record_actual_trajectory=False)
        sim.robot.reset(obs.configuration.copy(),obs.velocity.copy())
        with IdealDepthObserver(sim.model,sim.data,spec=obs.depth_spec) as camera:
            depth,mask=camera.observe()
        assert np.array_equal(mask,obs.depth_valid) and np.allclose(depth,obs.depth_m,atol=1e-6)


def test_sampling_provenance_records_nondefault_time_bounds(tmp_path):
    import json
    config=DatasetConfig(profile='rolling_v2',scene_count=3,samples_per_scene=1,split_counts=(1,1,1),
                         snapshot_min=.25,snapshot_max=.55)
    path=generate_dataset(config,tmp_path/'provenance')
    manifest=json.loads(path.read_text())
    assert manifest['sampling_config']['snapshot_min']==.25
    assert manifest['sampling_config']['snapshot_max']==.55
    assert manifest['sampling_config']['max_attempts_per_sample']==500
