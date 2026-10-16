import numpy as np
from tests.yopo_dataset import write_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.simulation.depth_observation import IdealDepthObserver


def test_numeric_dataset_reconstructs_robot_depth_and_disjoint_scenes(tmp_path):
    root=tmp_path/'dataset_run'
    write_dataset(root, samples_per_scene=1)
    split_ids=[]
    for split in ('train','validation','test'):
        dataset=PlanningDataset(root/'dataset',split=split);assert len(dataset)==1
        observation,privileged=dataset[0];split_ids.extend(dataset.scene_ids)
        sim=MujocoSimulation(privileged.scene_path,planning_path_capacity=0,record_actual_trajectory=False)
        sim.robot.reset(observation.configuration.copy(),observation.velocity.copy())
        with IdealDepthObserver(sim.model,sim.data,spec=observation.depth_spec) as observer:
            depth,mask=observer.observe()
        assert np.array_equal(mask,observation.depth_valid)
        assert np.allclose(depth,observation.depth_m,atol=1e-6)
        assert not hasattr(observation,'scene_path') and not hasattr(observation,'esdf')
    assert len(set(split_ids))==3
