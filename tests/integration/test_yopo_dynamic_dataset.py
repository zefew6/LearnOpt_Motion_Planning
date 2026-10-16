import numpy as np
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset
from tests.yopo_dataset import write_dataset
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state


def test_dynamic_dataset_reconstructs_consistent_motion_and_depth(tmp_path):
    root=tmp_path/'dynamic';write_dataset(root, samples_per_scene=2)
    ds=PlanningDataset(root/'dataset');assert ds.manifest['schema_version']==2
    for obs,priv in (ds[i] for i in range(len(ds))):
        assert obs.derivative_source=='augmentation' and obs.timestamp==0
        assert np.linalg.norm(obs.velocity)>1e-4
        q,v=recover_state(np.vstack((obs.start_pva,obs.start_jerk)),.06)
        assert abs(q[3:7]@obs.configuration[3:7])>1-1e-8
        assert np.allclose(v,obs.velocity,atol=1e-7)
        sim=MujocoSimulation(priv.scene_path,planning_path_capacity=0,record_actual_trajectory=False)
        sim.robot.reset(obs.configuration.copy(),obs.velocity.copy())
        with IdealDepthObserver(sim.model,sim.data,spec=obs.depth_spec) as camera:
            depth,mask=camera.observe()
        assert np.array_equal(mask,obs.depth_valid) and np.allclose(depth,obs.depth_m,atol=1e-6)
