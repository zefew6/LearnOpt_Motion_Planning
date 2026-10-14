import numpy as np
import torch
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state


def test_torch_fk_matches_public_robot_positions_and_jacobians():
    from uav_ac.yopo.whole_body.model.kinematics import TorchKinematics, flatness_rotation
    sim=MujocoSimulation('uav_ac/simulation/models/yopo_bamboo.xml',planning_path_capacity=0,record_actual_trajectory=False)
    kin=TorchKinematics(sim.robot.kinematic_description())
    geoms=sim.robot.collision_geometries();names=[g['body'] for g in geoms]
    points=torch.tensor(np.stack([g['local_start'] for g in geoms]),dtype=torch.float64)
    sigma=torch.tensor([1.,2.,-2.,.4,.3,-1.2,1.4,-.2],dtype=torch.float64,requires_grad=True)
    acceleration=torch.tensor([.3,-.6,.4],dtype=torch.float64,requires_grad=True)
    rotation=flatness_rotation(acceleration,sigma[3])
    actual=kin.points(sigma[:3],rotation,sigma[4:],.06,names,points)
    derivatives=np.zeros((4,8));derivatives[0]=sigma.detach().numpy();derivatives[2,:3]=acceleration.detach().numpy()
    q,_=recover_state(derivatives,.06)
    expected,jac=sim.robot.point_positions_and_pose_jacobians(q,names,points.numpy())
    assert np.allclose(actual.detach().numpy(),expected,atol=1e-9)
    assert torch.autograd.gradcheck(lambda s,a:kin.points(s[:3],flatness_rotation(a,s[3]),s[4:],.06,names,points),(sigma,acceleration),atol=1e-5)
