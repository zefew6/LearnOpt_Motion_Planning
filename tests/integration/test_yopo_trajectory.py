from tests.yopo_scenes import scene_path
from pathlib import Path
import numpy as np
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.end2end.contracts import PlannerObservation
from uav_ac.planning.end2end.encoding import LocalFrame
from uav_ac.planning.end2end.parameterization import InteriorMap
from uav_ac.planning.end2end.decoder import decode_trajectory
from uav_ac.YOPO.privileged import PrivilegedEvaluator


def test_real_robot_reference_endpoints_and_exact_geometry():
    sim=MujocoSimulation(scene_path('free'),planning_path_capacity=0,record_actual_trajectory=False)
    q=sim.robot.configuration.copy();q[:3]=[0.,0.,-1.5];q[7:11]=0.;q[11]=.06
    sim.robot.reset(q,np.zeros(11))
    y=np.r_[q[:3],0.,q[7:11]];head=np.vstack((y,np.zeros((2,8))));goal=y.copy();goal[0]=.8
    obs=PlannerObservation(np.full((96,160),6.),np.zeros((96,160),bool),q,np.zeros(11),head,goal)
    frame=LocalFrame.from_observation(obs);mapping=InteriorMap(frame.local_pva(head),np.vstack((frame.local_goal(goal),np.zeros((2,8)))))
    free=mapping.seed();trajectory=decode_trajectory(free,head,goal,gripper_gap=.06)
    snapshot=sim.data.qpos.copy()
    for t in np.linspace(0,3,121):
        ref=trajectory.reference(t,sim.robot,gripper_opening=.06)
        result=sim.robot.check_collision(ref.configuration,clearance=0.)
        assert not result['collision']
    assert np.array_equal(snapshot,sim.data.qpos)
    for t in (0,3):
        ref=trajectory.reference(t,sim.robot,gripper_opening=.06)
        assert np.max(abs(ref.velocity))<1e-7
    oracle=PrivilegedEvaluator.from_simulation(sim)
    batch=oracle.evaluate(obs,free[None])
    assert np.isfinite(batch.costs).all() and np.isfinite(batch.free_gradients).all()
