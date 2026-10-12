"""Deterministic map-free trajectory decoding; no optimizer or search calls."""
import numpy as np
from .parameterization import InteriorMap
from .contracts import finite_array
from .encoding import LocalFrame
from ..trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory


def decode_trajectory(free,head_pva,goal,*,durations=None,gripper_gap):
    goal=finite_array(goal,(8,),'goal')
    if not np.isfinite(gripper_gap) or not .02<=gripper_gap<=.07:
        raise ValueError('gripper_gap outside model limits')
    tail=np.vstack((goal,np.zeros((2,8))))
    head=finite_array(head_pva,(3,8),'head_pva')
    frame=LocalFrame(head[0,:3],float(head[0,3]))
    local_head=frame.local_pva(head)
    local_tail=np.vstack((frame.local_goal(goal),np.zeros((2,8))))
    mapping=InteriorMap(local_head,local_tail,durations)
    points=frame.world_points8(mapping.decode(free))
    world_tail=local_tail.copy();world_tail[0,:3]=frame.to_world_points(local_tail[0,:3]);world_tail[0,3]+=frame.yaw
    world_minco=InteriorMap(head,world_tail,mapping.durations).minco
    flat,_=world_minco.solve(points,mapping.durations)
    return AerialManipulatorTrajectory(mapping.durations,flat.reshape(4,6,8),np.empty((0,8)),
                                      float('nan'),0,False,'network decode; not optimized or validated')
