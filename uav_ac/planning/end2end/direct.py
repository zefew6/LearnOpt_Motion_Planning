"""Fixed-semantics two-piece decoder matching YOPO-MINCO's variable roles.

34 physical parameters: inner8, terminal PVA24, durations2. Actual pose/rates
are observation features, not an exact-flatness admission check. The supplied
head PVA is honored; model mismatch is diagnosed by the training/evaluation layer.
"""
import numpy as np
from .contracts import finite_array
from .encoding import LocalFrame
from ..trajectory.gcopter.minco import MINCOQuintic
from ..trajectory.aerial_manipulator_minco.trajectory import AerialManipulatorTrajectory

PARAMETER_DIM = 34
PROFILE = 'direct_v3'


def unpack_parameters(parameters):
    z=finite_array(parameters,(34,),'trajectory_parameters')
    if np.any(z[-2:]<.1) or np.any(z[-2:]>3.):
        raise ValueError('segment durations must lie in [0.1,3]')
    return z[:8][None],z[8:32].reshape(3,8),z[-2:]


def world_problem(parameters,obs):
    points,tail,durations=unpack_parameters(parameters)
    frame=LocalFrame.from_observation(obs)
    world_tail=tail.copy()
    world_tail[0,:3]=frame.to_world_points(tail[0,:3])
    world_tail[1:,:3]=frame.to_world_vectors(tail[1:,:3])
    world_tail[0,3]+=frame.yaw
    minco=MINCOQuintic(obs.start_pva,world_tail,2)
    return minco,frame.world_points8(points),durations,world_tail,frame


def decode_direct(parameters,obs):
    minco,points,durations,_,_=world_problem(parameters,obs)
    coeff,_=minco.solve(points,durations)
    return AerialManipulatorTrajectory(durations,coeff.reshape(2,6,8),np.empty((0,8)),
                                      float('nan'),0,False,'direct_v3; no collision certification')
