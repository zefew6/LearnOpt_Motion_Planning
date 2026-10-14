import subprocess
import sys
import numpy as np
import pytest
from uav_ac.yopo.whole_body.trajectory.parameterization import InteriorMap
from uav_ac.yopo.whole_body.trajectory.parameterization import decode_trajectory


def boundaries():
    head=np.zeros((3,8));head[0,:3]=[0.,0.,-1.5]
    goal=head[0].copy();goal[0]=1.;goal[3]=.2
    tail=np.vstack((goal,np.zeros((2,8))))
    return head,tail


def test_decoder_hard_boundaries_jerk_continuity_and_pullback():
    head,tail=boundaries();mapping=InteriorMap(head,tail)
    free=mapping.seed();free[0]+=.05
    tr=decode_trajectory(free,head,tail[0],gripper_gap=.06)
    for d in range(3):
        assert np.allclose(tr.evaluate(0,d),head[d],atol=1e-8)
        assert np.allclose(tr.evaluate(3,d),tail[d],atol=1e-8)
    assert np.max(abs(tr.evaluate([0,3],3)[:,:2]))<1e-7
    for time in [.75,1.5,2.25]:
        for d in range(3): assert np.allclose(tr.evaluate(time-1e-9,d),tr.evaluate(time+1e-9,d),atol=1e-6)
    rng=np.random.default_rng(5);g=rng.normal(size=(3,8));direction=rng.normal(size=20)
    fd=np.sum(g*(mapping.decode(free+1e-6*direction)-mapping.decode(free-1e-6*direction)))/2e-6
    assert fd==pytest.approx(mapping.pullback(g)@direction,rel=1e-7)
    assert tr.rrt_path.shape==(0,8) and not tr.validation_performed
    with pytest.raises(ValueError): mapping.decode(np.full(20,np.nan))


def test_decoder_import_does_not_load_privileged_modules():
    script='''from uav_ac.yopo.whole_body.trajectory.parameterization import decode_trajectory
import sys
assert not any(x in sys.modules for x in ['uav_ac.yopo.whole_body.training.oracle','uav_ac.planning.geometry.esdf','uav_ac.planning.search.rrt_connect'])
from uav_ac.planning.trajectory import GCOPTER
from uav_ac.planning.trajectory.aerial_manipulator_minco import AerialManipulatorMINCO
'''
    subprocess.run([sys.executable,'-c',script],check=True)


def test_decoder_free_parameters_are_local_and_path_stays_near_endpoints():
    from uav_ac.yopo.whole_body.model.encoding import LocalFrame
    head,tail=boundaries();frame=LocalFrame(head[0,:3],head[0,3])
    mapping=InteriorMap(frame.local_pva(head),np.vstack((frame.local_goal(tail[0]),np.zeros((2,8)))))
    tr=decode_trajectory(mapping.seed(),head,tail[0],gripper_gap=.06)
    assert np.max(abs(tr.evaluate(np.linspace(0,3,33))[:,2]+1.5))<1e-8
