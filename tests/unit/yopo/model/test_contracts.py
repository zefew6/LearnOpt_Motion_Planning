import numpy as np
import pytest
from uav_ac.yopo.whole_body.contracts import DepthSpec, PlannerObservation
from uav_ac.yopo.whole_body.model.encoding import LocalFrame, Normalization, encode_observation


def observation(yaw=0.):
    y=np.array([1.,2.,-1.,yaw,0.,0.,0.,0.])
    q=np.r_[y[:3],np.cos(yaw/2),0.,0.,np.sin(yaw/2),y[4:],.06]
    return PlannerObservation(np.full((96,160),2.),np.ones((96,160),bool),q,
                              np.zeros(11),np.vstack((y,np.zeros((2,8)))),y.copy())


def test_observation_rejects_nonfinite_shape_and_quaternion():
    obs=observation()
    for field,value in [('goal',np.zeros(7)),('velocity',np.full(11,np.nan)),
                        ('configuration',np.zeros(12)),('depth_valid',np.zeros((2,2),bool))]:
        args={k:getattr(obs,k) for k in ('depth_m','depth_valid','configuration','velocity','start_pva','goal')}
        args[field]=value
        with pytest.raises(ValueError): PlannerObservation(**args)
    with pytest.raises(ValueError): DepthSpec(hfov_deg=180.)
    with pytest.raises(ValueError): DepthSpec(width=True)


def test_frame_roundtrip_and_body_rates_not_world_rotated():
    frame=LocalFrame(np.array([1.,2.,-3.]),.7)
    points=np.array([[2.,3.,-1.],[4.,5.,-2.]])
    assert np.max(abs(frame.to_world_points(frame.to_local_points(points))-points))<1e-12
    assert np.max(abs(frame.to_world_vectors(frame.to_local_vectors(points))-points))<1e-12
    obs=observation(.7)
    enc=encode_observation(obs,Normalization((-2.,)*4,(2.,)*4))
    assert enc.depth.shape==(1,2,96,160)
    assert np.all(np.isfinite(enc.state))
    flipped=np.array(obs.configuration);flipped[3:7]*=-1
    other=PlannerObservation(obs.depth_m,obs.depth_valid,flipped,obs.velocity,obs.start_pva,obs.goal)
    assert np.allclose(enc.state,encode_observation(other,Normalization((-2.,)*4,(2.,)*4)).state)


def test_yaw_wrap_absolute_joints_and_limit_validation():
    obs=observation(np.deg2rad(179.))
    goal=obs.goal.copy();goal[3]=np.deg2rad(-179.)
    changed=PlannerObservation(obs.depth_m,obs.depth_valid,obs.configuration,obs.velocity,obs.start_pva,goal)
    local=LocalFrame.from_observation(changed).local_goal(goal)
    assert local[3]==pytest.approx(np.deg2rad(2.))
    goal[4]=3.
    changed=PlannerObservation(obs.depth_m,obs.depth_valid,obs.configuration,obs.velocity,obs.start_pva,goal)
    with pytest.raises(ValueError,match='joint'): encode_observation(changed,Normalization((-2.,)*4,(2.,)*4))
    with pytest.raises(ValueError): Normalization((0.,)*4,(0.,)*4)


def test_nonbinary_depth_precision_and_camera_roundtrip():
    spec=DepthSpec(far_m=5.1)
    obs=observation();args={k:getattr(obs,k) for k in ('configuration','velocity','start_pva','goal')}
    supplied=PlannerObservation(np.full((96,160),5.1,dtype=np.float32),np.zeros((96,160),bool),**args,depth_spec=spec)
    frame=LocalFrame.from_observation(supplied);pose=frame.camera_to_local(supplied.configuration,spec)
    point=np.array([.2,.1,2.,1.]);assert np.allclose(np.linalg.inv(pose)@(pose@point),point)


def test_start_and_target_absolute_joints_cannot_be_replaced_by_delta():
    first=observation();q=first.configuration.copy();q[7:11]=.3
    head=first.start_pva.copy();head[0,4:]=.3;goal=first.goal.copy();goal[4:]=.3
    second=PlannerObservation(first.depth_m,first.depth_valid,q,first.velocity,head,goal)
    norm=Normalization((-2.,)*4,(2.,)*4)
    assert not np.allclose(encode_observation(first,norm).state,encode_observation(second,norm).state)
