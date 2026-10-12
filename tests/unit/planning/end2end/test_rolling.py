import numpy as np
import pytest
from uav_ac.planning.end2end.contracts import PlannerObservation
from uav_ac.planning.end2end.rolling import RollingMap, decode_rolling, observation_at, ReferenceSchedule
from uav_ac.planning.end2end.encoding import LocalFrame
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state


def moving_observation():
    derivatives=np.zeros((4,8));derivatives[0,:3]=[0,0,-1.5]
    derivatives[0,3]=.3;derivatives[1,:3]=[.4,.1,0]
    derivatives[2,:3]=[.6,.2,0];derivatives[3,:3]=[.2,-.1,.3]
    derivatives[1,3]=.1;derivatives[1,4:]=[.1,0,0,0]
    q,v=recover_state(derivatives,.06);goal=derivatives[0].copy();goal[0]+=2.
    return PlannerObservation(np.full((96,160),6.),np.zeros((96,160),bool),q,v,derivatives[:3],goal,
        timestamp=.5,start_jerk=derivatives[3],derivative_source='reference')


def test_rolling_map_and_decode_match_dynamic_head_without_stopping_tail():
    obs=moving_observation();frame=LocalFrame.from_observation(obs)
    head=np.vstack((frame.local_pva(obs.start_pva),obs.start_jerk.copy()))
    head[3,:3]=frame.to_local_vectors(obs.start_jerk[:3])
    mapping=RollingMap(head);free=mapping.seed(frame.local_goal(obs.goal))
    assert free.shape==(32,)
    tr=decode_rolling(free,obs)
    for d in range(3):assert np.allclose(tr.evaluate(0,d),obs.start_pva[d],atol=1e-8)
    q,v=recover_state(np.stack([tr.evaluate(0,d) for d in range(4)]),.06)
    assert abs(q[3:7]@obs.configuration[3:7])==pytest.approx(1.,abs=1e-8)
    assert np.allclose(v,obs.velocity,atol=1e-7)
    assert np.linalg.norm(tr.evaluate(tr.total_time,1))>.1
    assert not np.allclose(tr.evaluate(tr.total_time),obs.goal)
    rng=np.random.default_rng(2);gp=rng.normal(size=(1,8));gt=rng.normal(size=(3,8));d=rng.normal(size=32)
    def value(z):
        points,tail,times=mapping.decode(z);return np.sum(gp*points)+np.sum(gt*tail)
    fd=(value(free+1e-6*d)-value(free-1e-6*d))/2e-6
    assert fd==pytest.approx(mapping.pullback(gp,gt,np.zeros(2),free)@d,rel=1e-7)


def test_nominal_reference_replanning_handoff_and_history_time():
    obs=moving_observation();frame=LocalFrame.from_observation(obs)
    mapping=RollingMap.from_observation(obs);first=decode_rolling(mapping.seed(frame.local_goal(obs.goal)),obs)
    schedule=ReferenceSchedule(first,obs.timestamp)
    next_obs=observation_at(first,.1,obs.goal,obs.depth_m,obs.depth_valid,timestamp=obs.timestamp+.1)
    assert next_obs.timestamp>.5 and np.linalg.norm(next_obs.velocity)>0
    next_map=RollingMap.from_observation(next_obs)
    second=decode_rolling(next_map.seed(LocalFrame.from_observation(next_obs).local_goal(obs.goal)),next_obs)
    schedule.handoff(second,next_obs.timestamp)
    with pytest.raises(ValueError):schedule.handoff(first,0.)
    with pytest.raises(TimeoutError):schedule.derivatives_at(100.)


def test_timestamp_and_source_validation():
    obs=moving_observation()
    with pytest.raises(ValueError):
        PlannerObservation(obs.depth_m,obs.depth_valid,obs.configuration,obs.velocity,obs.start_pva,obs.goal,
                           timestamp=-1,start_jerk=obs.start_jerk,derivative_source='reference')


def test_coefficients_match_pinned_official_yopo_minco_matrix():
    import runpy
    from uav_ac.planning.trajectory.gcopter.minco import MINCOQuintic
    build=runpy.run_path('tests/fixtures/yopo_minco_reference.py')['build_A']
    obs=moving_observation();mapping=RollingMap.from_observation(obs)
    free=mapping.seed(LocalFrame.from_observation(obs).local_goal(obs.goal))
    points,tail,durations=mapping.decode(free)
    coefficients,_=MINCOQuintic(mapping.head[:3],tail,2).solve(points,durations)
    rhs=np.zeros((12,3));rhs[:3]=mapping.head[:3,:3];rhs[3]=points[0,:3];rhs[9:]=tail[:,:3]
    reference=np.linalg.solve(build(np.array([durations[0]]),np.array([durations[1]]))[0],rhs)
    assert np.allclose(coefficients[:,:3],reference,rtol=1e-8,atol=1e-8)


def test_history_rejects_backwards_observations_and_keeps_only_observed_frames():
    from dataclasses import replace
    from uav_ac.planning.end2end.rolling import ObservationHistory
    first=moving_observation();history=ObservationHistory(2)
    history.append(first);history.append(replace(first,timestamp=.6));history.append(replace(first,timestamp=.7))
    assert [frame.timestamp for frame in history.frames]==[.6,.7]
    with pytest.raises(ValueError):history.append(first)


def test_schedule_rejects_stale_nan_and_non_scalar_clock():
    obs=moving_observation();mapping=RollingMap.from_observation(obs)
    plan=decode_rolling(mapping.seed(LocalFrame.from_observation(obs).local_goal(obs.goal)),obs)
    schedule=ReferenceSchedule(plan,.5);schedule.derivatives_at(.8)
    for value in [.6,np.nan,np.inf,np.array([.9])]:
        with pytest.raises(ValueError):schedule.derivatives_at(value)
    with pytest.raises(ValueError):schedule.handoff(plan,.7)


def test_observation_timestamp_cannot_alias_mutable_array():
    from dataclasses import replace
    obs=moving_observation();value=np.array([.5])
    with pytest.raises(ValueError):replace(obs,timestamp=value)
    scalar=replace(obs,timestamp=np.float64(.6));assert type(scalar.timestamp) is float
