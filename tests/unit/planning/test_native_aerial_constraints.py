"""Native arithmetic equivalence without replacing exact geometry callbacks."""
from types import SimpleNamespace

import numpy as np
import pytest

from uav_ac.planning.trajectory.aerial_manipulator_minco import constraints as module
from uav_ac.planning.trajectory.aerial_manipulator_minco import constraints as constraints_module
from uav_ac.planning.trajectory.aerial_manipulator_minco import collision as collision_module
from uav_ac.planning.trajectory.aerial_manipulator_minco import flatness as flatness_module
from uav_ac.planning.trajectory.aerial_manipulator_minco.config import AerialManipulatorMINCOConfig

native = pytest.importorskip('uav_ac.planning.native._aerial_constraints')


def evaluator():
    obj = module.AerialManipulatorTrajectoryEvaluator.__new__(module.AerialManipulatorTrajectoryEvaluator)
    obj.config = AerialManipulatorMINCOConfig(
        max_speed=.4, max_acceleration=.7, max_body_rate=.2,
        max_yaw_rate=.3, max_yaw_acceleration=.4,
        joint_velocity_limits=(.2,)*4, joint_acceleration_limits=(.3,)*4,
        integral_resolution=5)
    obj.quad = SimpleNamespace(g=9.81, min_thrust=.1, max_thrust=8., max_tilt_angle=.2)
    obj.mass = 1.3
    obj.robot = SimpleNamespace(limits=SimpleNamespace(
        joint_lower=np.full(4,-.5), joint_upper=np.full(4,.5)))
    obj.bounds = np.array([[-1.,-1.,-2.], [1.,1.,0.]])
    obj.objective_samples = 0
    obj.carry_payload = False
    obj.collision_cost_gradient = lambda *args: (0.,np.zeros(8),np.zeros(8),-np.inf,np.inf,-np.inf,-np.inf)
    return obj


@pytest.mark.parametrize('case', ['random','hover','regularized','near_hover','heading_singular'])
def test_rate_and_tangent_reference(case):
    rng = np.random.default_rng(818)
    for _ in range(20):
        acceleration = rng.normal(size=3)
        if case == 'hover': acceleration[:] = 0
        if case == 'near_hover': acceleration *= 1e-7
        if case == 'regularized': acceleration = np.array([0.,0.,9.81])
        jerk=rng.normal(size=3); yaw=rng.normal(); yaw_rate=rng.normal()
        if case == 'heading_singular': acceleration=np.array([-1.,0.,9.81]); yaw=0.
        actual=native.flatness_body_rate_squared(acceleration,jerk,yaw,yaw_rate,9.81)
        expected=module._flatness_body_rate_squared_python(acceleration,jerk,yaw,yaw_rate,9.81)
        np.testing.assert_allclose(actual[0],expected[0],rtol=3e-12,atol=1e-12)
        np.testing.assert_allclose(actual[1],expected[1],rtol=3e-12,atol=1e-10)
        actual=native.flatness_attitude_tangent_jacobian(acceleration,yaw,9.81)
        expected=flatness_module._flatness_attitude_tangent_jacobian(acceleration,yaw,9.81)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=2e-12,atol=1e-10)


def test_physical_constraints_random_reference_and_finite_difference():
    obj=evaluator(); rng=np.random.default_rng(81)
    for _ in range(40):
        args=rng.normal(size=(4,8))
        expected=obj._physical_cost_gradient_python(*args)
        actual=obj._physical_cost_gradient(*args)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=3e-12,atol=1e-9)
    for order in range(4):
        for dimension in range(8):
            plus=args.copy(); minus=args.copy(); step=1e-6
            plus[order,dimension]+=step; minus[order,dimension]-=step
            fd=(obj._physical_cost_gradient(*plus)[0]-obj._physical_cost_gradient(*minus)[0])/(2*step)
            np.testing.assert_allclose(actual[1][order][dimension],fd,rtol=3e-5,atol=.003)


@pytest.mark.parametrize('pieces,resolution',[(1,1),(3,7),(7,11)])
def test_integration_reference_finite_difference_and_callback_ownership(pieces,resolution):
    obj=evaluator(); rng=np.random.default_rng(819)
    durations=rng.uniform(.3,1.1,size=pieces)
    coefficients=rng.normal(scale=.2,size=(pieces,6,8))
    retained=[]
    def callback(*args):
        retained.append((args,[a.copy() for a in args]))
        # A coupled smooth polynomial verifies every derivative order and time chain rule.
        cost=sum((order+1)*np.dot(a,a)/2 for order,a in enumerate(args))
        return cost,[(order+1)*a for order,a in enumerate(args)],float(args[0][0]),float(args[0][1])
    obj.sample_cost_gradient=callback
    obj.config=SimpleNamespace(**vars(obj.config))
    obj.config.integral_resolution=resolution
    obj.config.integral_resolution_floor_unloaded=resolution
    expected=obj._integrated_penalty_python(durations,coefficients)
    expected_metrics=(obj.last_violation,obj.minimum_clearance)
    retained.clear()
    actual=obj.integrated_penalty(durations,coefficients)
    assert len(retained)==pieces*(resolution+1)
    for inputs,copies in retained:
        for a,b in zip(inputs,copies): np.testing.assert_array_equal(a,b)
    for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=3e-13,atol=3e-12)
    np.testing.assert_allclose((obj.last_violation,obj.minimum_clearance),expected_metrics)
    for piece in range(pieces):
        plus=durations.copy();minus=durations.copy();step=1e-6
        plus[piece]+=step;minus[piece]-=step
        fd=(obj.integrated_penalty(plus,coefficients)[0]-obj.integrated_penalty(minus,coefficients)[0])/(2*step)
        np.testing.assert_allclose(actual[2][piece],fd,rtol=2e-7,atol=2e-6)
    for index in [(0,0,0),(pieces-1,5,7),(0,3,4)]:
        plus=coefficients.copy();minus=coefficients.copy();step=1e-6
        plus[index]+=step;minus[index]-=step
        fd=(obj.integrated_penalty(durations,plus)[0]-obj.integrated_penalty(durations,minus)[0])/(2*step)
        np.testing.assert_allclose(actual[1][index],fd,rtol=2e-7,atol=2e-6)


def test_native_missing_keeps_reference_fallback(monkeypatch):
    obj=evaluator();rng=np.random.default_rng(77);args=rng.normal(size=(4,8))
    expected=obj.sample_cost_gradient(*args)
    monkeypatch.setattr(constraints_module,'_native_aerial',None)
    actual=obj.sample_cost_gradient(*args)
    for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=3e-12,atol=1e-9)


def test_rate_gradient_and_tangent_finite_difference_near_hover():
    for acceleration in (np.array([.7,-.3,.2]),np.array([1e-5,-2e-5,0.])):
        jerk=np.array([.4,.1,-.2]); yaw=.6; yaw_rate=-.25
        variables=np.r_[acceleration,jerk,yaw,yaw_rate]
        _,gradient=native.flatness_body_rate_squared(acceleration,jerk,yaw,yaw_rate,9.81)
        numerical=[]
        for index in range(8):
            plus=variables.copy();minus=variables.copy();step=1e-6
            plus[index]+=step;minus[index]-=step
            numerical.append((native.flatness_body_rate_squared(plus[:3],plus[3:6],plus[6],plus[7],9.81)[0]
                              -native.flatness_body_rate_squared(minus[:3],minus[3:6],minus[6],minus[7],9.81)[0])/(2*step))
        np.testing.assert_allclose(gradient,numerical,rtol=2e-6,atol=2e-8)
        rotation,tangent=native.flatness_attitude_tangent_jacobian(acceleration,yaw)
        for index in range(4):
            plus=np.r_[acceleration,yaw];minus=plus.copy();step=1e-6
            plus[index]+=step;minus[index]-=step
            rp=native.flatness_attitude_tangent_jacobian(plus[:3],plus[3])[0]
            rm=native.flatness_attitude_tangent_jacobian(minus[:3],minus[3])[0]
            skew=((rp-rm)/(2*step))@rotation.T
            np.testing.assert_allclose(tangent[:,index],[skew[2,1],skew[0,2],skew[1,0]],rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize('violation',[-.01,0.,.0025,.01,.02])
def test_smoothing_boundaries_and_hover_physical_equivalence(violation):
    obj=evaluator();args=np.zeros((4,8));args[0,2]=-1
    args[1,0]=np.sqrt(obj.config.max_speed**2+violation)
    args[0,4]=obj.robot.limits.joint_upper[0]+violation
    for acceleration in (np.zeros(3),np.array([1e-8,-1e-8,0.]),np.array([0.,0.,9.81])):
        args[2,:3]=acceleration
        actual=obj._physical_cost_gradient(*args)
        expected=obj._physical_cost_gradient_python(*args)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=3e-12,atol=1e-7)


@pytest.mark.parametrize('count',[0,1,137])
@pytest.mark.parametrize('boundary',[False,True])
def test_exact_collision_accumulation_random_boundaries_and_immutable_inputs(count,boundary):
    rng=np.random.default_rng(190)
    kinds=np.resize(np.array([1,2,3],dtype=np.int8),count)
    margins=np.where(kinds==1,.35,.04)
    violations=np.resize(np.array([-.02,0.,.0025,.005,.01,.02]),count)
    if not boundary: violations=rng.uniform(-.1,.1,size=count)
    distances=margins-violations
    jacobians=rng.normal(size=(count,10))
    tangent=rng.normal(size=(3,4))
    state=rng.normal(size=8); acceleration=rng.normal(size=8)
    inputs=(12.5,state,acceleration,distances,jacobians,kinds,tangent,
            .35,.04,20000.,10000.,.01,-.4,-.2,-.1,.1)
    copies=[a.copy() for a in inputs if isinstance(a,np.ndarray)]
    expected=collision_module._exact_collision_penalty_python(*inputs)
    actual=native.exact_collision_penalty(*inputs)
    for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=2e-13,atol=2e-10)
    for a,b in zip((a for a in inputs if isinstance(a,np.ndarray)),copies):
        np.testing.assert_array_equal(a,b)
    if count:
        assert actual[4] == pytest.approx(float(np.min(distances-margins)))


@pytest.mark.parametrize('carry_payload',[False,True])
def test_actual_collision_native_reference_parity(carry_payload,monkeypatch):
    from uav_ac.planning.geometry.esdf import ESDF
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    sim=MujocoSimulation('tests/fixtures/aerial_manipulator_gradient.xml',record_actual_trajectory=False)
    esdf=ESDF.from_axis_aligned_boxes(np.empty((0,6)),[-2.,-2.,-3.],[2.,2.,1.],.2,ground_height=0.)
    cfg=AerialManipulatorMINCOConfig(obstacle_clearance=.35)
    obj=module.AerialManipulatorTrajectoryEvaluator(sim.robot,esdf,sim.quad,cfg,sim.space_limits,.06,carry_payload)
    rng=np.random.default_rng(390)
    metadata=[obj.self_pairs.copy(),obj.sphere_radii.copy()]
    for height in (-1.,-.35,-.2):
        sigma=rng.normal(scale=.025,size=8);sigma[2]=height
        acceleration=rng.normal(scale=.1,size=8)
        monkeypatch.setattr(collision_module,'_native_aerial',native)
        actual=obj.collision_cost_gradient(sigma,acceleration)
        monkeypatch.setattr(collision_module,'_native_aerial',None)
        expected=obj.collision_cost_gradient(sigma,acceleration)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-8)
    np.testing.assert_array_equal(obj.self_pairs,metadata[0])
    np.testing.assert_array_equal(obj.sphere_radii,metadata[1])
