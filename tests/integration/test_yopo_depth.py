import numpy as np
import mujoco
import pytest
from uav_ac.planning.end2end.contracts import DepthSpec
from uav_ac.simulation.depth_observation import IdealDepthObserver


def make_model(rear=3.):
    return mujoco.MjModel.from_xml_string(f'''<mujoco><visual><map znear="0.0001" zfar="100"/></visual>
    <worldbody><camera name="depth" pos="0 0 0" xyaxes="0 -1 0 0 0 1"
    sensorsize="2 2" resolution="160 96" focalpixel="80 83.1384387633"/>
    <geom name="wall" type="box" pos="2.05 0 0" size=".05 5 5"/>
    <geom name="rear" type="box" pos="{rear} 0 0" size=".05 5 5"/>
    </worldbody></mujoco>''')


def test_real_depth_distance_occlusion_and_state_unchanged():
    model=make_model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    snapshot=(data.qpos.copy(),data.qvel.copy(),float(data.time))
    with IdealDepthObserver(model,data,camera_name='depth',spec=DepthSpec()) as observer:
        depth,mask=observer.observe()
        assert np.max(abs(depth-2.))<.02
        assert mask.all()
        rear=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'rear')
        model.geom_pos[rear,0]=4.;mujoco.mj_forward(model,data)
        after,_=observer.observe()
        assert np.array_equal(depth,after)
        assert observer.intrinsics[0,0]==pytest.approx(80.)
        assert observer.intrinsics[1,1]==pytest.approx(83.1384387633)
    assert np.array_equal(snapshot[0],data.qpos) and np.array_equal(snapshot[1],data.qvel)
    assert data.time==snapshot[2]


def test_no_return_and_near_return_masks():
    model=make_model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    with IdealDepthObserver(model,data,camera_name='depth',spec=DepthSpec(far_m=1.)) as observer:
        depth,mask=observer.observe()
        assert not mask.any() and np.all(depth==1.)
    model.geom_pos[0,0]=.08;mujoco.mj_forward(model,data)
    with IdealDepthObserver(model,data,camera_name='depth',spec=DepthSpec()) as observer:
        depth,mask=observer.observe()
        assert mask.all() and np.all(depth==.1)


def test_off_axis_depth_matches_first_ray_hit():
    model=make_model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    spec=DepthSpec()
    with IdealDepthObserver(model,data,camera_name='depth',spec=spec) as observer:
        depth,_=observer.observe();k=spec.intrinsics
        optical_to_world=data.cam_xmat[0].reshape(3,3)@np.diag([1.,-1.,-1.])
        for row,col in [(10,10),(40,120),(80,150)]:
            ray=np.array([(col+.5-k[0,2])/k[0,0],(row+.5-k[1,2])/k[1,1],1.])
            norm=np.linalg.norm(ray);hit=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(model,data,data.cam_xpos[0],optical_to_world@ray/norm,None,1,-1,hit)
            assert distance>0 and depth[row,col]==pytest.approx(distance/norm,abs=.02)


def test_debug_geometry_removed_without_hiding_physical_robot_or_mutating_model():
    xml='''<mujoco><visual><map znear=".0001" zfar="100"/></visual><worldbody>
    <camera name="depth" xyaxes="0 -1 0 0 0 1" sensorsize="2 2" resolution="160 96" focalpixel="80 83.1384387633"/>
    <geom name="wall" type="box" pos="2.05 0 0" size=".05 5 5"/>
    <geom name="pick_target_marker" type="box" pos="1.05 0 0" size=".05 5 5" contype="0" conaffinity="0"/>
    </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    groups=model.geom_group.copy()
    with IdealDepthObserver(model,data,camera_name='depth') as observer:
        depth,_=observer.observe();assert np.allclose(depth,2.,atol=.02)
    assert np.array_equal(model.geom_group,groups)


def test_clipping_and_noncentral_projection_rejected():
    empty="""<mujoco><statistic extent='1'/><visual><map znear='.0001' zfar='2'/></visual>
    <worldbody><camera name='depth' xyaxes='0 -1 0 0 0 1' sensorsize='2 2' resolution='160 96' focalpixel='80 83.1384387633'/></worldbody></mujoco>"""
    model=mujoco.MjModel.from_xml_string(empty);data=mujoco.MjData(model)
    with pytest.raises(ValueError,match='clip'):IdealDepthObserver(model,data,camera_name='depth')
    model=make_model();model.cam_intrinsic[0,2]=.1;data=mujoco.MjData(model)
    with pytest.raises(ValueError,match='principal'):IdealDepthObserver(model,data,camera_name='depth')
    model=make_model();model.vis.map.znear=.5;data=mujoco.MjData(model)
    with pytest.raises(ValueError,match='clip'):IdealDepthObserver(model,data,camera_name='depth')


def test_body_camera_mount_must_match_declared_optical_extrinsic():
    model=mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name='quadrotor'>
    <camera name='depth' pos='1 0 0' xyaxes='0 -1 0 0 0 1' sensorsize='2 2' resolution='160 96' focalpixel='80 83.1384387633'/>
    <geom type='box' pos='2 0 0' size='.05 1 1'/></body></worldbody></mujoco>""")
    data=mujoco.MjData(model)
    with pytest.raises(ValueError,match='mount'):IdealDepthObserver(model,data,camera_name='depth')


def test_robot_probe_named_camera_center_distance():
    from pathlib import Path
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    sim=MujocoSimulation(Path('uav_ac/simulation/models/yopo_depth_probe.xml'),planning_path_capacity=0,record_actual_trajectory=False)
    with IdealDepthObserver(sim.model,sim.data) as observer:
        depth,mask=observer.observe()
        assert mask[48,80] and depth[48,80]==pytest.approx(2.,abs=.02)
