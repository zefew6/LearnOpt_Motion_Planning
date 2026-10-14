import numpy as np


def test_frame_conversion_and_height_preserving_goal_clip():
    from uav_ac.yopo.uav.deployment import body_observation,preprocess_depth
    state=np.r_[np.array([1.,2.,-1.5]),[1.,0.,0.,0.],[2.,.3,-.2],[0.,0.,0.]]
    goal=np.array([31.,2.,-3.5]);acc=np.array([.1,.2,.3])
    obs,rotation=body_observation(state,acc,goal,10.)
    np.testing.assert_allclose(obs[:3],[2.,-.3,.2])
    np.testing.assert_allclose(obs[3:6],[.1,-.2,-.3])
    assert np.isclose(obs[8],2.) and np.isclose(np.linalg.norm(obs[6:]),10.)
    np.testing.assert_array_equal(rotation,np.eye(3))
    image=preprocess_depth(np.full((96,160),10.,dtype=np.float32))
    assert image.shape==(1,1,96,160) and np.all(image==127/255.)


def test_official_corridor_selection_and_brake():
    from uav_ac.yopo.uav.deployment import select_candidate
    scores=np.array([3.,2.,1.]);inner=np.zeros((3,3));radius=np.zeros((3,20));radius[:,10:]=.05
    radius[:,0:10]=[[.02]*10,[.2]*10,[.4]*10]
    index,brake=select_candidate(scores,inner,radius)
    assert index==1 and not brake
    radius[:,:10]=0.
    index,brake=select_candidate(scores,inner,radius)
    assert index==0 and brake


def test_warmup_preserves_selection_history():
    import torch
    from uav_ac.yopo.uav.deployment import UavPlanner
    from uav_ac.yopo.uav.core import OfficialCore
    planner=object.__new__(UavPlanner);planner.core=OfficialCore(device='cpu')
    planner.previous_inner=np.array([1.,2.,3.]);planner.generation=4
    planner.warm_up()
    np.testing.assert_array_equal(planner.previous_inner,[1.,2.,3.])
    assert planner.generation==4


def test_upstream_native_depth_resize_uses_nearest_pixel_indices():
    from uav_ac.yopo.uav.deployment import preprocess_depth
    depth=np.broadcast_to(np.linspace(1.,19.,90,dtype=np.float32)[:,None],(90,160))
    result=preprocess_depth(depth)[0,0]
    expected=(np.uint8(depth[np.arange(96)*90//96]*255/20).astype(np.float32)/255.)
    np.testing.assert_array_equal(result,expected)
