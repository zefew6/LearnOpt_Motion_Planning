import xml.etree.ElementTree as ET
import numpy as np
import pytest
from uav_ac.yopo.data.config import DatasetConfig
from uav_ac.yopo.contracts import DepthSpec


def test_portal_profile_has_closed_room_and_three_aperture_families(tmp_path):
    from uav_ac.yopo.data.legacy import write_portal_scene
    cfg=DatasetConfig(profile='portal_v4',scene_count=9,samples_per_scene=8,split_counts=(3,3,3))
    assert cfg.profile=='portal_v4'
    for i,family in enumerate(('portal_rectangle','portal_step','portal_offset')):
        _,actual,path=write_portal_scene(tmp_path,i,61+i,DepthSpec())
        assert actual==family
        root=ET.parse(path).getroot()
        assert len(root.findall("worldbody/geom[@name='obstacle_room_side_1']"))==1
        assert root.find("custom/numeric[@name='portal_openings']") is not None
        vegetation=[g for g in root.findall('worldbody/geom') if g.get('name','').startswith(('obstacle_bamboo_','obstacle_tree_'))]
        assert any(g.get('name').startswith('obstacle_bamboo_') for g in vegetation)
        assert any(g.get('name').startswith('obstacle_tree_') for g in vegetation)
        tops=[float(g.get('pos').split()[2])+float(g.get('size').split()[1]) for g in vegetation]
        assert np.ptp(tops)>.1
        route=np.fromstring(root.find("custom/numeric[@name='portal_route_xy_ned']").get('data'),sep=' ').reshape(7,2)
        if i==2:assert np.ptp(route[:,1])>.15
    with pytest.raises(ValueError):DatasetConfig(profile='portal_v4',scene_count=6,split_counts=(2,2,2))


@pytest.mark.parametrize('index',[0,1,2])
def test_portal_reference_clears_wall_volume_while_frozen_arm_collides(tmp_path,index):
    from uav_ac.yopo.data.legacy import write_portal_scene
    from uav_ac.yopo.data.legacy import build_validated_sources
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    _,_,path=write_portal_scene(tmp_path,index,61+index*1009,DepthSpec())
    sim=MujocoSimulation(path,planning_path_capacity=0,record_actual_trajectory=False)
    sources=build_validated_sources(sim,.06,seed=62+index*1009)
    assert all(d['validation_passed'] and d['base_collision_frames']==0 and d['frozen_arm_collision_frames']>0 for _,d in sources)
    assert min(d['minimum_world_distance'] for _,d in sources)>=.05
    if index==2:assert np.ptp(sources[0][0].rrt_path[:,1])>.15


def test_tree_between_offset_walls_also_has_a_frozen_arm_collision_witness(tmp_path):
    from uav_ac.yopo.data.legacy import write_portal_scene
    from uav_ac.yopo.data.legacy import build_validated_sources
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    _,_,path=write_portal_scene(tmp_path,2,2079,DepthSpec())
    sim=MujocoSimulation(path,planning_path_capacity=0,record_actual_trajectory=False)
    sources=build_validated_sources(sim,.06,seed=2080)
    assert any(name.startswith('obstacle_tree_middle_') for _,d in sources for pair in d['frozen_collision_pairs'] for name in pair)
