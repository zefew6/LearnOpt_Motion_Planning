import xml.etree.ElementTree as ET
import numpy as np
import pytest
from uav_ac.yopo.data.config import DatasetConfig
from uav_ac.yopo.contracts import DepthSpec


def test_coupled_profile_and_scene_geometry(tmp_path):
    from uav_ac.yopo.data.legacy import write_coupled_scene
    config=DatasetConfig(profile='coupled_v3',scene_count=6,samples_per_scene=6,split_counts=(2,2,2))
    assert config.profile=='coupled_v3'
    for i,family in enumerate(('coupled_corridor','coupled_bamboo')):
        sid,actual,path=write_coupled_scene(tmp_path,i,19+i,DepthSpec())
        assert actual==family and sid==f'scene_{i:03d}'
        root=ET.parse(path).getroot();geoms=root.findall('worldbody/geom')
        low=[g for g in geoms if g.get('name','').startswith('obstacle_low_')]
        assert len(low)>=2
        tops=[np.fromstring(g.get('pos'),sep=' ')[2]+np.fromstring(g.get('size'),sep=' ')[-1] for g in low]
        assert np.ptp(tops)>.03
        assert root.find("custom/numeric[@name='planning_bounds']") is not None
        assert root.find("worldbody/body/camera[@name='yopo_depth']") is not None
    with pytest.raises(ValueError):DatasetConfig(profile='not_a_profile')


def test_coupled_reference_is_safe_but_frozen_arm_hits_lower_obstacles(tmp_path):
    from uav_ac.yopo.data.legacy import write_coupled_scene
    from uav_ac.yopo.data.legacy import build_validated_sources
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    _,_,path=write_coupled_scene(tmp_path,0,19,DepthSpec())
    sim=MujocoSimulation(path,planning_path_capacity=0,record_actual_trajectory=False)
    sources=build_validated_sources(sim,.06,seed=20)
    assert len(sources)==2
    for trajectory,diagnostic in sources:
        assert diagnostic['validation_passed']
        assert diagnostic['frozen_arm_collision_frames']>0
        assert diagnostic['base_collision_frames']==0
        assert diagnostic['maximum_joint_change_rad']>1.
        assert trajectory.validation_performed


@pytest.mark.parametrize('index',[0,1])
def test_both_lower_and_front_obstacles_constrain_arm_shape(tmp_path,index):
    from uav_ac.yopo.data.legacy import write_coupled_scene
    from uav_ac.yopo.data.legacy import build_validated_sources
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    _,_,path=write_coupled_scene(tmp_path,index,31+index*1009,DepthSpec())
    sim=MujocoSimulation(path,planning_path_capacity=0,record_actual_trajectory=False)
    sources=build_validated_sources(sim,.06,seed=32+index*1009)
    pairs=[name for _,d in sources for pair in d['frozen_collision_pairs'] for name in pair]
    assert any(name.startswith('obstacle_low_') for name in pairs)
    assert any(name.startswith(('obstacle_front_','obstacle_trunk_')) for name in pairs)
