from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def test_only_one_yopo_interactive_xml_remains():
    models=Path('uav_ac/simulation/models')
    assert sorted(p.name for p in models.glob('yopo*.xml'))==['yopo_bamboo.xml']
    assert not list(models.glob('whole_body*.xml'))


def test_bamboo_benchmark_has_only_distributed_variable_height_poles():
    root=ET.parse('uav_ac/simulation/models/yopo_bamboo.xml').getroot()
    poles=[g for g in root.findall('worldbody/geom') if g.get('name','').startswith('obstacle_')]
    assert len(poles)>=200
    assert all(g.get('type')=='cylinder' and g.get('name').startswith('obstacle_bamboo_') for g in poles)
    positions=np.array([np.fromstring(g.get('pos'),sep=' ') for g in poles])
    sizes=np.array([np.fromstring(g.get('size'),sep=' ') for g in poles])
    # Every longitudinal band contains obstacles on both sides, rather than one cluster.
    for left,right in zip(np.arange(1.,22.,3.),np.arange(4.,25.,3.)):
        band=positions[(positions[:,0]>=left)&(positions[:,0]<right)]
        assert len(band)>=15 and np.min(band[:,1])<-3. and np.max(band[:,1])>3.
    heights=2*sizes[:,1]
    assert np.count_nonzero(heights<1.8)>=50
    assert np.count_nonzero(heights>2.5)>=50
    assert np.ptp(sizes[:,0])>.03
    assert np.allclose(positions[:,2],sizes[:,1])  # Cylinders rest on the ground.


def test_bamboo_start_is_lowered_and_matches_rendered_pose():
    root=ET.parse('uav_ac/simulation/models/yopo_bamboo.xml').getroot()
    start=np.fromstring(root.find("custom/numeric[@name='whole_body_start']").get('data'),sep=' ')
    assert np.isclose(start[2],-1.2)
    # World poses use ENU; the planning endpoint uses NED.
    position=np.array([start[0],-start[1],-start[2]])
    for element in ('worldbody/body','worldbody/site[@name="start"]'):
        assert np.allclose(np.fromstring(root.find(element).get('pos'),sep=' '),position)
