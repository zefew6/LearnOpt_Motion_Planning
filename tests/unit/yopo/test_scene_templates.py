from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from uav_ac.yopo.contracts import DepthSpec


def test_legacy_generation_only_requires_current_benchmark(monkeypatch,tmp_path):
    from uav_ac.yopo.data import scenes as scene_sampling
    source=tmp_path/'simulation';(source/'models').mkdir(parents=True);(source/'model').mkdir()
    shutil.copyfile('uav_ac/simulation/models/whole_body_dense_outdoor.xml',source/'models/whole_body_dense_outdoor.xml')
    shutil.copyfile('uav_ac/simulation/model/aerial_manipulator.xml',source/'model/aerial_manipulator.xml')
    monkeypatch.setattr(scene_sampling,'MODEL_ROOT',source)
    for index,count in enumerate((0,1,2)):
        _,family,path=scene_sampling.write_scene(tmp_path/'generated',index,17,DepthSpec())
        root=ET.parse(path).getroot()
        assert len([g for g in root.findall('worldbody/geom') if g.get('name','').startswith('obstacle_')])==count
        assert not any(n.get('name','').startswith('whole_body_') for n in root.findall('custom/numeric'))
        bounds=np.fromstring(root.find("custom/numeric[@name='planning_bounds']").get('data'),sep=' ')
        assert np.allclose(bounds,[-1.2,-1.6,-2.4,2.8,1.6,-.75])
        assert root.find('worldbody/body').get('pos')=='0 0 1.5'


def test_only_one_yopo_interactive_xml_remains():
    models=Path('uav_ac/simulation/models')
    assert not list(models.glob('yopo*.xml'))
    assert sorted(p.name for p in models.glob('whole_body*.xml'))==['whole_body_dense_outdoor.xml']

