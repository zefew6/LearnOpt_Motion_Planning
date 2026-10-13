import json
import xml.etree.ElementTree as ET
import numpy as np
from uav_ac.YOPO.config import DatasetConfig
from uav_ac.YOPO.generate import generate_dataset
from uav_ac.YOPO.dataset import PlanningDataset


def test_portal_data_covers_each_wall_volume_and_preserves_schema(tmp_path):
    cfg=DatasetConfig(profile='portal_v4',scene_count=9,samples_per_scene=8,split_counts=(3,3,3))
    manifest=generate_dataset(cfg,tmp_path/'portals');data=PlanningDataset(manifest.parent)
    assert len(data)==72 and data.manifest['schema_version']==2
    for split in ('train','validation','test'):
        entries=[s for s in data.manifest['scenes'] if s['split']==split]
        assert {s['family'] for s in entries}=={'portal_rectangle','portal_step','portal_offset'}
        for scene in entries:
            root=ET.parse(manifest.parent/scene['xml']).getroot()
            openings=np.fromstring(root.find("custom/numeric[@name='portal_openings']").get('data'),sep=' ').reshape(-1,9)
            diagnostics=json.loads((manifest.parent/scene['source_diagnostics']).read_text())
            rows=[(o,p) for o,p in (data[i] for i in range(len(data))) if p.scene_id==scene['scene_id']]
            for variant in (0,1):
                observations=[o for (o,_),r in zip(rows,diagnostics['samples']) if r['variant']==variant]
                assert all(any(abs(o.configuration[0]-opening[0])<1e-5 for o in observations) for opening in openings)
    import pytest
    witness=manifest.parent/data.manifest['scenes'][0]['source_numeric']
    witness.write_bytes(witness.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source.*checksum'):PlanningDataset(manifest.parent)
