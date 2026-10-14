"""YOPO ownership and map-free deployment contract."""
import subprocess
import sys
from pathlib import Path


def test_consolidated_package_and_map_free_deployment():
    root=Path('uav_ac')
    assert (root/'yopo/whole_body/model/network.py').is_file()
    assert (root/'yopo/uav/core.py').is_file()
    assert (root/'yopo/common/depth.py').is_file()
    assert (root/'yopo/vendor/yopo_minco/provenance.json').is_file()
    assert not (root/'yopo/model').exists()
    assert not (root/'YOPO').exists()
    assert not (root/'planning/end2end').exists()
    script='''
import sys
from uav_ac.yopo.whole_body.deployment import YopoPlanner
assert not any(name.startswith(('uav_ac.yopo.whole_body.training', 'uav_ac.yopo.whole_body.data', 'uav_ac.yopo.whole_body.validation')) for name in sys.modules)
'''
    subprocess.run([sys.executable,'-c',script],check=True)
