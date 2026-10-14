"""YOPO ownership and map-free deployment contract."""
import subprocess
import sys
from pathlib import Path


def test_consolidated_package_and_map_free_deployment():
    root=Path('uav_ac')
    assert (root/'yopo/model/network.py').is_file()
    assert not (root/'YOPO').exists()
    assert not (root/'planning/end2end').exists()
    script='''
import sys
from uav_ac.yopo.deployment import YopoPlanner
assert not any(name.startswith(('uav_ac.yopo.training', 'uav_ac.yopo.data', 'uav_ac.yopo.validation')) for name in sys.modules)
'''
    subprocess.run([sys.executable,'-c',script],check=True)
