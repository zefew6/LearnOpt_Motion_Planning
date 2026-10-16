"""Ownership migration keeps old modules and command entrypoints usable."""
import importlib
import subprocess
import sys
from pathlib import Path


def test_old_and_new_whole_body_imports_share_module_and_class_identity():
    for suffix in ('model.network','data.dataset','training.runner','trajectory.minco','deployment','contracts'):
        old=importlib.import_module('uav_ac.yopo.'+suffix)
        new=importlib.import_module('uav_ac.yopo.whole_body.'+suffix)
        assert old is new
    for old_suffix,new_suffix in (('training.legacy','training.oracle'),('trajectory.legacy','trajectory.parameterization')):
        assert importlib.import_module('uav_ac.yopo.'+old_suffix) is importlib.import_module('uav_ac.yopo.whole_body.'+new_suffix)
    from uav_ac.yopo.common.depth import DepthSpec
    from uav_ac.yopo.contracts import DepthSpec as OldDepthSpec
    assert DepthSpec is OldDepthSpec


def test_old_and_new_cli_entrypoints_accept_help():
    for suffix in ('training',):
        for prefix in ('uav_ac.yopo.','uav_ac.yopo.whole_body.'):
            result=subprocess.run([sys.executable,'-m',prefix+suffix,'--help'],capture_output=True,text=True)
            assert result.returncode==0,result.stderr
            assert '--config' in result.stdout


def test_uav_observation_import_does_not_load_whole_body():
    script="import sys;import uav_ac.yopo.uav.deployment;assert not any(n.startswith('uav_ac.yopo.whole_body') for n in sys.modules)"
    subprocess.run([sys.executable,'-c',script],check=True)
