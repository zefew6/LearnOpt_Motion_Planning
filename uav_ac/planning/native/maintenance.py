"""Build/status/clean only the planning Cython namespace, never acados outputs."""

import importlib
import importlib.machinery
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

KERNELS = ('_collision_broadphase', '_trajectory_math', '_aerial_constraints')


def project_directory():
    return Path(__file__).resolve().parents[3]


def cache_directory(project_root=None):
    root=project_directory() if project_root is None else Path(project_root).resolve()
    return root/'c_generated_code'/'cython'


def clean_native(project_root=None):
    cache=cache_directory(project_root)
    if cache.is_symlink():
        raise ValueError('native cache must not be a symlink')
    if not cache.exists():
        return False
    shutil.rmtree(cache)
    importlib.invalidate_caches()
    return True


def build_native(project_root=None):
    root=project_directory() if project_root is None else Path(project_root).resolve()
    if not (root/'setup.py').is_file():
        raise RuntimeError('native builds require a project source checkout')
    if cache_directory(root).is_symlink():
        raise ValueError('native cache must not be a symlink')
    subprocess.run([sys.executable, str(root/'setup.py'), 'build_ext', '--inplace'],
                   cwd=root, check=True)
    importlib.invalidate_caches()
    status=native_status(root)
    missing=[name for name, path in status['modules'].items() if path is None]
    if missing:
        raise RuntimeError(f'native build did not produce kernels: {", ".join(missing)}')
    for name, binary in status['modules'].items():
        source=root/'uav_ac'/'planning'/'native'/f'{name}.pyx'
        if source.is_file() and Path(binary).stat().st_mtime_ns < source.stat().st_mtime_ns:
            raise RuntimeError(f'native build left a stale kernel: {name}')
    probe=(f'import sys, importlib; from pathlib import Path; sys.path.insert(0,{str(root)!r}); '
           f'cache=Path({str(cache_directory(root).resolve())!r}); '
           f'modules=[importlib.import_module("uav_ac.planning.native."+name) for name in {KERNELS!r}]; '
           'assert all(Path(module.__file__).resolve().is_relative_to(cache) for module in modules)')
    subprocess.run([sys.executable, '-c', probe], cwd=root, check=True)
    return status


def native_status(project_root=None):
    cache=cache_directory(project_root)
    library=cache/'lib'/'uav_ac'/'planning'/'native'
    modules={}
    for name in KERNELS:
        candidates=[library/(name+suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]
        modules[name]=next((str(path) for path in candidates if path.is_file()), None)
        if project_root is None:
            spec=importlib.util.find_spec(f'uav_ac.planning.native.{name}')
            if spec is not None:
                modules[name]=spec.origin
    groups={group:sum(1 for path in (cache/group).rglob('*') if path.is_file())
            for group in ('c','obj','lib','wheels','archive')}
    return dict(cache_directory=str(cache), modules=modules, artifact_counts=groups)
