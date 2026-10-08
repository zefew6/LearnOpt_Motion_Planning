"""Build optional planning kernels in their isolated generated-code namespace."""

import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

NATIVE_BUILD = Path('c_generated_code') / 'cython'
KERNELS = ('_collision_broadphase', '_trajectory_math', '_aerial_constraints')


class PlanningBuild(build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(NATIVE_BUILD)

    def finalize_options(self):
        super().finalize_options()
        self.build_lib = str(NATIVE_BUILD / 'lib')
        self.build_temp = str(NATIVE_BUILD / 'obj' /
                              f'{sysconfig.get_platform()}-{sys.implementation.cache_tag}')


class PlanningBuildExt(build_ext):
    def copy_extensions_to_source(self):
        # The source package extends its search path to the classified lib
        # directory. Never scatter binaries into source or acados's root.
        pass


extensions = [Extension(
    f'uav_ac.planning.native.{name}', [f'uav_ac/planning/native/{name}.pyx'],
    optional=True,
    extra_compile_args=[] if sys.platform == 'win32' else ['-O3', '-ffp-contract=off'])
    for name in KERNELS]

setup(cmdclass={'build': PlanningBuild, 'build_ext': PlanningBuildExt},
      ext_modules=cythonize(
          extensions, build_dir=str(NATIVE_BUILD / 'c'),
          compiler_directives={'language_level': 3, 'boundscheck': False,
                               'wraparound': False}))
