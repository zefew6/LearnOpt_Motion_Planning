"""Build the optional native broad-phase kernel during normal package installation."""

import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    [Extension('uav_ac.planning.geometry._collision_broadphase',
               ['uav_ac/planning/geometry/_collision_broadphase.pyx'], optional=True,
               extra_compile_args=[] if sys.platform == 'win32' else ['-O3', '-ffp-contract=off'])],
    build_dir='build/cython',
    compiler_directives={'language_level': 3, 'boundscheck': False, 'wraparound': False}))
