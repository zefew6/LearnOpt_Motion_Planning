"""Optional planning kernels; local build artifacts live outside the source tree."""

from pathlib import Path

# Installed wheels contain extensions beside this file. Source checkouts load
# only this package's extensions from the isolated Cython build namespace.
_local_extensions = (Path(__file__).resolve().parents[3] / 'c_generated_code'
                     / 'cython' / 'lib' / 'uav_ac' / 'planning' / 'native')
if str(_local_extensions) not in __path__:
    __path__.append(str(_local_extensions))
