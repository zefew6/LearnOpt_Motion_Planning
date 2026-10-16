"""Lazy aliases for pre-split whole-body imports and module entrypoints.

Aliases refer to the canonical module object, so saved imports and monkeypatches
cannot create a second network/config implementation. New code uses whole_body.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_ROOT = 'uav_ac.yopo.'
_OWNERS = {'model', 'data', 'training', 'trajectory', 'validation', 'contracts', 'deployment'}
_RENAMED = {'training.legacy': 'training.oracle',
            'trajectory.legacy': 'trajectory.parameterization'}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target, target_spec):
        self.target = target
        self.target_spec = target_spec

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.target_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery assigns the alias spec to a returned existing module.
        # Keep its canonical spec for reloads, resources and relative imports.
        module.__spec__ = self.target_spec

    def get_code(self, fullname):
        return self.target_spec.loader.get_code(self.target)


class _LegacyWholeBodyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_ROOT):
            return None
        suffix = fullname[len(_ROOT):]
        if suffix.split('.')[0] not in _OWNERS:
            return None
        canonical = _ROOT + 'whole_body.' + _RENAMED.get(suffix, suffix)
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical, spec),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None)


def install():
    if not any(isinstance(finder, _LegacyWholeBodyFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyWholeBodyFinder())
