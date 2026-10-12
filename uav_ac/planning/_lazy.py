"""Preserve package facades without loading unrelated solver dependencies."""
from importlib import import_module


def install_exports(namespace,exports):
    namespace['__all__']=list(exports)
    def get(name):
        if name not in exports: raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
        module,attribute=exports[name]
        value=getattr(import_module(module),attribute);namespace[name]=value
        return value
    namespace['__getattr__']=get
    namespace['__dir__']=lambda:sorted(set(namespace)|set(exports))
