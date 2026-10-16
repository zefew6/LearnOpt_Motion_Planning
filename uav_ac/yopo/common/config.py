"""YAML loading that rejects duplicate settings."""
import yaml

class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader,node,deep=False):
    values={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node,deep=deep)
        if key in values: raise ValueError(f'duplicate setting: {key}')
        values[key]=loader.construct_object(value_node,deep=deep)
    return values

_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_mapping)
