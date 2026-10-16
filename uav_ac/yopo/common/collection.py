"""Shared paths and output preparation for the pinned official map collector."""
from pathlib import Path
import yaml
from uav_ac.yopo.common.config import _UniqueLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = PROJECT_ROOT / 'runs'
SIMULATOR = Path(__file__).resolve().parents[1] / 'vendor/yopo_simulator'


def prepare_collection(config_path, run_dir, *, runs_root=RUNS_ROOT):
    """Write a collection config in a new run, without building or rendering."""
    run = Path(run_dir).resolve()
    if not run.is_relative_to(Path(runs_root).resolve()) or run == Path(runs_root).resolve():
        raise ValueError('collection run must be a new directory beneath runs/')
    with Path(config_path).open(encoding='utf-8') as stream:
        values = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(values, dict):
        raise ValueError('collection configuration must be a mapping')
    for key in ('env_num', 'image_num'):
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    values['save_path'] = str(run / 'dataset') + '/'
    values['tree_file'] = str(SIMULATOR / 'assets/tree.ply')
    run.mkdir(parents=True, exist_ok=False)
    config = run / 'collection.yaml'
    config.write_text(yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
    return config
