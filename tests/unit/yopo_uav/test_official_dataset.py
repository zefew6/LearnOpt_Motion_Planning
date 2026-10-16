"""Official collection setup must never write into source or overwrite a run."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_prepare_official_collection_without_rendering(tmp_path, monkeypatch):
    from uav_ac.yopo.uav import dataset
    monkeypatch.setattr(dataset, 'RUNS_ROOT', tmp_path / 'runs')
    run = tmp_path / 'runs' / 'uav' / 'tiny'
    config = dataset.prepare(ROOT / 'configs/yopo/dataset/yopo_uav.yaml', run)
    values = yaml.safe_load(config.read_text())
    assert values['env_num'] == 10 and values['image_num'] == 10000
    assert Path(values['tree_file']).is_file()
    assert Path(values['save_path']) == run / 'dataset'
    assert not (run / 'dataset').exists()
    assert sorted(p.name for p in run.iterdir()) == ['collection.yaml']
    with pytest.raises(FileExistsError):
        dataset.prepare(ROOT / 'configs/yopo/dataset/yopo_uav.yaml', run)
    assert config.is_file()


def test_collection_rejects_source_output_and_bad_counts_before_writing(tmp_path, monkeypatch):
    from uav_ac.yopo.uav import dataset
    monkeypatch.setattr(dataset, 'RUNS_ROOT', tmp_path / 'runs')
    config = ROOT / 'configs/yopo/dataset/yopo_uav.yaml'
    with pytest.raises(ValueError, match='runs'):
        dataset.prepare(config, tmp_path / 'source')
    for text in ('env_num: 0\nimage_num: 1\n',
                 'env_num: true\nimage_num: 1\n',
                 'env_num: 1\nenv_num: 2\nimage_num: 1\n'):
        invalid = tmp_path / 'bad.yaml'
        invalid.write_text(text)
        with pytest.raises(ValueError):
            dataset.prepare(invalid, tmp_path / 'runs' / 'bad')
    assert not (tmp_path / 'runs' / 'bad').exists()


def test_official_collection_help_needs_no_native_dependencies():
    result = subprocess.run([sys.executable, '-m', 'uav_ac.yopo.uav.dataset', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--prepare-only' in result.stdout and '--run-dir' in result.stdout


def test_retired_whole_body_sampler_profiles_are_not_importable():
    for name in ('fixed', 'sampling', 'scenes'):
        assert importlib.util.find_spec('uav_ac.yopo.whole_body.data.' + name) is None
