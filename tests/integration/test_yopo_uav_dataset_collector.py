"""Native collector checks run when an explicitly built binary is provided."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.fixture
def collector():
    value = os.environ.get('YOPO_DATASET_GENERATOR')
    if value is None:
        pytest.skip('set YOPO_DATASET_GENERATOR to a built official collector')
    binary = Path(value).resolve()
    assert binary.is_file()
    return binary


def test_native_collector_requires_explicit_config(collector):
    result = subprocess.run([str(collector), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'collection.yaml' in result.stdout
    result = subprocess.run([str(collector)], capture_output=True, text=True)
    assert result.returncode != 0 and 'Usage:' in result.stderr


def test_native_collector_preserves_existing_dataset(collector, tmp_path):
    values = yaml.safe_load(Path('configs/yopo/dataset/yopo_uav.yaml').read_text())
    destination = tmp_path / 'dataset'
    destination.mkdir()
    sentinel = destination / 'keep.txt'
    sentinel.write_text('existing user data')
    values.update(env_num=1, image_num=2, save_path=str(destination) + '/')
    config = tmp_path / 'collection.yaml'
    config.write_text(yaml.safe_dump(values))
    result = subprocess.run([str(collector), str(config)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Refusing to overwrite' in result.stderr
    assert sentinel.read_text() == 'existing user data'
    assert sorted(p.name for p in destination.iterdir()) == ['keep.txt']


def test_missing_cuda_fails_before_creating_dataset(collector, tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip('this check requires an unavailable CUDA runtime')
    values = yaml.safe_load(Path('configs/yopo/dataset/yopo_uav.yaml').read_text())
    destination = tmp_path / 'dataset'
    values.update(env_num=1, image_num=2, save_path=str(destination) + '/')
    config = tmp_path / 'collection.yaml'
    config.write_text(yaml.safe_dump(values))
    result = subprocess.run([str(collector), str(config)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'CUDA' in result.stderr
    assert not destination.exists()
