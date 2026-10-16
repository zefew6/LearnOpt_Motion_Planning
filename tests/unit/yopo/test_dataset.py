from pathlib import Path
import json
import numpy as np
import pytest
from tests.yopo_dataset import write_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset


def test_small_dataset_rebuild_and_privileged_isolation(tmp_path):
    output=tmp_path/'run'
    manifest=write_dataset(output, samples_per_scene=2)
    assert Path(manifest).is_file()
    train=PlanningDataset(output/'dataset',split='train')
    assert len(train)==2
    obs,privileged=train[0]
    assert not hasattr(obs,'scene_id') and not hasattr(obs,'esdf')
    assert privileged.scene_id in train.scene_ids
    assert obs.depth_m.shape==(96,160)
    val=PlanningDataset(output/'dataset',split='validation')
    assert not set(train.scene_ids)&set(val.scene_ids)
    other=tmp_path/'other';write_dataset(other, samples_per_scene=2)
    assert np.array_equal(train[0][0].depth_m,PlanningDataset(other/'dataset',split='train')[0][0].depth_m)
    with pytest.raises(FileExistsError):write_dataset(output, samples_per_scene=2)
    shard=next((output/'dataset'/'shards').glob('*.npz'));shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError):PlanningDataset(output/'dataset')


def test_manifest_cannot_alias_train_scene_into_test(tmp_path):
    import copy
    root=tmp_path/'run';write_dataset(root)
    path=root/'dataset'/'manifest.json';manifest=json.loads(path.read_text())
    alias=copy.deepcopy(manifest['scenes'][0]);alias['scene_id']='aliased_train';alias['split']='test'
    manifest['scenes'].append(alias);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='scene|alias'):PlanningDataset(root/'dataset',split='test')


def test_reader_bounds_shard_cache(tmp_path):
    write_dataset(tmp_path/'data', samples_per_scene=3)
    data=PlanningDataset(tmp_path/'data/dataset',max_cached_shards=1)
    for i in (0,1,3,6):
        data[i]
        assert len(data._shards)==1
