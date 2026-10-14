from pathlib import Path
import json
import numpy as np
import pytest
from uav_ac.yopo.whole_body.data.config import load_config
from uav_ac.yopo.whole_body.data.generate import generate_dataset
from uav_ac.yopo.whole_body.data.dataset import PlanningDataset


def tiny_config(tmp_path):
    path=tmp_path/'data.yaml'
    path.write_text('seed: 17\nscene_count: 3\nsamples_per_scene: 2\nsplit_counts: [1, 1, 1]\n')
    return load_config(path)


def test_config_rejects_unknown_duplicate_and_bad_split(tmp_path):
    for text in ['seed: 1\nseed: 2\n','unknown: 1\n','scene_count: 3\nsplit_counts: [1,1,2]\n',
                 'samples_per_scene: false\n','depth: {far_m: 0.01}\n']:
        p=tmp_path/'bad.yaml';p.write_text(text)
        with pytest.raises(ValueError):load_config(p)


def test_small_dataset_rebuild_and_privileged_isolation(tmp_path):
    config=tiny_config(tmp_path);output=tmp_path/'run'
    manifest=generate_dataset(config,output)
    assert Path(manifest).is_file()
    train=PlanningDataset(output/'dataset',split='train')
    assert len(train)==2
    obs,privileged=train[0]
    assert not hasattr(obs,'scene_id') and not hasattr(obs,'esdf')
    assert privileged.scene_id in train.scene_ids
    assert obs.depth_m.shape==(96,160)
    val=PlanningDataset(output/'dataset',split='validation')
    assert not set(train.scene_ids)&set(val.scene_ids)
    other=tmp_path/'other';generate_dataset(config,other)
    assert np.array_equal(train[0][0].depth_m,PlanningDataset(other/'dataset',split='train')[0][0].depth_m)
    with pytest.raises(FileExistsError):generate_dataset(config,output)
    shard=next((output/'dataset'/'shards').glob('*.npz'));shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError):PlanningDataset(output/'dataset')


def test_manifest_cannot_alias_train_scene_into_test(tmp_path):
    import copy
    config=tiny_config(tmp_path);root=tmp_path/'run';generate_dataset(config,root)
    path=root/'dataset'/'manifest.json';manifest=json.loads(path.read_text())
    alias=copy.deepcopy(manifest['scenes'][0]);alias['scene_id']='aliased_train';alias['split']='test'
    manifest['scenes'].append(alias);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='scene|alias'):PlanningDataset(root/'dataset',split='test')
