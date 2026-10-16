import struct

import numpy as np
import pytest


def test_official_ply_voxels_convert_nwu_to_ned_and_preserve_gaps(tmp_path):
    from uav_ac.yopo.whole_body.data.maps import read_points, voxel_boxes
    path=tmp_path/'map.ply'
    points=[(1.1,2.1,3.1),(1.1,2.1,3.3),(1.1,2.1,3.9),(0.,0.,0.)]
    header='ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    path.write_bytes(header.encode()+b''.join(struct.pack('<fff',*p) for p in points))
    loaded=read_points(path)
    np.testing.assert_allclose(loaded,points,atol=1e-6)
    boxes=voxel_boxes(loaded,.2)
    assert boxes.shape==(2,6)  # ground is represented separately as a plane
    boxes=boxes[np.argsort(boxes[:,2])]
    np.testing.assert_allclose(boxes[0],[1.1,-2.1,-3.9,.1,.1,.1],atol=1e-6)
    np.testing.assert_allclose(boxes[1],[1.1,-2.1,-3.2,.1,.1,.2],atol=1e-6)


def test_whole_body_collection_config_rejects_bad_sampling_before_outputs(tmp_path):
    from uav_ac.yopo.whole_body.data.config import load_config
    for text in ('unknown: 1\n','seed: 1\nseed: 2\n','samples_per_scene: false\n',
                 'bounds: [[0,0,0],[0,1,1]]\n','split_counts: [1,0,1]\n',
                 'gripper_gap: 1\n','map_options: {typo: 3}\n',
                 'map_options: {maze_type: 6, max_windows: 0}\n'):
        path=tmp_path/'bad.yaml';path.write_text(text)
        with pytest.raises(ValueError):load_config(path)


def test_whole_body_prepare_only_does_not_build_or_collect(tmp_path,monkeypatch):
    import yaml
    from uav_ac.yopo.whole_body.data import generate
    from uav_ac.yopo.whole_body.data.config import CollectionConfig
    monkeypatch.setattr(generate,'RUNS_ROOT',tmp_path/'runs')
    run=tmp_path/'runs/prepared'
    path=generate.generate_dataset(CollectionConfig(),run,map_binary=tmp_path/'missing',prepare_only=True)
    assert path.is_file()
    values=yaml.safe_load(path.read_text())
    assert values['env_num']==3 and values['image_num']==8
    assert sorted(p.name for p in run.iterdir())==['map_collection.yaml','sampling.yaml']
    assert not (run/'dataset').exists()
    with pytest.raises(FileExistsError):
        generate.generate_dataset(CollectionConfig(),run,prepare_only=True)
    with pytest.raises(ValueError,match='runs'):
        generate.generate_dataset(CollectionConfig(),tmp_path/'source',prepare_only=True)
