"""Official PNG/CSV dataset, with paths and split configuration supplied explicitly."""
from pathlib import Path
import hashlib

from uav_ac.yopo.vendor.yopo_minco.config.config import Config,configuration
from uav_ac.yopo.vendor.yopo_minco.policy.yopo_dataset import YOPODataset


def load_datasets(path, *, val_ratio=.1):
    path=Path(path).resolve()
    if not path.is_dir():raise ValueError('missing official UAV dataset')
    if not 0<val_ratio<1:raise ValueError('val_ratio must lie in (0,1)')
    config=Config(train=True);config['dataset_path']=str(path)
    with configuration(config):
        train=YOPODataset('train',val_ratio);valid=YOPODataset('valid',val_ratio)
    ids=set(train.map_idx)|set(valid.map_idx)
    if {p.name for p in path.glob('pointcloud-*.ply')}!={f'pointcloud-{i}.ply' for i in ids}:
        raise ValueError('official pointcloud/pose/image map IDs must match')
    return train,valid


def dataset_fingerprint(path, train, valid):
    """Bind resume to source maps/poses and the collected image inventory."""
    path=Path(path).resolve();digest=hashlib.sha256()
    for file in sorted([*path.glob('pose-*.csv'),*path.glob('pointcloud-*.ply')]):
        digest.update(file.name.encode());digest.update(file.read_bytes())
    for file in sorted(map(Path,train.img_list+valid.img_list)):
        stat=file.stat()
        digest.update(f'{file.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()
