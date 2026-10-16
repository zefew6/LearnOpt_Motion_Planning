"""Strict official UAV weight loading; incompatible whole-body weights are rejected."""
from pathlib import Path
import hashlib,json
import torch
from .core import OfficialCore,PROFILE,UPSTREAM_COMMIT


def save_checkpoint(path,network,metadata):
    from uav_ac.yopo.vendor.yopo_minco.config.config import Config
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    weights=path/'weights.pth';torch.save(network.state_dict(),weights)
    info={**metadata,'profile':PROFILE,'upstream_commit':UPSTREAM_COMMIT,
          'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
          'official_configuration':Config().snapshot()}
    (path/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')


def load_checkpoint(path,*,device='cpu',velocity=6.):
    path=Path(path)
    try:metadata=json.loads((path/'metadata.json').read_text())
    except (OSError,ValueError) as error:raise ValueError('missing/invalid official UAV metadata') from error
    if metadata.get('profile')!=PROFILE or metadata.get('upstream_commit')!=UPSTREAM_COMMIT:
        raise ValueError('incompatible official UAV checkpoint profile/version')
    weights_path=path/'weights.pth'
    try:
        digest=hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest!=metadata.get('weights_sha256'):raise ValueError('official UAV weights checksum mismatch')
        core=OfficialCore(device=device,velocity=velocity)
        if metadata.get('official_configuration')!=core.config.snapshot()|{'velocity':6.0}:
            # Metadata records unscaled upstream inference; deployment speed is explicit.
            raise ValueError('official UAV checkpoint configuration mismatch')
        state=torch.load(weights_path,map_location='cpu',weights_only=True)
        core.network.load_state_dict(state,strict=True)
    except (OSError,RuntimeError,KeyError,TypeError) as error:raise ValueError('invalid official UAV weights') from error
    return core,metadata
