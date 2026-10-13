"""Standalone weights/metadata IO; does not import training or environment maps."""
from pathlib import Path
import json
import torch
from .network import WholeBodyPlannerNet
from .encoding import Normalization,STATE_LAYOUT
from .contracts import DepthSpec


def save_checkpoint(path,model,metadata):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    norm=model.normalization
    info={**metadata,'schema_version':1,'profile':'direct_v3','parameter_dim':34,'state_layout':list(STATE_LAYOUT),
          'score_mode':model.score_mode,'width':model.width,'normalization':{
              'joint_lower':norm.joint_lower.tolist(),'joint_upper':norm.joint_upper.tolist(),
              'position_scale':norm.position_scale,'yaw_scale':norm.yaw_scale,'joint_scales':list(norm.joint_scales)}}
    (path/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    torch.save(model.state_dict(),path/'weights.pt')


def load_checkpoint(path,*,device='cpu',expected_robot_hash=None,expected_depth_spec=None):
    path=Path(path)
    try:info=json.loads((path/'metadata.json').read_text())
    except (OSError,ValueError) as error:raise ValueError('missing/invalid checkpoint metadata') from error
    if info.get('schema_version')!=1 or info.get('profile')!='direct_v3' or info.get('parameter_dim')!=34 or info.get('state_layout')!=list(STATE_LAYOUT):
        raise ValueError('incompatible checkpoint contract')
    if expected_robot_hash is not None and info.get('robot_hash')!=expected_robot_hash:raise ValueError('robot model mismatch')
    if expected_depth_spec is not None and info.get('depth_spec')!=expected_depth_spec.as_mapping():raise ValueError('depth calibration mismatch')
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise ValueError('requested CUDA is unavailable')
    try:
        normalization=Normalization(**info['normalization'])
        model=WholeBodyPlannerNet(normalization,score_mode=info['score_mode'],width=info['width'],depth_spec=DepthSpec(**info['depth_spec']))
        weights=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
        for name in ('scales','lower','upper','camera_rotation','camera_offset','fov_tangent'):
            if name not in weights or not torch.equal(weights[name],model.state_dict()[name]):
                raise ValueError('checkpoint geometry buffers contradict metadata')
        model.load_state_dict(weights)
    except (KeyError,RuntimeError,OSError) as error:raise ValueError('invalid checkpoint weights/configuration') from error
    return model.to(device).eval(),info
