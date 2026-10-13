"""Standalone weights/metadata IO; does not import training or environment maps."""
from pathlib import Path
import json
from dataclasses import asdict
from .primitive import PrimitiveSpec,PRIMITIVE_STATE_LAYOUT
from .primitive_network import PrimitivePlannerNet
from .goal_network import GoalPrimitivePlannerNet
import torch
from .network import WholeBodyPlannerNet
from .encoding import Normalization,STATE_LAYOUT
from .contracts import DepthSpec


def save_checkpoint(path,model,metadata):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    norm=model.normalization
    primitive=isinstance(model,PrimitivePlannerNet)
    info={**metadata,'schema_version':3 if isinstance(model,GoalPrimitivePlannerNet) else 2 if primitive else 1,'profile':model.profile if primitive else 'direct_v3','parameter_dim':34,'state_layout':list(PRIMITIVE_STATE_LAYOUT if primitive else STATE_LAYOUT),
          'score_mode':model.score_mode,'width':model.width,'normalization':{
              'joint_lower':norm.joint_lower.tolist(),'joint_upper':norm.joint_upper.tolist(),
              'position_scale':norm.position_scale,'yaw_scale':norm.yaw_scale,'joint_scales':list(norm.joint_scales)}}
    if primitive:
        if info.get('candidate_count')!=27:raise ValueError('primitive_v4 requires 27 candidates')
        info['primitive_spec']=asdict(model.spec)
    (path/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    torch.save(model.state_dict(),path/'weights.pt')


def load_checkpoint(path,*,device='cpu',expected_robot_hash=None,expected_depth_spec=None):
    path=Path(path)
    try:info=json.loads((path/'metadata.json').read_text())
    except (OSError,ValueError) as error:raise ValueError('missing/invalid checkpoint metadata') from error
    primitive=info.get('profile') in ('primitive_v4','primitive_goal_v5')
    goal_terminal=info.get('profile')=='primitive_goal_v5'
    if info.get('profile') not in ('direct_v3','primitive_v4','primitive_goal_v5') or info.get('schema_version')!=(3 if goal_terminal else 2 if primitive else 1) or info.get('parameter_dim')!=34 or info.get('state_layout')!=list(PRIMITIVE_STATE_LAYOUT if primitive else STATE_LAYOUT):
        raise ValueError('incompatible checkpoint contract')
    if expected_robot_hash is not None and info.get('robot_hash')!=expected_robot_hash:raise ValueError('robot model mismatch')
    if expected_depth_spec is not None and info.get('depth_spec')!=expected_depth_spec.as_mapping():raise ValueError('depth calibration mismatch')
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise ValueError('requested CUDA is unavailable')
    try:
        normalization=Normalization(**info['normalization'])
        if primitive and info.get('candidate_count')!=27:raise ValueError('incompatible primitive contract')
        cls=GoalPrimitivePlannerNet if goal_terminal else PrimitivePlannerNet if primitive else WholeBodyPlannerNet
        kwargs={'spec':PrimitiveSpec(**info['primitive_spec'])} if primitive else {}
        model=cls(normalization,score_mode=info['score_mode'],width=info['width'],depth_spec=DepthSpec(**info['depth_spec']),**kwargs)
        weights=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
        buffers=('lower','upper','camera_rotation','camera_offset','fov_tangent','bin_angles','primitive_rotations','sampling_xy','spec_values') if primitive else ('scales','lower','upper','camera_rotation','camera_offset','fov_tangent')
        for name in buffers:
            if name not in weights or not torch.equal(weights[name],model.state_dict()[name]):
                raise ValueError('checkpoint geometry buffers contradict metadata')
        model.load_state_dict(weights)
    except (KeyError,TypeError,RuntimeError,OSError) as error:raise ValueError('invalid checkpoint weights/configuration') from error
    return model.to(device).eval(),info
