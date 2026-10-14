from pathlib import Path
import pytest
import yaml
from uav_ac.runners.config import load_config


def config_file(tmp_path):
    # Run artifacts are not shipped: construct a valid local checkpoint fixture.
    from uav_ac.yopo.whole_body.model.network import WholeBodyPlannerNet
    from uav_ac.yopo.whole_body.model.encoding import Normalization
    from uav_ac.yopo.whole_body.contracts import DepthSpec
    from uav_ac.yopo.whole_body.model.checkpoint import save_checkpoint
    n=Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4))
    model=WholeBodyPlannerNet(n,score_mode='ranking',width=4)
    checkpoint=tmp_path/'policy';save_checkpoint(checkpoint,model,{'depth_spec':DepthSpec().as_mapping(),'candidate_count':27,'gripper_gap':.06})
    values=yaml.safe_load(Path('configs/flight/yopo.yaml').read_text())
    values['yopo']['checkpoint']=str(checkpoint)
    values['yopo']['device']='cpu'  # Fixture tests do not require CUDA.
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(values))
    return path,values


def test_whole_body_config_and_main_dispatch(monkeypatch,tmp_path):
    from uav_ac import main
    from uav_ac.runners import whole_body_planning
    path,_=config_file(tmp_path);config=load_config(path)
    assert config['task']=='whole_body_planning' and config['planner']=='yopo'
    assert Path(config['yopo']['checkpoint']).is_dir()
    seen=[];monkeypatch.setattr(whole_body_planning,'run_whole_body_planning',lambda c:seen.append(c) or {'state':'test'})
    assert main.run(config)=={'state':'test'} and seen==[config]


def test_invalid_online_options_fail_before_viewer(tmp_path):
    path,values=config_file(tmp_path)
    values['whole_body']={'unknown':1};path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='whole_body'):load_config(path)
    values['whole_body']={'replan_dt':.015};path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='replan_dt'):load_config(path)
    values['whole_body']={'threads':True};path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='threads'):load_config(path)
    values['whole_body']={};values['controller']='mpc';path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='whole_body_planning'):load_config(path)


def test_viewer_precedes_planning_and_feedback_uses_physics_period(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from uav_ac.runners import whole_body_planning as runner
    from uav_ac.yopo.whole_body.contracts import DepthSpec
    path,_=config_file(tmp_path);config=load_config(path)
    calls=[];periods=[]
    planner=SimpleNamespace(model=SimpleNamespace(depth_spec=DepthSpec()),plan=lambda obs:calls.append('plan'))
    monkeypatch.setattr(runner,'YopoPlanner',lambda *a,**k:planner)
    monkeypatch.setattr(runner,'IdealDepthObserver',lambda *a,**k:SimpleNamespace(close=lambda:None))
    real_controller=runner.CascadedController
    def controller(g,dt):periods.append(dt);return real_controller(g,dt)
    monkeypatch.setattr(runner,'CascadedController',controller)
    def viewer(self,step,reset,*,chase_camera,prepare):
        calls.append('viewer_open')
        assert prepare(lambda:None)
        assert calls==['viewer_open']  # initialization does not run the planner
        assert periods==[self.quad.dt]
    monkeypatch.setattr(runner.MujocoSimulation,'run_interactive',viewer)
    result=runner.run_whole_body_planning(config)
    assert result['failure_reason']=='viewer_closed'


def test_retired_end2end_selector_is_rejected(tmp_path):
    path,values=config_file(tmp_path)
    values['planner']='end2end';path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='planner'):load_config(path)


def test_invalid_yopo_speed_scale_is_rejected(tmp_path):
    path,values=config_file(tmp_path)
    values['yopo']['speed_scale']=True;path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='speed_scale'):load_config(path)


def test_optional_height_ceiling_contract(tmp_path):
    from uav_ac.runners.whole_body_config import WholeBodySettings
    assert WholeBodySettings().enforce_height_ceiling is True
    path,values=config_file(tmp_path)
    values['whole_body']['enforce_height_ceiling']=False;path.write_text(yaml.safe_dump(values))
    assert load_config(path)['whole_body']['enforce_height_ceiling'] is False
    for invalid in (0,1,'false',None):
        values['whole_body']['enforce_height_ceiling']=invalid;path.write_text(yaml.safe_dump(values))
        with pytest.raises(ValueError,match='enforce_height_ceiling'):load_config(path)
