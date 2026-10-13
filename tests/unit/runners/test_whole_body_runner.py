from pathlib import Path
import pytest
import yaml
from uav_ac.runners.config import load_config


def config_file(tmp_path):
    # Run artifacts are not shipped: construct a valid local checkpoint fixture.
    from uav_ac.planning.end2end.network import WholeBodyPlannerNet
    from uav_ac.planning.end2end.encoding import Normalization
    from uav_ac.planning.end2end.contracts import DepthSpec
    from uav_ac.planning.end2end.checkpoint import save_checkpoint
    n=Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4))
    model=WholeBodyPlannerNet(n,score_mode='ranking',width=4)
    checkpoint=tmp_path/'policy';save_checkpoint(checkpoint,model,{'depth_spec':DepthSpec().as_mapping(),'candidate_count':27,'gripper_gap':.06})
    values=yaml.safe_load(Path('configs/flight/whole_body_online.yaml').read_text())
    values['end2end']['checkpoint']=str(checkpoint)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(values))
    return path,values


def test_whole_body_config_and_main_dispatch(monkeypatch,tmp_path):
    from uav_ac import main
    from uav_ac.runners import whole_body_planning
    path,_=config_file(tmp_path);config=load_config(path)
    assert config['task']=='whole_body_planning' and config['planner']=='end2end'
    assert Path(config['end2end']['checkpoint']).is_dir()
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
    from uav_ac.planning.end2end.contracts import DepthSpec
    path,_=config_file(tmp_path);config=load_config(path)
    calls=[];periods=[]
    planner=SimpleNamespace(model=SimpleNamespace(depth_spec=DepthSpec()),plan=lambda obs:calls.append('plan'))
    monkeypatch.setattr(runner,'End2EndPlanner',lambda *a,**k:planner)
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
