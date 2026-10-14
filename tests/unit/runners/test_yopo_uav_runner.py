from pathlib import Path
import json
import pytest
import yaml
from uav_ac.runners.config import load_config


def fixture_config(tmp_path):
    checkpoint=tmp_path/'policy';checkpoint.mkdir()
    (checkpoint/'metadata.json').write_text(json.dumps({'profile':'uav_official_minco_v1','upstream_commit':'d5e1e60f83cc991602f142d9b6b96204a24d6b9f'}))
    (checkpoint/'weights.pth').write_bytes(b'fixture')
    values={'task':'yopo_navigation','scene':'uav_yopo_bamboo','planner':'yopo','controller':'cascaded','wind':'none','control_dt':.02,'yopo_uav':{'checkpoint':str(checkpoint),'device':'cpu'}}
    path=tmp_path/'flight.yaml';path.write_text(yaml.safe_dump(values));return path,values


def test_uav_dispatch_and_isolated_configuration(tmp_path,monkeypatch):
    from uav_ac import main
    from uav_ac.runners import yopo_navigation
    path,_=fixture_config(tmp_path);c=load_config(path)
    assert c['task']=='yopo_navigation' and c['yopo']=={}
    monkeypatch.setattr(yopo_navigation,'run_yopo_navigation',lambda c:{'task':c['task']})
    assert main.run(c)=={'task':'yopo_navigation'}


def test_uav_configuration_rejects_wrong_robot_bad_period_and_inactive_options(tmp_path):
    path,v=fixture_config(tmp_path)
    for key,bad,match in (('scene','yopo_bamboo','quadrotor'),('control_dt',.0015,'control_dt'),('controller','mpc','yopo_navigation')):
        original=v.get(key);v[key]=bad;path.write_text(yaml.safe_dump(v))
        with pytest.raises(ValueError,match=match):load_config(path)
        v[key]=original
    v['yopo_uav']['topk']=True;path.write_text(yaml.safe_dump(v))
    with pytest.raises(ValueError,match='topk'):load_config(path)
    v['task']='trajectory_tracking';v['planner']='mini_snap';v['yopo_uav']={'unknown':1};path.write_text(yaml.safe_dump(v))
    with pytest.raises(ValueError,match='yopo_uav'):load_config(path)


def test_uav_viewer_opens_before_real_planning_with_physics_feedback(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from uav_ac.runners.yopo_navigation import run_yopo_navigation
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    from uav_ac.simulation import depth_observation
    from uav_ac.yopo.uav import deployment
    import uav_ac.control as control
    path,values=fixture_config(tmp_path)
    values['visualize']=True;path.write_text(yaml.safe_dump(values));config=load_config(path)
    calls=[];periods=[]
    planner=SimpleNamespace(depth_spec=None,warm_up=lambda:calls.append('warmup'),reset=lambda:None,
        plan=lambda *a:calls.append('plan'))
    monkeypatch.setattr(deployment,'UavPlanner',lambda *a,**k:planner)
    monkeypatch.setattr(depth_observation,'IdealDepthObserver',lambda *a,**k:SimpleNamespace(close=lambda:None))
    real_controller=control.CascadedController
    def controller(g,dt):periods.append(dt);return real_controller(g,dt)
    monkeypatch.setattr(control,'CascadedController',controller)
    def viewer(self,step,reset,*,chase_camera,prepare):
        calls.append('viewer_open');assert prepare(lambda:None)
        assert calls==['warmup','viewer_open']
        assert periods==[self.quad.dt]
    monkeypatch.setattr(MujocoSimulation,'run_interactive',viewer)
    assert run_yopo_navigation(config)['failure_reason']=='viewer_closed'
