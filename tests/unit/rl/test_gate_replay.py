from contextlib import nullcontext
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch

from uav_ac.rl.tasks.gate_racing import evaluation
from uav_ac.rl.tasks.gate_racing.config import settings_from
from uav_ac.rl.tasks.gate_racing.environment import make_environment
from uav_ac.simulation.recording import default_camera, update_chase_camera


@pytest.mark.parametrize("yaw", [0., np.pi/2, -np.pi/2])
def test_chase_camera_is_physically_behind_and_above(yaw):
    env = make_environment(settings_from({}), perturb=False)
    try:
        env.reset()
        sim = env.unwrapped.simulation
        state = sim.quad.X.copy()
        state[3:7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
        sim.reset(state)
        camera = default_camera(sim.model)
        update_chase_camera(camera,sim)
        scene = mujoco.MjvScene(sim.model,maxgeom=2000)
        mujoco.mjv_updateScene(sim.model,sim.data,mujoco.MjvOption(),None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        eye = (scene.camera[0].pos+scene.camera[1].pos)/2
        origin = sim.data.xpos[sim._body_id]
        forward = sim.data.xmat[sim._body_id].reshape(3,3)[:,0]
        assert np.dot(eye-origin,forward) < -4
        assert eye[2]-origin[2] > 3
        np.testing.assert_allclose(camera.lookat,origin+[0,0,.5])
        assert camera.elevation == -20 and camera.distance == 8
    finally:
        env.close()


@pytest.mark.parametrize("record", [False,True])
@pytest.mark.parametrize("follow", [False,True])
def test_interactive_and_record_share_camera_updates(monkeypatch,tmp_path,record,follow):
    settings = settings_from({"episode_seconds":.02})
    class Hover:
        device = torch.device("cpu")
        policy = object()
        def predict(self,obs,deterministic=True):
            return np.zeros((len(obs),4)),None
    monkeypatch.setattr(evaluation,"load_model",lambda *args,**kwargs:(Hover(),settings))
    import uav_ac.simulation.recording as recording
    from mujoco import viewer
    updates = Mock(wraps=recording.update_chase_camera)
    monkeypatch.setattr(recording,"update_chase_camera",updates)
    fake_viewer = Mock(cam=mujoco.MjvCamera())
    fake_viewer.lock.side_effect = lambda:nullcontext()
    # Simulate a manual close after the terminal frame; the simulator should
    # continue stepping between the terminal event and this close.
    # One reset frame, two control steps, one post-terminal control step, then
    # the false result returned after the manual close.
    fake_viewer.is_running.side_effect = [True, True, True, True, False]
    monkeypatch.setattr(viewer,"launch_passive",lambda *args:fake_viewer)
    fake_renderer = Mock()
    fake_renderer.render.return_value = np.zeros((48,64,3),dtype=np.uint8)
    monkeypatch.setattr(mujoco,"Renderer",lambda *args,**kwargs:fake_renderer)
    recorder = Mock()
    recorder.__enter__ = Mock(return_value=recorder)
    monkeypatch.setattr(recording,"Mp4Recorder",lambda *args:recorder)
    monkeypatch.setattr(evaluation.time,"sleep",lambda *args:None)
    result = evaluation.replay(tmp_path,output=tmp_path/"clip.mp4" if record else None,
                               follow_camera=follow,width=64,height=48)
    assert result["timeout_rate"] == 1
    assert (updates.call_count >= 2) if follow else updates.call_count == 0
    if record:
        assert fake_renderer.update_scene.call_count >= 2
        recorder.close.assert_called_once()
    else:
        fake_viewer.close.assert_called_once()


def test_requested_checkpoint_is_loaded(monkeypatch,tmp_path):
    settings = settings_from({})
    monkeypatch.setattr(evaluation,"read_run",lambda directory:({},settings))
    load = Mock(return_value=object())
    monkeypatch.setattr(evaluation.PPO,"load",load)
    selected = tmp_path/"final_model.zip"
    evaluation.load_model(tmp_path,checkpoint=selected)
    assert load.call_args.args[0] == selected
