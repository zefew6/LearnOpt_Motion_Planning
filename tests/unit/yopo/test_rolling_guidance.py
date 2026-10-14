from tests.yopo_scenes import scene_path
import runpy
import numpy as np
import pytest
from uav_ac.YOPO.privileged import RollingEvaluator
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.planning.end2end.rolling import RollingMap
from uav_ac.planning.end2end.encoding import LocalFrame


def test_progress_and_dynamic_boundary_gradients_match_finite_difference():
    obs=runpy.run_path('tests/unit/planning/end2end/test_rolling.py')['moving_observation']()
    sim=MujocoSimulation(scene_path('free'),planning_path_capacity=0,record_actual_trajectory=False)
    oracle=RollingEvaluator.from_simulation(sim)
    mapping=RollingMap.from_observation(obs);free=mapping.seed(LocalFrame.from_observation(obs).local_goal(obs.goal))
    result=oracle.evaluate(obs,free[None]);assert 'progress' in result.components
    assert result.costs[0]==pytest.approx(sum(v[0] for v in result.components.values()))
    rng=np.random.default_rng(14)
    for _ in range(10):
        d=rng.normal(size=32);d/=np.linalg.norm(d)
        fd=(oracle.evaluate(obs,(free+1e-6*d)[None]).costs[0]-oracle.evaluate(obs,(free-1e-6*d)[None]).costs[0])/2e-6
        ad=result.free_gradients[0]@d
        assert abs(fd-ad)/max(1.,abs(fd),abs(ad))<1e-4
