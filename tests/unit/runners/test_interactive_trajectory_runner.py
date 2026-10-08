from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def prepare(monkeypatch, *, dt=0.01, xml_wind=(0, 0, 0)):
    from uav_ac.runners import trajectory_tracking as runner

    simulation = SimpleNamespace(
        quad=SimpleNamespace(dt=0.001, position=np.zeros(3)),
        model=SimpleNamespace(opt=SimpleNamespace(wind=np.array(xml_wind))),
        data=SimpleNamespace(time=1.25), goal_position=np.zeros(3),
        collision_detected=False, set_trajectory_visualization=Mock(),
        set_external_force_world=Mock(), run_interactive=Mock(),
    )
    tracker = Mock()
    constructor = Mock(return_value=simulation)
    monkeypatch.setattr(runner, 'MujocoSimulation', constructor)
    monkeypatch.setattr(runner, 'build_controller', lambda *_: (object(), dt))
    monkeypatch.setattr(runner, 'plan_trajectory', lambda *_: np.zeros((2, 10)))
    monkeypatch.setattr(runner, 'TrajectoryController', Mock(return_value=tracker))
    config = dict(scene='scene.xml', planner='gcopter', controller='cascaded',
                  wind='none', wind_options={}, follow_camera=True)
    return runner, config, simulation, tracker, constructor


def test_wind_is_converted_before_tracking_and_reset_clears_force(monkeypatch):
    runner, config, simulation, tracker, _ = prepare(monkeypatch)
    config['wind'] = 'fixed_gust'
    events = []
    wind = SimpleNamespace(force_ned=lambda time: np.array([1., 2., 3.]))
    monkeypatch.setattr(runner, 'GustingCrosswind', lambda **_: wind)
    simulation.set_external_force_world.side_effect = lambda force: events.append(force.copy())
    tracker.step.side_effect = lambda: events.append('step')
    tracker.reset.side_effect = lambda: events.append('reset')
    runner.run_trajectory_tracking(config)
    step, reset = simulation.run_interactive.call_args.args
    step()
    reset()
    np.testing.assert_array_equal(events[0], [1., -2., -3.])
    assert events[1:3] == ['step', 'reset']
    np.testing.assert_array_equal(events[3], [0., 0., 0.])
    assert simulation.run_interactive.call_args.kwargs['chase_camera'] is True


@pytest.mark.parametrize('dt', [0.0005, 0.0015])
def test_controller_period_rejected_before_planning(monkeypatch, dt):
    runner, config, _, _, _ = prepare(monkeypatch, dt=dt)
    monkeypatch.setattr(runner, 'plan_trajectory', lambda *_: pytest.fail('planned invalid period'))
    with pytest.raises(ValueError, match='controller period'):
        runner.run_trajectory_tracking(config)


def test_xml_wind_conflict_rejected_before_controller_creation(monkeypatch):
    runner, config, _, _, _ = prepare(monkeypatch, xml_wind=(1, 0, 0))
    config['wind'] = 'fixed_gust'
    monkeypatch.setattr(runner, 'build_controller', lambda *_: pytest.fail('built conflicting wind'))
    with pytest.raises(ValueError, match='conflicts'):
        runner.run_trajectory_tracking(config)


@pytest.mark.parametrize('planner, capacity', [('bmtp', 2), ('gcopter', 0)])
def test_planning_visualization_capacity(monkeypatch, planner, capacity):
    runner, config, _, _, constructor = prepare(monkeypatch)
    config['planner'] = planner
    runner.run_trajectory_tracking(config)
    assert constructor.call_args.kwargs['planning_path_capacity'] == capacity
