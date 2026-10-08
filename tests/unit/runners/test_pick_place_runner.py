from types import SimpleNamespace

from uav_ac.main import load_config
from uav_ac.runners import aerial_pick_place as runner


def test_early_planning_failure_includes_map_initialization_time(monkeypatch):
    config = load_config('configs/aerial_manipulator_workcell.yaml')
    config['visualize'] = False

    def fail_plan(**kwargs):
        raise ValueError('target unreachable')

    context = SimpleNamespace(
        initialization_seconds=5.,
        initialization_metrics=dict(occupancy_seconds=2., esdf_seconds=1., astar_map_seconds=2.),
        plan=fail_plan)
    monkeypatch.setattr(runner, 'PickPlacePlanner', lambda *args, **kwargs: context)
    result = runner.run_aerial_pick_place(config)
    assert result['state'] == 'FAILED'
    assert result['failure_reason'] == 'target unreachable'
    assert result['planning_metrics']['planning_seconds'] >= 5.


def test_map_initialization_failure_returns_failed_task_result(monkeypatch):
    config = load_config('configs/aerial_manipulator_workcell.yaml')
    config['visualize'] = False

    def fail_initialization(*args, **kwargs):
        raise ValueError('invalid planning bounds')

    monkeypatch.setattr(runner, 'PickPlacePlanner', fail_initialization)
    result = runner.run_aerial_pick_place(config)
    assert result['state'] == 'FAILED'
    assert result['failure_reason'] == 'invalid planning bounds'
    assert not result['pick_plan_valid'] and not result['place_plan_valid']
