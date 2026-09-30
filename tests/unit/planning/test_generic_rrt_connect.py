import numpy as np
import pytest
from ompl import base as ob

from uav_ac.planning.search.rrt_connect import (
    RRTConnectPlanningError, StateSpaceAdapter, plan_rrt_connect,
)


class VectorStateSpaceAdapter:
    def __init__(self):
        self.space = None
        self.bounds_history = []

    def create_space(self, lower, upper, metric_scale):
        assert len(lower) == len(upper) == len(metric_scale) == 2
        self.space = ob.RealVectorStateSpace(2)
        self.set_sampling_bounds(lower, upper)
        return self.space

    def read_state(self, state):
        return np.asarray([state[index] for index in range(2)], dtype=float)

    def write_state(self, state, values):
        values = np.asarray(values, dtype=float)
        for index, value in enumerate(values):
            state[index] = float(value)

    def set_sampling_bounds(self, lower, upper):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        bounds = ob.RealVectorBounds(2)
        for index in range(2):
            bounds.setLow(index, float(lower[index]))
            bounds.setHigh(index, float(upper[index]))
        self.space.setBounds(bounds)
        self.bounds_history.append((lower.copy(), upper.copy()))


def test_generic_rrt_connect_uses_adapter_and_staged_sampling_bounds():
    adapter = VectorStateSpaceAdapter()
    assert isinstance(adapter, StateSpaceAdapter)
    lower, upper = np.array([-1., -1.]), np.array([2., 2.])
    start, goal = np.array([0., 0.]), np.array([1., 1.])
    checked_edges = []

    def edge_valid(first, second):
        checked_edges.append((first.copy(), second.copy()))
        return True

    path, metrics = plan_rrt_connect(
        adapter, start, goal, lower, upper, np.ones(2),
        state_valid=lambda state: True,
        edge_valid=edge_valid,
        range_size=.4,
        sampling_regions=[
            {"name": "local", "lower": [-.2, -.2], "upper": [1.2, 1.2], "fraction": .2},
            {"name": "global", "lower": lower, "upper": upper, "fraction": .8},
        ],
        timeout_s=1.,
    )

    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_array_equal(path[-1], goal)
    assert checked_edges
    assert "local" in metrics["rrt_sampling_stage_seconds"]
    np.testing.assert_allclose(adapter.bounds_history[1][0], [-.2, -.2])
    np.testing.assert_allclose(adapter.bounds_history[1][1], [1.2, 1.2])


def test_generic_rrt_connect_reports_timeout_metrics_without_solution():
    adapter = VectorStateSpaceAdapter()
    with pytest.raises(RRTConnectPlanningError) as caught:
        plan_rrt_connect(
            adapter, np.zeros(2), np.ones(2), np.zeros(2), np.ones(2), np.ones(2),
            state_valid=lambda state: True,
            edge_valid=lambda first, second: False,
            range_size=.2,
            timeout_s=.03,
        )

    assert caught.value.metrics["rrt_sampling_stage_seconds"]
    assert "rrt_motion_checks" in caught.value.metrics
