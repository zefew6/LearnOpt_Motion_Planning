from types import SimpleNamespace

import numpy as np
import pytest

from uav_ac.planning.geometry.esdf import ESDF
from uav_ac.planning.geometry.grid_map import GridMap
from uav_ac.planning.trajectory.aerial_manipulator_minco import AerialManipulatorMINCOConfig
from uav_ac.scenes.loader import SceneGeometry


def simulation():
    ground = SceneGeometry('ground', 'plane', np.zeros(3), np.eye(3), np.zeros(3), 1, 1)
    return SimpleNamespace(
        space_limits=np.array([[-.4, -.4, -1.], [.4, .4, -.3]]),
        scene_geometries=(ground,), pick_place=None, quad=object(),
        robot=SimpleNamespace(configuration=np.r_[0., 0., -.5, 1., 0., 0., 0., np.zeros(5)],
                              base_inscribed_collision_radius=lambda: .02))


def settings():
    return dict(pick_position_ned=[-.1, 0., -.5], place_position_ned=[.1, 0., -.5],
                gripper_open=.06, gripper_closed=0., pick_yaw=0., place_yaw=0.,
                pick_nominal_joints=[0.]*4, place_nominal_joints=[0.]*4)


def test_planner_initializes_once_and_reuses_maps_for_both_legs(monkeypatch):
    from uav_ac.planning.pipeline import aerial_pick_place as pipeline

    counts = dict(occupancy=0, esdf=0, search_grid=0)
    original_occupancy = GridMap.from_scene_geometries
    original_esdf = ESDF.from_occupancy
    original_search_grid = GridMap.to_pathfinding3d_matrix

    def build_occupancy(*args, **kwargs):
        counts['occupancy'] += 1
        return original_occupancy(*args, **kwargs)

    def build_esdf(*args, **kwargs):
        counts['esdf'] += 1
        return original_esdf(*args, **kwargs)

    def build_search_grid(self, *args, **kwargs):
        counts['search_grid'] += 1
        return original_search_grid(self, *args, **kwargs)

    monkeypatch.setattr(GridMap, 'from_scene_geometries', build_occupancy)
    monkeypatch.setattr(ESDF, 'from_occupancy', build_esdf)
    monkeypatch.setattr(GridMap, 'to_pathfinding3d_matrix', build_search_grid)
    cfg = AerialManipulatorMINCOConfig(esdf_resolution=.1, esdf_discretization_margin=.18,
                                      astar_grid_resolution=.2, astar_fallback_grid_resolution=.1)
    planner = pipeline.PickPlacePlanner(simulation(), {}, settings=settings(), planner_config=cfg)
    assert counts == dict(occupancy=1, esdf=1, search_grid=2)

    def unexpected(*args, **kwargs):
        pytest.fail('planning reconstructed initialized maps')

    monkeypatch.setattr(GridMap, 'from_scene_geometries', unexpected)
    monkeypatch.setattr(ESDF, 'from_occupancy', unexpected)
    monkeypatch.setattr(GridMap, 'to_pathfinding3d_matrix', unexpected)
    monkeypatch.setattr(pipeline, 'make_terminal_state',
                        lambda robot, position, yaw, joints, **_: np.r_[position, yaw, joints])
    seen = []

    def search(self, start, goal, **kwargs):
        seen.append((kwargs['occupancy'], kwargs['esdf'], kwargs['astar_maps']))
        return SimpleNamespace(path=np.array([start, goal]), exact_solution=True, metrics={})

    monkeypatch.setattr(pipeline.AerialManipulatorMINCO, 'plan', search)
    for _ in range(2):
        diagnostics = {}
        result = planner.plan(search_only=True, diagnostics=diagnostics)
        assert set(result['searches']) == {'pick', 'place'}
        assert diagnostics['planning_seconds'] >= planner.initialization_seconds
        assert diagnostics['esdf_grid_shape'] == list(planner.occupancy.shape)
    assert len(seen) == 4
    assert all(occupancy is planner.occupancy and esdf is planner.esdf
               and maps is planner.astar_maps for occupancy, esdf, maps in seen)


def test_disabled_guidance_does_not_prepare_a_star_maps():
    from uav_ac.planning.pipeline.aerial_pick_place import PickPlacePlanner

    cfg = AerialManipulatorMINCOConfig(esdf_resolution=.1, esdf_discretization_margin=.18,
                                      astar_guidance_enabled=False)
    planner = PickPlacePlanner(simulation(), {}, settings=settings(), planner_config=cfg)
    assert planner.astar_maps is None
    assert planner.esdf is not None


@pytest.mark.parametrize('invalid_leg', ['pick', 'place'])
def test_validation_failure_is_reported_without_blocking_execution(monkeypatch, invalid_leg):
    from uav_ac.planning.pipeline import aerial_pick_place as pipeline

    cfg = AerialManipulatorMINCOConfig(esdf_resolution=.1, esdf_discretization_margin=.18,
                                      astar_guidance_enabled=False)
    planner = pipeline.PickPlacePlanner(simulation(), {}, settings=settings(), planner_config=cfg)
    monkeypatch.setattr(pipeline, 'make_terminal_state',
                        lambda robot, position, yaw, joints, **_: np.r_[position, yaw, joints])
    planned_legs, published = [], []

    def plan_leg(self, start, goal, **kwargs):
        name = 'place' if kwargs['carry_payload'] else 'pick'
        planned_legs.append(name)
        valid = name != invalid_leg
        self.last_metrics = dict(maximum_violation_kind='world_clearance')
        return SimpleNamespace(
            validation_passed=valid, maximum_violation=0. if valid else .06,
            minimum_clearance=.1 if valid else -.01,
            optimizer_converged=True, iterations=1, validation_sample_dt=.025,
            total_time=1., evaluate=lambda times: np.tile(start, (len(times), 1)))

    monkeypatch.setattr(pipeline.AerialManipulatorMINCO, 'plan', plan_leg)
    diagnostics = {}
    result = planner.plan(diagnostics=diagnostics,
                          on_minco_trajectory=lambda name, _: published.append(name))
    assert planned_legs == ['pick', 'place']
    assert published == ['pick', 'place']
    assert not diagnostics['legs'][invalid_leg]['validation_passed']
    assert diagnostics['legs'][invalid_leg]['minimum_clearance'] == -.01
    assert not result['plans'][invalid_leg].validation_passed
    assert set(result['plans']) == {'pick', 'place'}
