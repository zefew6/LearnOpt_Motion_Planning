import numpy as np
import pytest

from uav_ac.planning.geometry import ConvexPolytope
from uav_ac.planning.pipeline import build_gcs_corridor, gcs_controller_trajectory
from uav_ac.planning.trajectory.gcs import GCSConfig, GCSGraph, GCSPlanner
from uav_ac.planning.trajectory.gcs.bezier import (
    derivative_energy_matrix,
    endpoint_derivative_coefficients,
)
from uav_ac.planning.trajectory.gcs.formulation import solve_relaxation
from uav_ac.simulation.mujoco_sim import GCS_BUILDING_SCENE_PATH, MujocoSimulation


def _box(lower, upper):
    return ConvexPolytope(
        np.vstack((np.eye(3), -np.eye(3))),
        np.concatenate((np.asarray(upper), -np.asarray(lower))),
    )


def _branching_regions():
    return [
        _box([0.0, -0.7, -0.2], [1.1, 0.7, 0.2]),
        _box([0.9, 0.5, -0.2], [2.1, 1.5, 0.2]),
        _box([0.9, -0.6, -0.2], [2.1, -0.3, 0.2]),
        _box([1.9, -0.7, -0.2], [3.0, 0.7, 0.2]),
    ]


def test_gcs_should_choose_shorter_branch_and_return_c1_bezier_path():
    start = np.array([0.1, 0.0, 0.0])
    goal = np.array([2.9, 0.0, 0.0])
    regions = _branching_regions()

    trajectory = GCSPlanner(GCSConfig(degree=3, continuity=1)).plan(
        start, goal, regions)

    assert trajectory.region_indices.tolist() == [0, 2, 3]
    assert trajectory.sample(20)[0] == pytest.approx(start, abs=2.0e-7)
    assert trajectory.sample(20)[-1] == pytest.approx(goal, abs=2.0e-7)
    for region_index, control_points in zip(
            trajectory.region_indices, trajectory.control_points, strict=True):
        region = regions[int(region_index)]
        assert np.max(region.A @ control_points.T - region.b[:, None]) <= 2.0e-6
    end_velocity = endpoint_derivative_coefficients(3, 1, at_end=True)
    start_velocity = endpoint_derivative_coefficients(3, 1, at_end=False)
    for first, second in zip(
            trajectory.control_points[:-1], trajectory.control_points[1:], strict=True):
        assert first[-1] == pytest.approx(second[0], abs=2.0e-6)
        assert end_velocity @ first == pytest.approx(
            start_velocity @ second, abs=2.0e-6)


def test_gcs_relaxation_should_satisfy_flow_and_perspective_containment():
    start = np.array([0.1, 0.0, 0.0])
    goal = np.array([2.9, 0.0, 0.0])
    graph = GCSGraph.from_regions(_branching_regions(), start, goal)
    config = GCSConfig()

    solution = solve_relaxation(graph, start, goal, config)

    for vertex in range(len(graph.regions)):
        outgoing = [i for i, edge in enumerate(graph.edges) if edge[0] == vertex]
        incoming = [i for i, edge in enumerate(graph.edges) if edge[1] == vertex]
        residual = np.sum(solution.flows[outgoing]) - np.sum(solution.flows[incoming])
        expected = 1.0 if vertex == graph.source else -1.0 if vertex == graph.target else 0.0
        assert residual == pytest.approx(expected, abs=2.0e-6)
    for edge_index, (tail, head) in enumerate(graph.edges):
        flow = solution.flows[edge_index]
        tail_region = graph.regions[tail]
        head_region = graph.regions[head]
        assert np.max(
            tail_region.A @ solution.tail_control_points[edge_index].T
            - tail_region.b[:, None] * flow) <= 2.0e-6
        assert np.max(
            head_region.A @ solution.head_control_points[edge_index].T
            - head_region.b[:, None] * flow) <= 2.0e-6


def test_gcs_should_reject_disconnected_region_cover():
    regions = [
        _box([0.0, -0.2, -0.2], [1.0, 0.2, 0.2]),
        _box([2.0, -0.2, -0.2], [3.0, 0.2, 0.2]),
    ]
    with pytest.raises(ValueError, match="does not connect"):
        GCSPlanner().plan(np.array([0.1, 0.0, 0.0]),
                          np.array([2.9, 0.0, 0.0]), regions)


def test_gcs_configuration_should_validate_degree_and_continuity():
    with pytest.raises(ValueError, match="degree"):
        GCSConfig(degree=0)
    with pytest.raises(ValueError, match="continuity"):
        GCSConfig(degree=2, continuity=2)


def test_bezier_derivative_energy_should_match_numerical_quadrature():
    rng = np.random.default_rng(4)
    control = rng.normal(size=(6, 3))
    matrix = derivative_energy_matrix(5, 2)
    exact = np.trace(control.T @ matrix @ control)
    derivative_control = 20.0 * np.diff(control, n=2, axis=0)
    times = np.linspace(0.0, 1.0, 10001)
    values = np.stack([
        (1.0 - times) ** 3,
        3.0 * (1.0 - times) ** 2 * times,
        3.0 * (1.0 - times) * times**2,
        times**3,
    ], axis=1) @ derivative_control
    numerical = np.trapezoid(np.sum(values * values, axis=1), times)
    assert exact == pytest.approx(numerical, rel=2.0e-7)


def test_default_gcs_should_return_quintic_c2_trajectory():
    trajectory = GCSPlanner().plan(
        np.array([0.1, 0.0, 0.0]), np.array([2.9, 0.0, 0.0]),
        _branching_regions())
    assert trajectory.control_points.shape[1:] == (6, 3)
    for derivative in range(3):
        end = endpoint_derivative_coefficients(5, derivative, at_end=True)
        begin = endpoint_derivative_coefficients(5, derivative, at_end=False)
        for first, second in zip(
                trajectory.control_points[:-1], trajectory.control_points[1:], strict=True):
            assert end @ first == pytest.approx(begin @ second, abs=3.0e-6)


def test_gcs_controller_samples_should_obey_speed_and_rest_boundaries():
    start = np.array([0.1, 0.0, 0.0])
    goal = np.array([2.9, 0.0, 0.0])
    trajectory = GCSPlanner().plan(start, goal, _branching_regions())

    samples = gcs_controller_trajectory(
        trajectory, max_velocity=1.5, dt=0.01)

    assert samples.shape[1] == 10
    assert np.all(np.isfinite(samples))
    assert samples[[0, -1], :3] == pytest.approx(
        np.vstack((start, goal)), abs=3.0e-6)
    assert np.max(np.linalg.norm(samples[:, 3:6], axis=1)) <= 1.5 + 1.0e-8
    assert samples[[0, -1], 3:9] == pytest.approx(0.0, abs=3.0e-6)
    path_length = np.sum(np.linalg.norm(
        np.diff(trajectory.sample(400), axis=0), axis=1))
    duration = (len(samples) - 1) * 0.01
    assert duration <= 1.3 * path_length / 1.5 + 0.01


def test_two_storey_maze_should_cover_free_space_and_use_southwest_stair():
    simulation = MujocoSimulation(GCS_BUILDING_SCENE_PATH)

    corridor = build_gcs_corridor(simulation)
    graph = GCSGraph.from_regions(
        corridor.regions, simulation.start_position, simulation.goal_position)
    free_samples = simulation.sample_free_space(
        spacing=(0.75, 0.75, 0.5), clearance=0.15)
    covered = np.any(np.stack([
        region.contains(free_samples) for region in corridor.regions
    ]), axis=0)

    assert len(corridor.regions) <= 40
    assert np.mean(covered) >= 0.95
    # In NED coordinates northeast has large x/small y; southeast has large x/y.
    assert simulation.start_position[:2] == pytest.approx([22.0, 2.0])
    assert simulation.goal_position == pytest.approx([22.0, 12.0, -4.4])

    stair_regions = []
    for index, region in enumerate(corridor.regions):
        vertices = region.vertices()
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        if (lower[2] <= -3.0 <= upper[2]
                and upper[0] <= 5.5 and lower[1] >= 9.0):
            stair_regions.append(index)
    assert stair_regions
    without_stair = [
        region for index, region in enumerate(corridor.regions)
        if index not in stair_regions
    ]
    with pytest.raises(ValueError, match="does not connect"):
        GCSGraph.from_regions(
            without_stair, simulation.start_position, simulation.goal_position)

    trajectory = GCSPlanner().plan(
        simulation.start_position, simulation.goal_position, corridor.regions)
    crossing = trajectory.sample(100)
    crossing = crossing[(crossing[:, 2] < -2.8) & (crossing[:, 2] > -3.2)]
    controller_samples = gcs_controller_trajectory(
        trajectory, max_velocity=3.0, dt=0.01)
    assert len(crossing) > 0
    assert np.all(crossing[:, 0] < 5.0)
    assert np.all(crossing[:, 1] > 9.0)
    assert np.max(np.abs(controller_samples[:, 5])) <= 1.0 + 1.0e-8
    assert np.max(np.linalg.norm(
        controller_samples[:, 6:9], axis=1)) <= 6.0 + 1.0e-8
