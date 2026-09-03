import numpy as np
import pytest

from uav_ac.planning.corridor.firi import FIRI3D, FIRIConfig
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_region_should_expose_box_vertices_edges_and_containment():
    planner = FIRI3D(
        np.zeros((0, 3)),
        np.array([-1.0, -2.0, -3.0]),
        np.array([1.0, 2.0, 3.0]),
        FIRIConfig(max_iterations=1),
    )

    region = planner.inflate(np.zeros((1, 3)))

    assert len(region.vertices()) == 8
    assert len(region.edges()) == 12
    assert region.contains(np.zeros(3)) is True
    assert region.contains(np.array([2.0, 0.0, 0.0])) is False
    assert region.ellipsoid_volume > 0.0


def test_inflate_should_contain_seed_and_exclude_obstacle_samples():
    obstacles = np.array([
        [-1.0, 0.8, -1.0],
        [-1.0, 0.8, 1.0],
        [1.0, 0.8, -1.0],
        [1.0, 0.8, 1.0],
    ])
    seed = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    planner = FIRI3D(
        obstacles, -2.0 * np.ones(3), 2.0 * np.ones(3),
        FIRIConfig(max_iterations=4),
    )

    region = planner.inflate(seed)

    assert np.all(region.contains(seed))
    assert not np.any(region.contains(obstacles, tolerance=-1e-8))
    assert np.all(np.diag(region.shape) > 0.0)
    assert region.contains(region.center) is True
    assert np.all(
        region.A @ region.center + np.linalg.norm(region.A @ region.shape, axis=1)
        <= region.b + 1.0e-9)


def test_mvie_analytic_gradient_should_match_finite_difference():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(12, 3))
    variables = rng.normal(scale=0.3, size=9)
    variables[3:6] += 1.0
    _, analytic = FIRI3D._mvie_cost_gradient(
        variables, A, smoothing=1.0e-2, penalty=1.0e3)

    step = 1.0e-6
    finite_difference = np.empty(9)
    for index in range(9):
        plus, minus = variables.copy(), variables.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            FIRI3D._mvie_cost_gradient(plus, A, 1.0e-2, 1.0e3)[0]
            - FIRI3D._mvie_cost_gradient(minus, A, 1.0e-2, 1.0e3)[0]
        ) / (2.0 * step)
    assert analytic == pytest.approx(finite_difference, rel=5.0e-8, abs=2.0e-6)


def test_deepest_interior_should_recover_box_chebyshev_center():
    A = np.vstack((np.eye(3), -np.eye(3)))
    b = np.ones(6)

    center, depth = FIRI3D._deepest_interior(A, b)

    assert center == pytest.approx(np.zeros(3), abs=1.0e-10)
    assert depth == pytest.approx(1.0, abs=1.0e-10)


def test_from_aabbs_should_sample_padded_obstacle_surface():
    obstacles = np.array([[-0.5, 0.5, 0.8, 1.2, -0.5, 0.5]])

    planner = FIRI3D.from_aabbs(
        obstacles,
        -2.0 * np.ones(3),
        2.0 * np.ones(3),
        surface_spacing=0.5,
        obstacle_padding=0.1,
    )

    assert np.min(planner.obstacle_points, axis=0) == pytest.approx([-0.6, 0.7, -0.6])
    assert np.max(planner.obstacle_points, axis=0) == pytest.approx([0.6, 1.3, 0.6])


def test_path_sampling_should_preserve_rrt_corners():
    path = np.array([[0., 0., 0.], [1., 2., 0.], [3., 2., 1.]])

    samples = FIRI3D.sample_path_preserving_vertices(path, spacing=10.0)

    assert samples == pytest.approx(path)


def test_minimum_norm_solver_should_project_origin_to_feasible_boundary():
    normals = np.array([[-1.0, 0.0, 0.0]])
    bounds = np.array([-2.0])

    point, feasible = FIRI3D._minimum_norm_point(normals, bounds)

    assert feasible is True
    assert point == pytest.approx([2.0, 0.0, 0.0])


def test_safe_corridor_should_follow_geometric_path_in_laboratory_scene():
    simulation = MujocoSimulation()
    obstacle_points = simulation.get_planning_obstacle_points(
        spacing=0.7, padding=0.15)
    planner = FIRI3D(
        obstacle_points, simulation.space_limits[0], simulation.space_limits[1])
    path = simulation.mission_waypoints[1:]
    path_samples = planner.sample_path(path, spacing=1.0)

    regions = planner.build_safe_flight_corridor(path, seed_spacing=1.0)
    surface_count = simulation.set_convex_polyhedra_visualization(regions)

    assert len(regions) == len(path_samples) - 2
    assert surface_count == len(regions)
    assert np.count_nonzero(
        simulation.model.geom_rgba[simulation._corridor_region_ids, 3]) == surface_count
    for index, region in enumerate(regions):
        assert np.all(region.contains(path_samples[index:index + 3]))
        assert not np.any(region.contains(obstacle_points, tolerance=-1e-7))


def test_free_space_cover_should_reach_target_in_laboratory_scene():
    simulation = MujocoSimulation()
    planner = FIRI3D(
        simulation.get_planning_obstacle_points(spacing=0.7, padding=0.15),
        simulation.space_limits[0],
        simulation.space_limits[1],
    )
    free_samples = simulation.sample_free_space(
        spacing=(3.0, 3.0, 1.5), clearance=0.15)

    cover = planner.cover_free_space(
        free_samples,
        local_half_size=(4.0, 4.0, 2.5),
        target_coverage=0.9,
        max_regions=24,
        max_iterations=1,
    )

    assert cover.coverage_fraction >= 0.9
    assert len(cover.regions) <= 24
    assert np.all(cover.covered == np.any(np.stack([
        region.contains(free_samples) for region in cover.regions
    ]), axis=0))
