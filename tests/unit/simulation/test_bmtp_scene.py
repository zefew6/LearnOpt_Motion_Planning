"""Scene contract independent of any BMTP route loader or planner."""

from itertools import product

import mujoco
import numpy as np
import pytest

from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation
from uav_ac.simulation.recording import default_camera
from uav_ac.robot.quadrotor.quad import (
    DEFAULT_DRAG_TO_THRUST, DEFAULT_FORCE_COEFFICIENT,
    DEFAULT_MOTOR_TIME_CONSTANTS, DEFAULT_THRUST_LIMITS,
)


SCENE_PATH = DEFAULT_SCENE_PATH.with_name("bmtp_village.xml")
SEED_ROUTES = [
    [(-0.2, -0.2, -2), (-0.2, 8.4, -2.75), (-0.2, 17, -3.5),
     (-0.2, 25.6, -4.25), (-0.2, 34.2, -5), (8.4, 34.2, -5.75),
     (17, 34.2, -6.5), (25.6, 34.2, -7.25), (34.2, 34.2, -8)],
]


@pytest.fixture
def simulation():
    return MujocoSimulation(SCENE_PATH, record_actual_trajectory=False)


def _segment_hits_box(start, end, lower, upper):
    """Slab intersection on the entire closed segment, including tangencies."""
    enter, leave = 0.0, 1.0
    for origin, delta, low, high in zip(start, end - start, lower, upper):
        if abs(delta) < 1e-12:
            if origin < low or origin > high:
                return False
            continue
        first, last = sorted(((low - origin) / delta, (high - origin) / delta))
        enter, leave = max(enter, first), min(leave, last)
        if enter > leave:
            return False
    return True


def test_bmtp_scene_loads_requested_mission_without_initial_contact(simulation):
    assert simulation.start_position == pytest.approx([-0.2, -0.2, -2])
    assert simulation.goal_position == pytest.approx([34.2, 34.2, -8])
    assert simulation.space_limits == pytest.approx(np.array([[-1, -1, -11], [35, 35, 1]]))
    assert simulation.data.ncon == 0
    assert not simulation.has_collision
    assert not simulation.collision_detected
    # The current loader requires a waypoint; it imposes no extra location.
    assert simulation.mission_waypoints == pytest.approx(
        np.array([[-0.2, -0.2, -2], [-0.2, -0.2, -2], [34.2, 34.2, -8]]))
    assert simulation.model.site("waypoint_00").rgba[3] == 0


def test_bmtp_scene_recreates_the_paper_village_with_521_convex_obstacles(simulation):
    boxes = simulation.obstacles
    assert boxes.shape == (521, 6)
    np.testing.assert_allclose(boxes[0], [-0.5, 34.5, -0.5, 34.5, 0.01, 0.02])
    geom = simulation.model.geom("obstacle_000")
    assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
    assert geom.bodyid == 0
    assert geom.contype != 0 and geom.conaffinity != 0
    np.testing.assert_allclose(simulation.data.geom_xmat[geom.id].reshape(3, 3), np.eye(3))
    assert any(_segment_hits_box(simulation.start_position, simulation.goal_position,
                                 box[::2], box[1::2]) for box in boxes[1:])
    assert simulation.model.geom("ground").type == mujoco.mjtGeom.mjGEOM_PLANE


def test_bmtp_seed_sites_encode_the_paper_outer_perimeter_initialization(simulation):
    route_index = 0
    names = sorted(
        simulation.model.site(index).name for index in range(simulation.model.nsite)
        if simulation.model.site(index).name.startswith(f"bmtp_route_{route_index:02d}_")
    )
    assert names == [f"bmtp_route_{route_index:02d}_{index:02d}"
                     for index in range(len(SEED_ROUTES[route_index]))]
    sites = [simulation.model.site(name) for name in names]
    # Convert independently of the loader's route support and transform constant.
    route = np.array([simulation.data.site_xpos[site.id] for site in sites]) * [1, -1, -1]
    np.testing.assert_allclose(route, SEED_ROUTES[route_index])
    assert all(site.rgba[3] == 0 and site.bodyid == 0 for site in sites)
    for start, end in zip(route[:-1], route[1:]):
        for index, box in enumerate(simulation.obstacles):
            assert not _segment_hits_box(start, end, box[::2], box[1::2]), (
                f"route {route_index}, segment {start} -> {end}, obstacle {index}"
            )
    route_names = [simulation.model.site(index).name for index in range(simulation.model.nsite)
                   if simulation.model.site(index).name.startswith("bmtp_route_")]
    assert len(route_names) == len(SEED_ROUTES[0])


@pytest.mark.parametrize(("start", "end", "expected"), [
    ((-1, 0.5, 0.5), (2, 0.5, 0.5), True),  # Both endpoints outside; interior crosses.
    ((-1, 1, 0.5), (2, 1, 0.5), True),      # Tangency counts as collision.
    ((-1, 2, 0.5), (2, 2, 0.5), False),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True),
])
def test_segment_box_helper_checks_full_segments(start, end, expected):
    assert _segment_hits_box(np.array(start), np.array(end), np.zeros(3), np.ones(3)) == expected


def test_bmtp_reuses_laboratory_vehicle_physics():
    village = mujoco.MjModel.from_xml_path(str(SCENE_PATH.resolve()))
    lab = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH.resolve()))
    village_body = mujoco.mj_name2id(village, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    lab_body = mujoco.mj_name2id(lab, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    assert village.body_mass[village_body] == lab.body_mass[lab_body]
    np.testing.assert_array_equal(village.body_inertia[village_body], lab.body_inertia[lab_body])

    def vehicle_geometry(model, body_id):
        start = model.body_geomadr[body_id]
        stop = start + model.body_geomnum[body_id]
        return {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id): (
                model.geom_type[geom_id], model.geom_size[geom_id].copy(),
                model.geom_pos[geom_id].copy(), model.geom_rgba[geom_id].copy(),
                model.geom_contype[geom_id], model.geom_conaffinity[geom_id],
            )
            for geom_id in range(start, stop)
        }

    village_geometry = vehicle_geometry(village, village_body)
    lab_geometry = vehicle_geometry(lab, lab_body)
    assert village_geometry.keys() == lab_geometry.keys()
    for name in village_geometry:
        for village_value, lab_value in zip(village_geometry[name], lab_geometry[name]):
            np.testing.assert_array_equal(village_value, lab_value)

    village_quad = MujocoSimulation(SCENE_PATH, record_actual_trajectory=False).quad
    lab_quad = MujocoSimulation(DEFAULT_SCENE_PATH, record_actual_trajectory=False).quad
    assert village_quad.kf == lab_quad.kf == DEFAULT_FORCE_COEFFICIENT
    assert village_quad.kappa == lab_quad.kappa == DEFAULT_DRAG_TO_THRUST
    np.testing.assert_array_equal(
        [village_quad.min_thrust, village_quad.max_thrust], DEFAULT_THRUST_LIMITS)
    np.testing.assert_array_equal(
        [village_quad.motor_rise_time_constant, village_quad.motor_fall_time_constant],
        DEFAULT_MOTOR_TIME_CONSTANTS)


def test_bmtp_vehicle_dynamics_match_laboratory_scene(simulation):
    reference = MujocoSimulation(DEFAULT_SCENE_PATH, record_actual_trajectory=False)
    state = simulation.quad.X.copy()
    motors = np.array([1.0, 1.1, 1.2, 1.3])
    for scene in (simulation, reference):
        scene.reset(state, motors)
        for _ in range(20):
            scene.step()
        assert not scene.has_collision
    np.testing.assert_allclose(simulation.quad.X, reference.quad.X, rtol=0, atol=1e-12)


def test_bmtp_default_camera_frames_whole_village(simulation):
    camera = default_camera(simulation.model)
    np.testing.assert_allclose(camera.lookat, [17, -17, 5])
    assert camera.elevation < 0
    corners = np.array(list(product((-1, 35), (-35, 1), (0, 11))))
    radius = np.linalg.norm(corners - camera.lookat, axis=1).max()
    # A sphere containing the entire planning volume fits the vertical FOV.
    half_fov = np.deg2rad(simulation.model.vis.global_.fovy / 2)
    assert radius < camera.distance * np.sin(half_fov)
    overhead = simulation.model.camera("overhead")
    np.testing.assert_allclose(overhead.pos[:2], [17, -17])
    np.testing.assert_allclose(overhead.quat, [1, 0, 0, 0])
    assert (overhead.pos[2] - 11) * np.tan(np.deg2rad(overhead.fovy[0] / 2)) > 18
