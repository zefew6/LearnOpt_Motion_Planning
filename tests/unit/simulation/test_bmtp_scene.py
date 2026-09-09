"""Scene contract independent of any BMTP route loader or planner."""

from itertools import product
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation
from uav_ac.simulation.recording import default_camera


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


def _xml_signature(element):
    return element.tag, element.attrib, [_xml_signature(child) for child in element]


def test_bmtp_reuses_laboratory_vehicle_physics():
    village = ET.parse(SCENE_PATH).getroot()
    lab = ET.parse(DEFAULT_SCENE_PATH).getroot()
    for tag in ("compiler", "option", "size"):
        assert _xml_signature(village.find(tag)) == _xml_signature(lab.find(tag))
    for numeric in lab.findall("custom/numeric"):
        if numeric.get("name") != "planning_bounds":
            assert village.find(f"custom/numeric[@name='{numeric.get('name')}']").attrib == numeric.attrib
    vehicle = village.find("worldbody/body[@name='quadrotor']")
    reference = lab.find("worldbody/body[@name='quadrotor']")
    # Only the initial world position differs, not the vehicle definition.
    vehicle.attrib["pos"] = reference.attrib["pos"]
    assert _xml_signature(vehicle) == _xml_signature(reference)


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
