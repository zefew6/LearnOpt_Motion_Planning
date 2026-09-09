import xml.etree.ElementTree as ET

import numpy as np
import pytest

from uav_ac.scenes import load_scene
from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, OPEN_FIELD_SCENE_PATH, MujocoSimulation
from uav_ac.tasks.trajectory_tracking import MujocoTrajectoryTrackingEnv


@pytest.mark.parametrize("path", [DEFAULT_SCENE_PATH, OPEN_FIELD_SCENE_PATH])
def test_explicit_scene_metadata_matches_simulation(path):
    scene = load_scene(path)
    simulation = MujocoSimulation(path)
    assert scene.model_path == path.resolve()
    for name in ("start_position", "goal_position", "mission_waypoints", "space_limits", "obstacles"):
        np.testing.assert_array_equal(getattr(scene, name), getattr(simulation, name))


def test_scene_loader_never_substitutes_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(tmp_path / "unknown.xml")


@pytest.mark.parametrize("missing", ["goal", "waypoints", "planning_bounds"])
def test_tracking_validates_optional_scene_requirements_separately(tmp_path, missing):
    root = ET.parse(OPEN_FIELD_SCENE_PATH).getroot()
    for parent in root.iter():
        for child in list(parent):
            name = child.get("name", "")
            if name == missing or (missing == "waypoints" and name.startswith("waypoint_")):
                parent.remove(child)
    path = tmp_path / "optional.xml"
    ET.ElementTree(root).write(path)
    simulation = MujocoSimulation(path)
    simulation.step()
    with pytest.raises(ValueError, match="trajectory tracking requires"):
        MujocoTrajectoryTrackingEnv(np.zeros((1, 10)), model_path=path)
