import shutil
from pathlib import Path

import numpy as np
import pytest

from uav_ac.planning.search import RRTStar


@pytest.fixture
def rrt_object():
    return RRTStar(space_limits=np.array([[0, 0, 0], [10, 10, 10]]),
                   start=np.array([0, 0, 0]),
                   goal=np.array([8, 8, 8]),
                   max_distance=2,
                   max_iterations=1)


@pytest.fixture
def copy_scene_with_models(tmp_path):
    """Copy a scene and its relative robot includes while keeping package layout."""
    def copy(scene_path, filename=None):
        scene_path = Path(scene_path).resolve()
        package_root = scene_path.parents[2]
        target_root = tmp_path / "uav_ac"
        target_scene = target_root / "simulation" / "models" / (filename or scene_path.name)
        target_scene.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package_root / "simulation" / "model",
                        target_root / "simulation" / "model", dirs_exist_ok=True)
        shutil.copy2(scene_path, target_scene)
        return target_scene

    return copy
