"""Generated YOPO scene fixtures, kept outside the interactive scene inventory."""
from functools import lru_cache
from pathlib import Path
import tempfile
from uav_ac.yopo.whole_body.data.scenes import write_debug_scene


@lru_cache(maxsize=4)
def scene_path(family='free'):
    root=Path(__file__).resolve().parents[1]/'runs/yopo_whole_body/test_scenes'
    root.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='fixture_',dir=root))
    return write_debug_scene(directory,family)
