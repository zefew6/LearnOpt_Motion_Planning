import importlib.util

import uav_ac.planning.search as search
from uav_ac.planning.trajectory.aerial_manipulator_minco import search_adapter


def test_search_package_exports_only_generic_search_api():
    assert set(search.__all__) == {
        "AStarResult",
        "RRTConnectPlanningError",
        "RRTStar",
        "StateSpaceAdapter",
        "astar_search",
        "plan_rrt_connect",
    }
    assert not hasattr(search, "AStarGuide")
    assert not hasattr(search, "plan_aerial_astar_guide")
    assert hasattr(search_adapter, "AStarGuide")
    assert hasattr(search_adapter, "AerialManipulatorStateSpaceAdapter")
    assert importlib.util.find_spec("uav_ac.planning.search.A_star") is None
    assert importlib.util.find_spec("uav_ac.planning.search.RRT_connect") is None
