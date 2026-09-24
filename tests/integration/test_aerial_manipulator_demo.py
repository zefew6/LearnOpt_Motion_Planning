from pathlib import Path

from uav_ac.main import load_config, run


def test_twenty_second_hover_and_slow_arm_motion_meets_demo_limits():
    config = load_config(Path("configs/aerial_manipulator_hover.yaml"))
    result = run(config)
    assert not result["collision"]
    assert result["stable_peak_position_error"] < 0.15
    assert result["stable_peak_tilt"] < 0.174533  # 10 degrees
    assert result["peak_joint_tracking_error"] < 0.1
