"""Record-only plot regressions; no planner or simulator execution required."""

from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from uav_ac.visualization import bmtp


def _records():
    initial = [[0., 0., -1.], [2., 1., -2.], [4., 4., -1.]]
    arrays = {"bad": np.array(initial), "good": np.array(initial) + [0, 0, -1]}
    steps = [dict(iteration=0, accepted=True, collisions=[], new_tags=[],
                  duration=99., elapsed_seconds=0., residuals={"continuity": 1.}, key="bad"),
             dict(iteration=1, accepted=False, collisions=[[0, 0]], new_tags=[[0, 0]],
                  duration=8., elapsed_seconds=.1, residuals={"continuity": 0.}, key="bad"),
             dict(iteration=2, accepted=True, collisions=[], new_tags=[],
                  duration=6., elapsed_seconds=.3, residuals={"continuity": 0.}, key="good")]
    good = dict(id=0, kind="fixed", initial_path=initial, status="converged", message="done",
                success=True, converged=True, duration=6., initial_length=6., final_length=5.,
                timings={"total_seconds": .4}, history=steps, final_key="good")
    failed = dict(id=1, kind="fixed", initial_path=initial, status="solver_failure",
                  message="synthetic infeasible solve", success=False, converged=False,
                  duration=None, initial_length=6., final_length=None,
                  timings={"total_seconds": .2}, history=[deepcopy(steps[1])], final_key=None)
    metadata = dict(records=[good, failed], obstacles=[[1, 2, 1, 2, -3, -1]],
                    inflation=.2, bounds=[[0, 0, -5], [5, 5, 0]],
                    settings={"planner": {"feasibility_tolerance": 1e-5, "continuity_order": 4}})
    return metadata, arrays


@pytest.mark.parametrize("count", [1, 2])
def test_static_exports_preserve_records(tmp_path, count):
    metadata, arrays = _records()
    metadata["records"] = metadata["records"][-count:]
    before = deepcopy(metadata)
    before_arrays = {k: v.copy() for k, v in arrays.items()}
    assert bmtp.plot_results(metadata, arrays, tmp_path, animation=False) is None
    assert {p.name for p in tmp_path.iterdir()} == {
        "summary.png", "summary.svg", "convergence.png", "outcomes.png"}
    for name in ("summary.png", "convergence.png", "outcomes.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()
    svg = (tmp_path / "summary.svg").read_text()
    assert "FAIL: solver_failure" in svg
    assert "altitude matters" in svg
    assert metadata == before
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before_arrays[key])


def test_summary_altitude_physical_boxes_and_matched_colors():
    metadata, arrays = _records()
    fig = bmtp._summary(metadata, arrays)
    try:
        top, spatial = fig.axes
        assert top.lines[0].get_linestyle() == "--"
        assert top.lines[1].get_linestyle() == "-"
        assert top.lines[0].get_color() == top.lines[1].get_color()
        np.testing.assert_array_equal(spatial.lines[1].get_data_3d()[2], -arrays["good"][:, 2])
        assert spatial.get_zlim() == (0, 5)
        faces = bmtp._box_faces(metadata["obstacles"][0])
        assert faces[:, :, 2].min() == 1
        assert faces[:, :, 2].max() == 3
        assert top.patches[0].get_x() == 1  # No second inflation of physical boxes.
        assert "FAIL: solver_failure" in top.lines[-1].get_label()
    finally:
        fig.clear()


def test_convergence_excludes_initial_rejected_and_invalid_c4():
    metadata, _ = _records()
    invalid = deepcopy(metadata["records"][0]["history"][-1])
    invalid.update(iteration=3, duration=4., residuals={"continuity": .1})
    metadata["records"][0]["history"].append(invalid)
    fig = bmtp._convergence(metadata)
    try:
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [2])
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_xdata(), [.3])
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [6.])
        assert not len(fig.axes[0].lines[1].get_ydata())
        assert "FAIL" in fig.axes[0].lines[1].get_label()
    finally:
        fig.clear()


def test_outcomes_include_perturbed_failure_without_fake_duration():
    metadata, _ = _records()
    metadata["records"][1]["kind"] = "perturbed"
    fig = bmtp._outcomes(metadata)
    try:
        duration, wall = fig.axes
        assert [p.get_height() for p in duration.patches] == [6.]
        assert [p.get_height() for p in wall.patches] == [.4, .2]
        assert wall.patches[1].get_hatch() == "//"
        assert any("perturbed 1" in t.get_text() for t in wall.get_xticklabels())
        assert any("synthetic infeasible solve" in t.get_text() for t in fig.texts)
    finally:
        fig.clear()


def test_tiny_animation_uses_actual_candidates_and_holds_failure(tmp_path, monkeypatch):
    metadata, arrays = _records()
    metadata["records"][0]["history"] = metadata["records"][0]["history"][1:]
    seen = []
    original = bmtp._animation_frame

    def inspect(fig, axes, metadata, arrays, records, iteration, selected):
        original(fig, axes, metadata, arrays, records, iteration, selected)
        seen.append(iteration)
        candidate = next(line for line in axes[0].lines if line.get_label() == "Current candidate")
        expected = arrays["bad" if iteration == 1 else "good"]
        np.testing.assert_array_equal(np.array(candidate.get_data_3d()).T, bmtp._path(expected))
        if iteration == 1:
            assert candidate.get_color() == "red"
            assert "COLLISION" in axes[0].get_title()
            assert not any(line.get_label() == "Best accepted" for line in axes[0].lines)
        else:
            assert any(line.get_label() == "Best accepted" for line in axes[0].lines)
        assert "FAIL: solver_failure" in axes[1].get_title()
        assert "holding last recorded state" in axes[1].get_title()
        assert len(axes) == 4

    monkeypatch.setattr(bmtp, "_animation_frame", inspect)
    bmtp.plot_results(metadata, arrays, tmp_path, animation=True)
    assert seen == [1, 2]
    with Image.open(tmp_path / "iterations.gif") as gif:
        assert gif.n_frames == 2
        assert gif.info["duration"] == pytest.approx(1000/3, abs=10)
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()


def test_long_animation_retains_endings_and_discloses_omissions():
    metadata, arrays = _records()
    first, failed = metadata["records"]
    template = first["history"][-1]
    first["history"] = [dict(template, iteration=i) for i in range(1, 121)]
    failed["history"][0]["iteration"] = 117
    with pytest.warns(UserWarning, match="condensed from 120"):
        frames = bmtp._frame_steps(metadata["records"])
    assert len(frames) <= 100
    assert {1, 117, 120}.issubset(frames)
    fig = bmtp._figure(figsize=(11, 9))
    axes = [fig.add_subplot(2, 2, i+1, projection="3d") for i in range(4)]
    try:
        bmtp._animation_frame(fig, axes, metadata, arrays, metadata["records"], 120, set(frames))
        assert "20 omitted steps" in axes[0].get_title()
        assert "FAIL: solver_failure" in axes[1].get_title()
    finally:
        fig.clear()
    first["history"] = first["history"][:50]
    assert bmtp._frame_steps([first]) == list(range(1, 51))


def test_output_directory_is_created_by_caller(tmp_path):
    metadata, arrays = _records()
    missing = tmp_path / "not_created"
    with pytest.raises(FileNotFoundError, match="Caller must create"):
        bmtp.plot_results(metadata, arrays, missing, animation=False)
    assert not missing.exists()
