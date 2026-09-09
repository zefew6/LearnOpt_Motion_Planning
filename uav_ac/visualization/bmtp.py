"""Headless BMTP reports from serialized records and sampled NED paths only."""

from pathlib import Path
import textwrap
import warnings

import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


_COLORS = ("#0072B2", "#009E73", "#CC79A7", "#8C6D31")
_PROJECTION_NOTE = "Top-down overlap does not establish collision freedom; altitude matters."


def _figure(**kwargs):
    fig = Figure(layout="constrained", **kwargs)
    FigureCanvasAgg(fig)
    return fig


def _path(points):
    points = np.asarray(points, dtype=float).copy()
    points[:, 2] *= -1
    return points


def _box_faces(row):
    xmin, xmax, ymin, ymax, zmin, zmax = row
    lo, hi = -zmax, -zmin
    vertices = np.array([[xmin, ymin, lo], [xmax, ymin, lo],
                         [xmax, ymax, lo], [xmin, ymax, lo],
                         [xmin, ymin, hi], [xmax, ymin, hi],
                         [xmax, ymax, hi], [xmin, ymax, hi]])
    return vertices[np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                              [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])]


def _scene(ax, metadata, *, spatial=False, highlighted=()):
    for index, row in enumerate(metadata.get("obstacles", [])):
        color = "#E69F00" if index in highlighted else "#84929D"
        alpha = .5 if index in highlighted else .13
        if spatial:
            ax.add_collection3d(Poly3DCollection(
                _box_faces(row), facecolors=color, edgecolors=color,
                linewidths=.6, alpha=alpha))
        else:
            xmin, xmax, ymin, ymax, _, _ = row
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   facecolor=color, edgecolor=color, alpha=alpha))
    bounds = np.asarray(metadata["bounds"], dtype=float)
    ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1],
           xlabel="North x (m)", ylabel="East y (m)")
    if spatial:
        ax.set(zlim=(-bounds[1, 2], -bounds[0, 2]), zlabel="Altitude -z (m)")
        ax.set_box_aspect(np.maximum(bounds[1]-bounds[0], .01))
        ax.view_init(elev=26, azim=-58)
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)


def _line(ax, points, *, spatial=False, **kwargs):
    p = _path(points)
    return ax.plot(*p[:, :3 if spatial else 2].T, **kwargs)[0]


def _status(record):
    prefix = "OK" if record["success"] else "FAIL"
    return f"{prefix}: {record['status']}"


def _accepted(step, metadata):
    # Acceptance is recorded by the solver; sampled positions cannot certify C4.
    planner = metadata.get("settings", {}).get("planner", {})
    tolerance = planner.get("feasibility_tolerance", 1e-5)
    residuals = list(step.get("residuals", {}).values())
    return (step["iteration"] > 0 and step["accepted"] and not step["collisions"]
            and planner.get("continuity_order", 4) >= 4
            and all(np.isfinite(r) and r <= tolerance for r in residuals)
            and step["duration"] is not None and np.isfinite(step["duration"])
            and step["duration"] > 0)


def _summary(metadata, arrays):
    fig = _figure(figsize=(13, 6))
    axes = [fig.add_subplot(121), fig.add_subplot(122, projection="3d")]
    fixed = [r for r in metadata["records"] if r["kind"] == "fixed"]
    for ax, spatial in zip(axes, (False, True)):
        _scene(ax, metadata, spatial=spatial)
        ax.set_title("Top-down (altitude omitted)" if not spatial else "Physical NED boxes, altitude = -z")
        for index, record in enumerate(fixed):
            color = _COLORS[index % len(_COLORS)]
            label = f"Run {record['id']} initial"
            if not record["success"]:
                label += f" — {_status(record)}"
            _line(ax, record["initial_path"], spatial=spatial, color=color,
                  linestyle="--", linewidth=1.4, label=label)
            key = record["final_key"]
            if key is not None:
                _line(ax, arrays[key], spatial=spatial, color=color, linewidth=2,
                      label=f"Run {record['id']} final — {_status(record)}")
        if fixed:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("BMTP fixed initializations: dashed initial / solid final")
    fig.supxlabel(f"{_PROJECTION_NOTE}\nPhysical boxes shown; planning inflation = {metadata.get('inflation', 0):g} m.", fontsize=9)
    return fig


def _convergence(metadata):
    fig = _figure(figsize=(12, 5))
    axes = fig.subplots(1, 2)
    for index, record in enumerate(metadata["records"]):
        steps = [s for s in record["history"] if _accepted(s, metadata)]
        color = _COLORS[index % len(_COLORS)]
        label = f"{record['kind']} {record['id']} — {_status(record)}"
        if not steps:
            label += " (no accepted C4 update)"
        for ax, key in zip(axes, ("iteration", "elapsed_seconds")):
            ax.plot([s[key] for s in steps], [s["duration"] for s in steps],
                    marker="o", markersize=3, color=color,
                    linestyle="-" if record["kind"] == "fixed" else ":", label=label)
    for ax, xlabel in zip(axes, ("Trajectory update index", "Wall seconds since run start")):
        ax.set(xlabel=xlabel, ylabel="Accepted trajectory duration (s)")
        ax.grid(alpha=.2)
    if metadata["records"]:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
                   ncols=2, fontsize=7)
    fig.suptitle("Accepted C4-feasible, collision-free updates only (initial duration excluded)")
    return fig


def _outcomes(metadata):
    records = metadata["records"]
    fig = _figure(figsize=(max(10, .55*len(records)), 7))
    axes = fig.subplots(1, 2)
    positions = np.arange(len(records))
    labels = [f"{r['kind']} {r['id']}\n{_status(r)}" for r in records]
    for ax, metric, title in zip(axes, ("duration", "total_seconds"),
                               ("Final certified trajectory duration (s)", "Total wall time (s)")):
        for i, record in enumerate(records):
            value = record["duration"] if metric == "duration" else record["timings"].get(metric)
            if metric == "duration" and not record["success"]:
                value = None
            color = _COLORS[i % len(_COLORS)] if record["success"] else "#D55E00"
            if value is not None and np.isfinite(value):
                ax.bar(i, value, color=color, alpha=.8,
                       hatch="//" if record["kind"] == "perturbed" else None)
            else:
                ax.scatter([i], [.025], marker="x", color=color,
                           transform=ax.get_xaxis_transform())
                ax.text(i, .055, "FAIL" if not record["success"] else "N/A",
                        transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set_xticks(positions, labels, rotation=60, ha="right", fontsize=8)
        ax.set(title=title, ylabel="Seconds", ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
    failures = [f"{r['kind']} {r['id']} — {_status(r)}: {r['message']}"
                for r in records if not r["success"]]
    if failures:
        detail = "\n".join(textwrap.fill(s, width=130) for s in failures)
        fig.set_figheight(7 + .18*len(detail.splitlines()))
        fig.supxlabel(detail, fontsize=8, ha="left", x=.02)
    fig.suptitle("All runs — failures have no certified duration; hatched bars: perturbed")
    return fig


def _frame_steps(records):
    """Keep every normal update; cap long runs with explicitly disclosed thinning."""
    steps = sorted({s["iteration"] for r in records for s in r["history"]})
    if len(steps) <= 100:
        return steps or [0]
    terminal = {r["history"][-1]["iteration"] for r in records if r["history"]}
    terminal.add(steps[0])
    events = sorted({s["iteration"] for r in records for s in r["history"]
                     if s["collisions"] or s["new_tags"] or not s["accepted"]} - terminal)
    selected = set(terminal)
    for pool in (events, sorted(set(steps)-selected-set(events))):
        count = min(100-len(selected), len(pool))
        if count:
            selected.update(pool[i] for i in np.linspace(0, len(pool)-1, count, dtype=int))
    warnings.warn(f"Animation condensed from {len(steps)} steps to {len(selected)} frames; "
                  "omitted collision/rejected steps are counted on each panel; all run endings retained.",
                  UserWarning, stacklevel=2)
    return sorted(selected)


def _animation_frame(fig, axes, metadata, arrays, records, iteration, selected):
    for ax in axes:
        ax.clear()
    for index, ax in enumerate(axes):
        if index >= len(records):
            ax.set_axis_off()
            ax.text2D(.5, .5, "No fixed run recorded", transform=ax.transAxes, ha="center")
            continue
        record = records[index]
        color = _COLORS[index]
        history = [s for s in record["history"] if s["iteration"] <= iteration]
        step = history[-1] if history else None
        tagged = {pair[1] for pair in step["new_tags"]} if step else set()
        _scene(ax, metadata, spatial=True, highlighted=tagged)
        _line(ax, record["initial_path"], spatial=True, color="gray", linestyle="--",
              linewidth=1, label="Initial (not C4-certified)")
        accepted = [s for s in history if _accepted(s, metadata)]
        if accepted:
            best = min(accepted, key=lambda s: s["duration"])
            _line(ax, arrays[best["key"]], spatial=True, color=color, linewidth=3,
                  label="Best accepted")
        if step:
            collides = bool(step["collisions"])
            _line(ax, arrays[step["key"]], spatial=True,
                  color="red" if collides else color, linestyle=":", linewidth=2,
                  label="Current candidate")
            state = "COLLISION / rejected" if collides else (
                "accepted C4" if _accepted(step, metadata) else "rejected / not C4-certified")
            duration = step["duration"]
            duration_label = "N/A" if duration is None else f"{duration:.3g} s"
            title = f"Run {record['id']} | step {step['iteration']} | T={duration_label}\n{state}"
        else:
            title = f"Run {record['id']} | no candidate yet"
        exhausted = not record["history"] or iteration >= record["history"][-1]["iteration"]
        if exhausted:
            title += f"\n{_status(record)} — ended; holding last recorded state"
        omitted = [s for s in history if s["iteration"] not in selected]
        if omitted:
            rejected = sum(bool(s["collisions"]) or not s["accepted"] for s in omitted)
            title += f"\nCondensed: {len(omitted)} omitted steps ({rejected} collision/rejected)"
        ax.set_title(title, fontsize=8)
        ax.legend(loc="upper left", fontsize=6)
    fig.suptitle(f"BMTP recorded iteration {iteration} — no trajectory interpolation", fontsize=12)


def _animate(metadata, arrays, output_dir):
    records = [r for r in metadata["records"] if r["kind"] == "fixed"]
    if len(records) > 4:
        raise ValueError("iteration animation supports at most four fixed runs")
    frames = _frame_steps(records)
    selected = set(frames)
    fig = _figure(figsize=(11, 9))
    axes = [fig.add_subplot(2, 2, i+1, projection="3d") for i in range(4)]
    fig.supxlabel("Orange: newly tagged physical obstacle; red: recorded collision.\n"
                  "Ended runs hold their last recorded state while other runs continue.\n"
                  f"{len(frames)} actual-step frames; inflation = {metadata.get('inflation', 0):g} m (boxes are physical).",
                  fontsize=8)
    writer = PillowWriter(fps=3)
    try:
        with writer.saving(fig, str(output_dir / "iterations.gif"), dpi=80):
            for iteration in frames:
                _animation_frame(fig, axes, metadata, arrays, records, iteration, selected)
                writer.grab_frame()
    finally:
        fig.clear()


def plot_results(metadata: dict, arrays: dict[str, np.ndarray], output_dir: Path,
                 animation: bool = True) -> None:
    """Write summary PNG/SVG, convergence/outcomes PNG, and optionally a 3 fps GIF.

    The caller creates ``output_dir``. Paths/boxes remain unmodified in NED;
    display coordinates use north, east, altitude=-z. Feasibility comes from
    recorded acceptance/residuals, never from a sampled curve or its projection.
    Histories longer than 100 distinct iterations are visibly condensed, with
    terminal states retained and omitted collision/rejection counts disclosed.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Caller must create output directory: {output_dir}")
    for make, names in ((_summary, ("summary.png", "summary.svg")),
                        (_convergence, ("convergence.png",)), (_outcomes, ("outcomes.png",))):
        fig = make(metadata, arrays) if make is _summary else make(metadata)
        try:
            for name in names:
                fig.savefig(output_dir / name, dpi=160, facecolor="white")
        finally:
            fig.clear()
    if animation:
        _animate(metadata, arrays, output_dir)


__all__ = ["plot_results"]
