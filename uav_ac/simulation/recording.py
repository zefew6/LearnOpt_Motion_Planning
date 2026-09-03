"""Small dependency-free MP4 writer for deterministic MuJoCo experiments."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np


class Mp4Recorder:
    """Stream RGB frames to FFmpeg without opening a MuJoCo viewer."""

    def __init__(self, output_path: str | Path, width: int, height: int, fps: float):
        if width < 1 or height < 1:
            raise ValueError("video dimensions must be positive")
        if fps <= 0.0 or not np.isfinite(fps):
            raise ValueError("video FPS must be positive and finite")

        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._process: subprocess.Popen[bytes] | None = None
        self.frame_count = 0

    def __enter__(self) -> "Mp4Recorder":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "recording requires FFmpeg; install it with your system package manager")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        """Append one uint8 RGB frame with the configured dimensions."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("the recorder is not open")

        frame = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f"frame must have shape {expected_shape}, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            stderr = self._process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"FFmpeg stopped while recording: {stderr}") from error
        self.frame_count += 1

    def close(self) -> None:
        if self._process is None:
            return

        process = self._process
        self._process = None
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr}")

    def __exit__(self, *_: object) -> None:
        self.close()


def default_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Return a stable camera matching the scene's MuJoCo visual defaults."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = model.stat.center
    camera.distance = 1.5 * model.stat.extent
    camera.azimuth = model.vis.global_.azimuth
    camera.elevation = model.vis.global_.elevation
    return camera


def render_offscreen_frame(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
) -> np.ndarray:
    """Render one RGB frame without opening a MuJoCo viewer."""
    old_offwidth = int(model.vis.global_.offwidth)
    old_offheight = int(model.vis.global_.offheight)
    model.vis.global_.offwidth = max(old_offwidth, width)
    model.vis.global_.offheight = max(old_offheight, height)
    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=camera)
        return renderer.render()
    finally:
        if renderer is not None:
            renderer.close()
        model.vis.global_.offwidth = old_offwidth
        model.vis.global_.offheight = old_offheight


def save_png(frame: np.ndarray, output_path: str | Path) -> None:
    """Save one RGB frame as a lossless PNG using the system FFmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "image export requires FFmpeg; install it with your system package manager")
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    result = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-i", "-",
            "-frames:v", "1", "-f", "image2", str(output_path),
        ],
        input=np.ascontiguousarray(frame).tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg failed to write {output_path}: {stderr}")


__all__ = ["Mp4Recorder", "default_camera", "render_offscreen_frame", "save_png"]
