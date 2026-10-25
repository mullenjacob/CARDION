"""Frame sources for the app: webcam, video file, and a built-in simulated face with a known pulse.

Every source yields ``(frame_rgb_uint8, t_seconds, roi_override_or_None)``; ``None`` frame means end of stream. Sources never raise
into the UI: failures are reported through :class:`SourceError` with a human-readable message.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def open_capture(index: int, backend):
    """``cv2.VideoCapture`` for a camera index. If ``CARDION_MOCK_CAMERA=<video file>`` is set, camera 0 replays that video as a live webcam
    (used by the automated tests and to try the camera code path without hardware); other indices do not exist."""
    mock = os.environ.get("CARDION_MOCK_CAMERA")
    if mock:
        from .cameratest import ClosedCapture, MockCapture
        return MockCapture(Path(mock)) if index == 0 else ClosedCapture()
    return cv2.VideoCapture(index, backend)


class SourceError(RuntimeError):
    """A source could not be opened or failed; the message is shown to the user."""


def list_cameras(max_index: int = 5, backend=None) -> list[dict]:
    """Probe camera indices ``0..max_index-1`` and return ``[{"index": i, "name": ..., "width": w, "height": h}]``.

    Each probe opens the device briefly (the camera light may flash). Indices that cannot deliver a frame are skipped.
    """
    backend = backend if backend is not None else (cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    found = []
    for i in range(max_index):
        cap = None
        try:
            cap = open_capture(i, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({"index": i, "name": f"Camera {i}", "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        except Exception:  # noqa: BLE001 - probing must never crash the app
            pass
        finally:
            if cap is not None:
                cap.release()
    return found


class CameraSource:
    """Live webcam (DirectShow on Windows). Timestamps are wall-clock seconds."""

    kind = "camera"

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = open_capture(index, backend)
        if not self.cap.isOpened():
            raise SourceError(f"Camera {index} could not be opened. Close other apps using the camera and check Windows "
                              "Settings > Privacy & security > Camera (allow desktop apps).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or fps)
        self.t0 = time.perf_counter()
        self.failures = 0
        self.label = f"Camera {index}"

    def read(self):
        """Next frame or ``(None, ...)`` after repeated failures."""
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            self.failures += 1
            if self.failures > 30:
                raise SourceError("The camera stopped delivering frames (unplugged or in use by another app).")
            time.sleep(0.03)
            return "retry", 0.0, None
        self.failures = 0
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.perf_counter() - self.t0, None

    def close(self) -> None:
        """Release the device."""
        self.cap.release()


class VideoSource:
    """Video file. Timestamps are ``frame_index / fps`` (virtual time), so analysis speed can differ from playback speed."""

    kind = "video"

    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise SourceError(f"Video file not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened() and not str(self.path).isascii():       # OpenCV on Windows cannot open some non-ASCII paths
            import shutil
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="cardion_")) / ("video" + self.path.suffix)
            shutil.copyfile(self.path, tmp)
            self.cap = cv2.VideoCapture(str(tmp))
        if not self.cap.isOpened():
            raise SourceError(f"Could not open {self.path.name}. Supported: common AVI/MP4/MOV/MKV files.")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = float(fps_override or (fps if 5.0 <= fps <= 120.0 else 30.0))
        self.n = 0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.label = self.path.name

    def read(self):
        """Next frame, or ``(None, ...)`` at the end of the file."""
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None, 0.0, None
        t = self.n / self.fps
        self.n += 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t, None

    def close(self) -> None:
        """Release the file."""
        self.cap.release()


class DemoSource:
    """Simulated face with a *known* pulse (so the app works with no camera and can be checked against ground truth).

    Frames are generated with the same model as the research code (``datasets.make_synthetic_clip``): skin shading, texture, slow
    head motion, illumination drift, sensor noise, and a pulsatile colour change of known rate. The clip loops.
    """

    kind = "demo"

    def __init__(self, cfg: dict, bpm: float = 72.0, seconds: float = 24.0, size: int = 240, noise: float = 1.5, seed: int = 7) -> None:
        from ..datasets import make_synthetic_clip
        sc = {**cfg, "datasets": {**cfg["datasets"], "synthetic": {**cfg["datasets"]["synthetic"], "size": size, "clip_seconds": seconds,
                                                                      "fps": 30, "noise_sigma_range": [noise, noise]}}}
        self.clip = make_synthetic_clip(1, sc, seed=seed, hr0=bpm)
        best = self.clip
        self.fps = 30.0
        self.true_bpm = float(best.meta["hr0_bpm"])
        self.n = 0
        h, w = best.face.shape[1:3]
        yy, xx = np.mgrid[0:h, 0:w]
        self.mask = ((((xx - w / 2) / (w * 0.29)) ** 2 + ((yy - h / 2) / (h * 0.37)) ** 2) <= 1.0).astype(np.uint8)
        self.box = (w / 2, h / 2, 0.9 * w)
        self.label = f"Simulated face ({self.true_bpm:.0f} BPM)"

    def read(self):
        """Next frame (looping)."""
        i = self.n % len(self.clip.face)
        t = self.n / self.fps
        self.n += 1
        return self.clip.face[i], t, (self.mask, self.box)

    def close(self) -> None:
        """Nothing to release."""
