"""Webcam / video capture and face ROI extraction (MediaPipe FaceLandmarker, Tasks API).

The skin ROI is the face-oval polygon minus eyes, eyebrows and lips. Landmarks are refreshed every
``face.redetect_every`` frames; in between the last mask is held, which is cheap and stable for
the roughly stationary subjects in rPPG datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .common import add_common_args, repo_path, resolve_cfg, setup_logging

log = logging.getLogger("cardion.capture")


def ensure_model(cfg: dict) -> Path:
    """Return the FaceLandmarker model path, downloading it (with sha256 log) if absent."""
    path = repo_path(cfg["face"]["landmarker_model"])
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["face"]["landmarker_url"]
    log.info("downloading face landmarker model from %s", url)
    urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info("saved %s (%.2f MB) sha256=%s", path, path.stat().st_size / 1e6, digest)
    return path


def _ordered_loop(connections: list) -> list[int]:
    """Turn an unordered edge list into an ordered vertex loop by following edges."""
    nxt = {c.start: c.end for c in connections}
    start = connections[0].start
    loop, cur = [start], nxt[start]
    while cur != start and len(loop) <= len(nxt):
        loop.append(cur)
        cur = nxt.get(cur, start)
    return loop


class LandmarkSkinROI:
    """Compute a skin mask for each frame from MediaPipe face landmarks."""

    def __init__(self, cfg: dict) -> None:
        from mediapipe.tasks.python import BaseOptions, vision
        self._vision = vision
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=ensure_model(cfg).read_bytes()),   # buffer, not path: MediaPipe cannot open non-ASCII Windows paths
            running_mode=vision.RunningMode.IMAGE, num_faces=2)      # a second face is only used to warn "one person at a time"
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        conn = vision.FaceLandmarksConnections
        self.oval = _ordered_loop(conn.FACE_LANDMARKS_FACE_OVAL)
        self.holes = [_ordered_loop(c) for c in (
            conn.FACE_LANDMARKS_LEFT_EYE, conn.FACE_LANDMARKS_RIGHT_EYE,
            conn.FACE_LANDMARKS_LEFT_EYEBROW, conn.FACE_LANDMARKS_RIGHT_EYEBROW, conn.FACE_LANDMARKS_LIPS)]
        self.every = int(cfg["face"]["redetect_every"])
        self._count = 0
        self._last: np.ndarray | None = None

    def close(self) -> None:
        """Release the MediaPipe graph now. (Letting the garbage collector do it can deadlock MediaPipe, so owners call this explicitly.)"""
        lm, self.landmarker = getattr(self, "landmarker", None), None
        if lm is not None:
            lm.close()

    def detect_all(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Return a (478, 2) pixel-landmark array for every detected face (at most two), best first; empty list if none."""
        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self.landmarker.detect(img)
        h, w = frame_rgb.shape[:2]
        return [np.array([[p.x * w, p.y * h] for p in face], dtype=np.float32) for face in res.face_landmarks]

    def landmarks(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Return (478, 2) pixel landmarks of the first face or None if no face is found."""
        faces = self.detect_all(frame_rgb)
        return faces[0] if faces else None

    def mask_from_landmarks(self, pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Rasterise the skin polygon (oval minus holes) to a uint8 mask."""
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [pts[self.oval].astype(np.int32)], 1)
        for hole in self.holes:
            cv2.fillPoly(mask, [pts[hole].astype(np.int32)], 0)
        return mask

    def mask(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Skin mask for this frame (refreshes landmarks every N frames, else holds the last one)."""
        if self._last is None or self._count % self.every == 0:
            pts = self.landmarks(frame_rgb)
            if pts is not None:
                self._last = self.mask_from_landmarks(pts, frame_rgb.shape[:2])
        self._count += 1
        return self._last

    def close(self) -> None:
        """Release the landmarker."""
        self.landmarker.close()


def masked_mean_rgb(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean RGB of the pixels under ``mask`` (float64, shape (3,))."""
    sel = frame_rgb[mask.astype(bool)]
    return sel.mean(axis=0) if len(sel) else np.full(3, np.nan)


def video_frames(path: str | Path) -> tuple[Iterator[np.ndarray], float, int]:
    """Open a video file; return (RGB frame iterator, fps, frame_count)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def gen() -> Iterator[np.ndarray]:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()
    return gen(), float(fps), n


class WebcamCapture:
    """Thin wrapper over cv2.VideoCapture for a live camera (DirectShow on Windows)."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise IOError(f"cannot open webcam index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray | None:
        """Return the next RGB frame or None."""
        ok, bgr = self.cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def close(self) -> None:
        """Release the camera."""
        self.cap.release()


def extract_rgb_trace(path: str | Path, cfg: dict, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Run the skin ROI over a video file and return ((T,3) mean-RGB trace, fps)."""
    frames, fps, n = video_frames(path)
    roi = LandmarkSkinROI(cfg)
    trace: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        if max_frames and i >= max_frames:
            break
        m = roi.mask(fr)
        trace.append(masked_mean_rgb(fr, m) if m is not None else np.full(3, np.nan))
        if i % 300 == 0:
            log.info("%s: frame %d/%d", Path(path).name, i, n)
    roi.close()
    arr = np.asarray(trace)
    bad = np.isnan(arr).any(axis=1)
    if bad.any():
        log.warning("%d/%d frames without a face; interpolating", bad.sum(), len(arr))
        idx = np.arange(len(arr))
        for c in range(3):
            arr[bad, c] = np.interp(idx[bad], idx[~bad], arr[~bad, c])
    return arr, fps


def main() -> None:
    """CLI: print the skin-ROI mean for a still image (smoke test for the face pipeline)."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--image", help="RGB image file to test the face ROI on (default: skimage astronaut)")
    ap.add_argument("--save", help="optional path for an overlay PNG")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    if args.image:
        frame = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    else:
        from skimage import data
        frame = data.astronaut()
    roi = LandmarkSkinROI(cfg)
    m = roi.mask(frame)
    if m is None:
        raise SystemExit("no face found")
    log.info("frame %s, skin pixels %d (%.1f%% of frame), mean RGB %s", frame.shape, int(m.sum()),
             100 * m.mean(), np.round(masked_mean_rgb(frame, m), 2))
    if args.save:
        over = frame.copy()
        over[m.astype(bool)] = (0.5 * over[m.astype(bool)] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(args.save, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        log.info("wrote %s", args.save)


if __name__ == "__main__":
    main()
