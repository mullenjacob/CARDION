"""Test scenarios for the mock camera: a recorded real face is moved, scaled, darkened, shaken or duplicated before it reaches the app,
so the positioning guidance can be tested without a person standing in front of a camera.

``apply_scenario(frame, "shift_x:0.3", n)`` -> frame. Specs (``name`` or ``name:value``):

``good`` unchanged | ``shift_x:f`` face moves right in the RAW camera image by f x frame width (negative = left) |
``shift_y:f`` moves down by f x frame height | ``scale:f`` face scaled about the centre (0.4 = small, 2.2 = large / cropped) |
``dark:f`` brightness multiplied by f | ``flicker:f`` brightness oscillates +-f at 2 Hz | ``sway:f`` slow side-to-side movement of amplitude f x frame width | ``shake:f`` random jitter of +-f x frame width every frame |
``two_faces`` a second, smaller copy of the face is placed in the corner | ``noface`` no face (noise) | ``turn:f`` perspective squeeze that
imitates a head turn | ``uneven:f`` left half darkened by factor f | ``disconnect`` the camera stops delivering frames.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

SCENARIO_FILE_ENV = "CARDION_MOCK_SCENARIO_FILE"       # a text file holding the current spec; re-read while running (used by browser tests)
SCENARIO_ENV = "CARDION_MOCK_SCENARIO"


def current_spec(default: str = "good") -> str:
    """The scenario requested through the environment (file first, then variable)."""
    f = os.environ.get(SCENARIO_FILE_ENV)
    if f:
        try:
            txt = Path(f).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except OSError:
            pass
    return os.environ.get(SCENARIO_ENV, default)


def _affine(frame: np.ndarray, m: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_scenario(frame: np.ndarray, spec: str, n: int = 0) -> np.ndarray | None:
    """Return the frame as the scenario would show it (``None`` = camera delivers nothing)."""
    name, _, val = spec.partition(":")
    f = float(val) if val else None
    h, w = frame.shape[:2]
    if name in ("good", ""):
        return frame
    if name == "disconnect":
        return None
    if name == "noface":
        return np.random.default_rng(n).integers(60, 110, frame.shape, dtype=np.uint8)
    if name == "shift_x":
        return _affine(frame, np.float32([[1, 0, (f or 0.0) * w], [0, 1, 0]]))
    if name == "shift_y":
        return _affine(frame, np.float32([[1, 0, 0], [0, 1, (f or 0.0) * h]]))
    if name == "scale":
        s = f or 1.0
        return _affine(frame, np.float32([[s, 0, w / 2 * (1 - s)], [0, s, h / 2 * (1 - s)]]))
    if name == "dark":
        return np.clip(frame.astype(np.float32) * (f if f is not None else 0.25), 0, 255).astype(np.uint8)
    if name == "flicker":
        k = 1.0 + (f if f is not None else 0.35) * np.sin(2 * np.pi * 2.0 * n / 30.0)
        return np.clip(frame.astype(np.float32) * k, 0, 255).astype(np.uint8)
    if name == "shake":
        rng = np.random.default_rng(n)
        dx, dy = rng.uniform(-1, 1, 2) * (f if f is not None else 0.06) * w
        return _affine(frame, np.float32([[1, 0, dx], [0, 1, dy]]))
    if name == "sway":                                     # slow side-to-side head movement (0.4 Hz)
        dx = (f if f is not None else 0.05) * w * np.sin(2 * np.pi * 0.4 * n / 30.0)
        return _affine(frame, np.float32([[1, 0, dx], [0, 1, 0]]))
    if name == "uneven":
        out = frame.astype(np.float32)
        out[:, : w // 2] *= (f if f is not None else 0.45)
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == "turn":
        k = f if f is not None else 0.35
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * k, h * k * 0.5], [w, 0], [w, h], [w * k, h * (1 - k * 0.5)]])
        return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if name == "two_faces":
        out = frame.copy()
        small = cv2.resize(frame, (int(w * 0.45), int(h * 0.45)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        out[h - sh:, w - sw:] = small
        return out
    raise ValueError(f"unknown mock scenario {spec!r}")


def reframe_matrix(frames: list[np.ndarray], cfg: dict, target_h: float = 0.45) -> np.ndarray:
    """Affine map (2x3) that re-frames a recording like a webcam shot of a person sitting at a normal distance: the face (found on a few
    frames) ends up centred and about ``target_h`` of the frame height. The research recordings show small, off-centre faces, so the
    framing tests start from this."""
    from ..capture import LandmarkSkinROI
    roi = LandmarkSkinROI(cfg)
    boxes = []
    for fr in frames:
        faces = roi.detect_all(fr)
        if faces:
            lo, hi = faces[0].min(axis=0), faces[0].max(axis=0)
            boxes.append([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[1] - lo[1]])
    roi.close()
    if not boxes:
        return np.float32([[1, 0, 0], [0, 1, 0]])
    cx, cy, fh = np.median(np.array(boxes), axis=0)
    ch = fh / target_h
    s = frames[0].shape[0] / ch
    h, w = frames[0].shape[:2]
    return np.float32([[s, 0, w / 2 - s * cx], [0, s, h / 2 - s * cy]])


def reframe(frame: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply :func:`reframe_matrix` to a frame."""
    return _affine(frame, m)
