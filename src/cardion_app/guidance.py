"""Live positioning and quality guidance for the CARDION camera mode.

This module only *advises*; it never changes how the heart rate is computed. It turns what the face tracker already provides
(MediaPipe landmarks, the skin mask, the existing quality assessment) into short, stable instructions for the person in front of the
camera, plus a per-dimension quality breakdown.

Design rules
- Everything is measured, nothing is guessed: face position and size are fractions of the camera frame (a webcam has no calibrated
  distance, so CARDION never claims to know how far away you are), head turn comes from landmark geometry, motion / lighting come from
  the same numbers the reliability gate uses.
- Directions are from the *user's* point of view. The camera sees you as others do: a face at the right of the raw image is on the user's
  left, so the advice is "Move slightly right". The preview is shown mirrored (like a mirror), where that is simply "toward the middle".
- Advice follows signal quality, not cosmetics: while a reliable pulse is being measured, only serious framing problems are mentioned.
- Instructions are stable: exponential smoothing, separate enter/exit thresholds (hysteresis), a persistence time before a new
  instruction is shown and a minimum time it stays up.
- Good framing is never presented as proof of an accurate measurement: framing, tracking, lighting, motion, pulse signal, method
  agreement and amount of data are reported separately, from real observations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

NOSE, L_CHEEK, R_CHEEK, FOREHEAD, CHIN = 1, 234, 454, 10, 152     # MediaPipe face-mesh landmark indices
FACE_ASPECT = 0.75                                                  # typical face width / height

# --------------------------------------------------------------------------------------------------- messages
# code -> (short instruction shown on the preview, level, arrow, optional explanation shown beside the heart rate)
MESSAGES = {
    "no_face": ("Position your face in front of the camera.", "bad", "none", ""),
    "no_face_long": ("Position your face in front of the camera.", "bad", "none",
                     "If you are already in view, move a little closer and add light - very small faces are not detected."),
    "multi_face": ("Only one person should be in view.", "warn", "none",
                   "More than one face is visible. The blue solid box is the face being measured; dashed red boxes are not analysed. "
                   "The reading is paused until only one person remains."),
    "move_left": ("Move slightly left.", "warn", "left", ""),
    "move_right": ("Move slightly right.", "warn", "right", ""),
    "move_up": ("Move slightly up.", "warn", "up", ""),
    "move_down": ("Move slightly down.", "warn", "down", ""),
    "closer": ("Move a little closer.", "warn", "closer", "Your face fills only a small part of the picture, which makes the pulse harder to see."),
    "farther": ("Move a little farther away.", "warn", "farther", "Your face is very large or cropped by the edge of the picture."),
    "turn_front": ("Face the camera directly.", "warn", "none", ""),
    "center": ("Center your face in the camera.", "warn", "none", "Part of your face is outside the frame."),
    "hold_still": ("Too much movement. Hold still for a moment.", "warn", "none", ""),
    "dark": ("Your face is too dark. Increase the lighting in front of you.", "warn", "none", ""),
    "bright": ("Too much light on your face. Reduce glare or direct light.", "warn", "none", ""),
    "uneven_light": ("Try more even lighting.", "warn", "none", "One side of your face is much brighter than the other."),
    "unstable_light": ("Lighting is changing. Try steady indoor lighting.", "warn", "none", ""),
    "measuring": ("Great positioning. Measuring your pulse.", "ok", "none", ""),
    "collecting": ("Great positioning. Collecting enough data to estimate your heart rate.", "ok", "none", ""),
    "stabilizing": ("Signal is stabilizing. Stay still in steady light.", "info", "none", ""),
    "weak_long": ("The pulse signal is still weak.", "warn", "none", "Hold still, face the camera and try brighter, even light."),
    "disagree": ("The pulse estimates do not agree yet.", "info", "none", "Hold still in steady light."),
    "reliable": ("Reliable pulse detected.", "ok", "none", ""),
}
GEOMETRY_CODES =("center", "farther", "closer", "move_left", "move_right", "move_up", "move_down", "turn_front")


@dataclass
class GuidanceParams:
    """Thresholds. ``enter``/``exit`` pairs give hysteresis; ``hard`` limits are the only ones used while a reliable pulse is being measured."""
    smooth: float = 0.35                  # exponential smoothing of face geometry per landmark update (5 Hz)
    too_small: tuple = (0.20, 0.25)       # face height / frame height: advise "closer" below enter, release above exit
    too_large: tuple = (0.80, 0.74)
    off_x: tuple = (0.30, 0.20)           # |face centre - frame centre| / frame width. Sweep on real clips: moving the face this far changes the pulse estimate by < 1 BPM
    off_y: tuple = (0.30, 0.20)                      # (out/guidance_calibration_sweep_*.json) - so centring advice is only given when the face approaches the frame edge
    yaw: tuple = (0.20, 0.14)             # (nose offset from the mid-line of the face) / face width
    pitch: tuple = (0.12, 0.08)           # deviation of the nose height ratio from its frontal value
    pitch_frontal: float = 0.557
    uneven: tuple = (0.45, 0.30)          # |left - right| mean luminance / mean luminance inside the face
    outside: float = 0.03                 # fraction of landmarks outside the frame -> "center your face"
    edge: float = 0.01                    # face touching the frame edge (fraction of the frame)
    hard_small: float = 0.13              # while measuring reliably, only these still trigger advice
    hard_large: float = 0.92
    hard_off: float = 0.40
    hard_yaw: float = 0.34
    no_face_long_after: float = 5.0       # no face this long -> add the "move closer / add light" hint (very small faces are not detected)
    second_face_min: float = 0.30         # another face counts if at least this fraction of the main face's height
    multi_dwell: float = 0.8
    dwell: float = 1.2                    # a new instruction must persist this long (s) before it is shown
    dwell_urgent: float = 0.8             # no face / multiple faces
    dwell_ok: float = 0.8                 # returning to "all good"
    hold: float = 1.5                     # minimum time (s) an instruction stays up
    lost_after: float = 1.5               # seconds without a face before smoothing state is dropped
    weak_after: float = 12.0              # "signal weak" for this long -> show the longer tip


@dataclass
class Geometry:
    """Face position / size / pose, all relative to the camera frame (raw, un-mirrored image coordinates)."""
    cx: float
    cy: float
    w: float
    h: float
    yaw: float
    pitch: float
    outside: float
    touching: bool
    luma_lr: tuple[float, float] | None = None
    box: tuple[float, float, float, float] = (0, 0, 0, 0)


def geometry_from_landmarks(pts: np.ndarray, shape: tuple[int, int], luma_lr: tuple[float, float] | None = None) -> Geometry:
    """Measure a face from its (N, 2) pixel landmarks; ``shape`` is the frame (height, width)."""
    H, W = shape
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    fw = max(float(pts[R_CHEEK, 0] - pts[L_CHEEK, 0]), 1e-3)
    yaw = float((pts[NOSE, 0] - (pts[L_CHEEK, 0] + pts[R_CHEEK, 0]) / 2) / fw)
    fh = max(float(pts[CHIN, 1] - pts[FOREHEAD, 1]), 1e-3)
    pitch = float((pts[NOSE, 1] - pts[FOREHEAD, 1]) / fh)
    out = float(np.mean((pts[:, 0] < 0) | (pts[:, 0] > W) | (pts[:, 1] < 0) | (pts[:, 1] > H)))
    touching = bool(x0 < 0.01 * W or x1 > 0.99 * W or y0 < 0.01 * H or y1 > 0.99 * H)        # landmarks reach the frame edge
    return Geometry(cx=float((x0 + x1) / 2 / W), cy=float((y0 + y1) / 2 / H), w=float((x1 - x0) / W), h=float((y1 - y0) / H), yaw=yaw, pitch=pitch,
                    outside=out, touching=touching, luma_lr=luma_lr, box=(float(x0 / W), float(y0 / H), float(x1 / W), float(y1 / H)))


def side_luma(frame: np.ndarray, mask: np.ndarray, cx_px: float) -> tuple[float, float] | None:
    """Mean luminance of the skin mask left / right of the face centre (used for the uneven-lighting hint)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    px = frame[ys, xs].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    left = xs < cx_px
    if left.sum() < 50 or (~left).sum() < 50:
        return None
    return float(px[left].mean()), float(px[~left].mean())


def target_box(frame_aspect: float) -> tuple[float, float, float, float]:
    """Where the face should roughly be (normalised, raw image coordinates): centred, about 45% of the frame height."""
    h = 0.46
    w = h * FACE_ASPECT / frame_aspect
    return (0.5 - w / 2, 0.5 - h / 2, 0.5 + w / 2, 0.5 + h / 2)


class GuidanceTracker:
    """Stateful, time-stable guidance. Feed it face geometry (about 5 Hz) and the 1 Hz reliability assessment; read :meth:`snapshot`."""

    def __init__(self, params: GuidanceParams | None = None, frame_aspect: float = 4 / 3) -> None:
        self.p = params or GuidanceParams()
        self.aspect = frame_aspect
        self.reset()

    def reset(self) -> None:
        self.sm: dict[str, float] | None = None
        self.geom: Geometry | None = None
        self.active: set[str] = set()
        self.last_face_t: float | None = None
        self.t = 0.0
        self.multi: deque = deque(maxlen=40)           # (t, another face visible?)
        self.others: list[tuple] = []
        self.a_status = "collecting"
        self.a_t = -1e9
        self.env: dict = {}
        self.detail: dict = {}
        self.raw = "no_face"
        self.raw_since = 0.0
        self.shown = "no_face"
        self.shown_since = 0.0
        self.weak_since: float | None = None
        self.no_face_since: float | None = None
        self.reliable_once = False

    # ------------------------------------------------------------------ inputs
    def update(self, t: float, geom: Geometry | None, others: list[tuple] | None = None) -> None:
        """One face-tracker update. ``geom`` is None when no face was found; ``others`` are boxes of other (non-analysed) faces."""
        self.t = t
        self.others = list(others or [])
        self.multi.append((t, bool(self.others)))
        if geom is None:
            if self.no_face_since is None:
                self.no_face_since = t
            if self.last_face_t is not None and t - self.last_face_t > self.p.lost_after:
                self.sm, self.geom, self.active = None, None, set()
            self.geom = None if self.sm is None else self.geom
            self._tick(t, face=False)
            return
        self.last_face_t = t
        self.no_face_since = None
        self.geom = geom
        cur = {"cx": geom.cx, "cy": geom.cy, "h": geom.h, "yaw": abs(geom.yaw), "pitch": geom.pitch, "out": geom.outside}
        if geom.luma_lr:
            lm = max((geom.luma_lr[0] + geom.luma_lr[1]) / 2, 1.0)
            cur["uneven"] = abs(geom.luma_lr[0] - geom.luma_lr[1]) / lm
            cur["luma"] = lm
        a = self.p.smooth
        self.sm = cur if self.sm is None else {k: (1 - a) * self.sm.get(k, v) + a * v for k, v in cur.items()}
        self._tick(t, face=True)

    def set_assessment(self, status: str, env: dict, detail: dict, t: float) -> None:
        """The 1 Hz reliability assessment from the pipeline (status, environment numbers, estimator agreement)."""
        self.a_status, self.a_t, self.env, self.detail = status, t, dict(env), dict(detail)
        self.t = max(self.t, t)
        if status in ("weak_signal", "disagree"):
            self.weak_since = self.weak_since if self.weak_since is not None else t
        else:
            self.weak_since = None
        if status == "ok":
            self.reliable_once = True
        self._tick(t, face=self.geom is not None)

    # ------------------------------------------------------------------ decision
    def _flag(self, name: str, value: float, pair: tuple, above: bool = True) -> bool:
        """Hysteresis: switch on beyond ``pair[0]``, off again only once back past ``pair[1]``."""
        on = name in self.active
        beyond = value > (pair[1] if on else pair[0]) if above else value < (pair[1] if on else pair[0])
        (self.active.add if beyond else self.active.discard)(name)
        return beyond

    def _reliable(self, t: float) -> bool:
        return self.a_status == "ok" and t - self.a_t < 3.5

    def decide(self, t: float, face: bool) -> str:
        """The instruction that applies right now (before time-stability filtering)."""
        p = self.p
        if not face or self.sm is None:
            return "no_face_long" if self.no_face_since is not None and t - self.no_face_since >= p.no_face_long_after else "no_face"
        recent = [m for tt, m in self.multi if t - tt <= 2.0]
        if recent and np.mean(recent) > 0.3:
            return "multi_face"
        s = self.sm
        reliable = self._reliable(t)
        g = self.geom
        clipped = s["out"] > p.outside or (g is not None and g.touching and s["h"] < 0.9)
        # geometry problems (hard limits while reliable, soft limits otherwise)
        size_small = self._flag("small", s["h"], p.too_small, above=False) if not reliable else s["h"] < p.hard_small
        size_large = self._flag("large", s["h"], p.too_large) if not reliable else s["h"] > p.hard_large
        dx, dy = s["cx"] - 0.5, s["cy"] - 0.5
        off_x = self._flag("off_x", abs(dx), p.off_x) if not reliable else abs(dx) > p.hard_off
        off_y = self._flag("off_y", abs(dy), p.off_y) if not reliable else abs(dy) > p.hard_off
        turned = self._flag("yaw", s["yaw"], p.yaw) if not reliable else s["yaw"] > p.hard_yaw
        pitched = self._flag("pitch", abs(s["pitch"] - p.pitch_frontal), p.pitch) if not reliable else False
        if (clipped and s["h"] > 0.72) or size_large:
            return "farther"
        if off_x:
            return "move_right" if dx > 0 else "move_left"        # raw image: a face at the right of the image is on the user's left
        if off_y:
            return "move_down" if dy < 0 else "move_up"           # a face too high in the image -> move down
        if clipped and (not reliable or s["out"] > 2 * p.outside):
            return "center"
        if size_small:
            return "closer"
        if turned or pitched:
            return "turn_front"
        st = self.a_status if t - self.a_t < 3.5 else "collecting"
        if st == "motion":
            return "hold_still"
        luma = s.get("luma")
        if st == "dark" or (luma is not None and luma < 35):
            return "dark"
        if st == "bright" or (luma is not None and luma > 235):
            return "bright"
        if st == "unstable_light":
            return "unstable_light"
        if "uneven" in s and not reliable and self._flag("uneven", s["uneven"], p.uneven):
            return "uneven_light"
        if st == "ok":
            return "reliable"
        if st == "weak_signal" or st == "disagree":
            if self.weak_since is not None and t - self.weak_since >= p.weak_after and st == "weak_signal":
                return "weak_long"
            return "disagree" if st == "disagree" else "stabilizing"
        if st == "no_face":                                       # the gate saw too few tracked frames although a face is visible now
            return "collecting"
        if st == "multi_face":
            return "multi_face"
        return "collecting" if float(self.env.get("seconds", 0.0)) < float(self.detail.get("min_seconds", 8.0)) else "measuring"

    def _tick(self, t: float, face: bool) -> None:
        """Apply persistence and minimum-hold rules and update the instruction that is shown."""
        p = self.p
        raw = self.decide(t, face)
        if raw != self.raw:
            self.raw, self.raw_since = raw, t
        if raw == self.shown:
            return
        if raw == "reliable" or self.shown == "reliable":          # "reliable pulse" must mirror the reliability gate exactly: never linger, never lag
            self.shown, self.shown_since = raw, t
            return
        urgent = raw in ("no_face", "no_face_long", "multi_face") or self.shown in ("no_face", "no_face_long", "multi_face")
        good = raw in ("reliable", "measuring", "collecting")
        need = p.dwell_urgent if urgent else (p.dwell_ok if good else p.dwell)
        if t - self.raw_since >= need and t - self.shown_since >= (0.0 if urgent else p.hold):
            self.shown, self.shown_since = raw, t

    # ------------------------------------------------------------------ output
    def stage(self) -> str:
        """Where the user is in the measurement flow: positioning, collecting, stabilizing or reliable."""
        if self.shown in ("no_face", "no_face_long", "multi_face") or self.shown in GEOMETRY_CODES:
            return "positioning"
        if self.shown == "reliable":
            return "reliable"
        if self.a_status in ("weak_signal", "disagree"):
            return "stabilizing"
        if self.a_status in ("motion", "dark", "bright", "unstable_light"):
            return "adjusting"
        return "collecting"

    def snapshot(self) -> dict:
        """JSON-ready state for the UI overlay."""
        text, level, arrow, explain = MESSAGES[self.shown]
        if self.shown == "collecting":
            need = float(self.detail.get("min_seconds", 8.0))
            text = f"Great positioning. Collecting data ({min(float(self.env.get('seconds', 0.0)), need):.0f} of {need:.0f} s)."
        face = None
        if self.geom is not None and self.sm is not None:
            x0, y0, x1, y1 = self.geom.box
            face = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        tx0, ty0, tx1, ty1 = target_box(self.aspect)
        framing_ok = face is not None and self.shown not in GEOMETRY_CODES and self.shown not in ("no_face", "no_face_long", "multi_face")
        return {"enabled": True, "code": self.shown, "message": text, "explain": explain, "level": level, "arrow": arrow, "stage": self.stage(),
                "framing_ok": framing_ok, "face": face, "target": {"x0": tx0, "y0": ty0, "x1": tx1, "y1": ty1},
                "others": [{"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]} for b in self.others], "mirrored": True,
                "metrics": {k: round(float(v), 3) for k, v in (self.sm or {}).items()}}


def quality_dimensions(guide: dict | None, assess_status: str, env: dict, detail: dict, min_seconds: float, qp) -> list[dict]:
    """Seven separate quality checks, each from real observed numbers (state: good / warn / bad / pending). No invented scores."""
    seconds = float(env.get("seconds", 0.0))
    dims: list[dict] = []

    def add(key: str, label: str, state: str, text: str) -> None:
        dims.append({"key": key, "label": label, "state": state, "detail": text})

    if guide is None:
        add("framing", "Camera framing", "pending", "not assessed for this source")
    elif guide["face"] is None:
        add("framing", "Camera framing", "bad", "no face in view")
    elif guide["code"] in ("multi_face",):
        add("framing", "Camera framing", "bad", "more than one face in view")
    elif guide["code"] in GEOMETRY_CODES:
        add("framing", "Camera framing", "warn", MESSAGES[guide["code"]][0].rstrip("."))
    else:
        add("framing", "Camera framing", "good", "face well placed in the frame")
    ff = env.get("face_fraction")
    add("tracking", "Face tracking", "pending" if ff is None else ("good" if ff >= qp.min_face_fraction else "bad"),
        "—" if ff is None else f"face tracked in {ff * 100:.0f}% of recent frames (need {qp.min_face_fraction * 100:.0f}%)")
    lv, luma = env.get("luma_var"), env.get("luma")
    if luma is None or seconds < 2:
        add("lighting", "Lighting", "pending", "—")
    elif luma < qp.min_luma:
        add("lighting", "Lighting", "bad", "face too dark")
    elif luma > qp.max_luma:
        add("lighting", "Lighting", "bad", "face too bright")
    elif lv is not None and lv > qp.max_luma_var:
        add("lighting", "Lighting", "bad", "brightness is changing")
    else:
        add("lighting", "Lighting", "good", "adequate and steady")
    mo = env.get("motion")
    add("motion", "Movement", "pending" if mo is None or seconds < 2 else ("good" if mo <= qp.max_motion else "bad"),
        "—" if mo is None else ("still" if mo < qp.max_motion * 0.4 else "slight" if mo <= qp.max_motion else "too much"))
    strong, voters = detail.get("strong"), detail.get("voters")
    if seconds < min_seconds or strong is None:
        add("signal", "Pulse signal", "pending", "not evaluated yet")
    else:
        add("signal", "Pulse signal", "good" if strong >= 1 else "bad", f"clear pulse peak in {strong} of {voters} video methods")
    agree = detail.get("agree")
    if agree is None:
        add("agreement", "Method agreement", "pending", "not evaluated yet")
    else:
        add("agreement", "Method agreement", "good" if agree >= qp.min_agree else "bad", f"{agree} methods agree (need {qp.min_agree})")
    add("data", "Data collected", "good" if seconds >= min_seconds else "pending", f"{min(seconds, 99):.0f} s of {min_seconds:.0f} s needed")
    return dims
