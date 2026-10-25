"""Regression tests for the live positioning / quality guidance (src/cardion_app/guidance.py).

Synthetic-landmark tests check the logic precisely (directions from the USER's point of view, hysteresis, persistence, "do not nag while a
reliable pulse is measured"). The scenario tests push a real recorded face, moved / scaled / darkened / duplicated, through the real
pipeline (MediaPipe landmarks) and check the advice that comes out.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.cardion_app.guidance import (FOREHEAD, CHIN, L_CHEEK, NOSE, R_CHEEK, GuidanceParams, GuidanceTracker, geometry_from_landmarks,
                                      quality_dimensions)
from src.cardion_app.quality import QualityParams, assess

H, W = 480, 640
ENV_OK = {"seconds": 12.0, "face_fraction": 1.0, "motion": 0.005, "luma": 120.0, "luma_var": 0.01, "multi_face": 0.0}


def fake_face(cx=0.5, cy=0.5, h=0.45, yaw=0.0, pitch=0.557) -> np.ndarray:
    """478 landmarks of a face-shaped cloud: centre (cx, cy) and height h as fractions of the frame; yaw/pitch as the guidance measures them."""
    rng = np.random.default_rng(0)
    ph = h * H
    pw = 0.75 * ph
    pts = np.stack([cx * W + (rng.random(478) - 0.5) * pw, cy * H + (rng.random(478) - 0.5) * ph], axis=1).astype(np.float32)
    pts[L_CHEEK] = [cx * W - pw / 2, cy * H]
    pts[R_CHEEK] = [cx * W + pw / 2, cy * H]
    pts[FOREHEAD] = [cx * W, cy * H - ph / 2]
    pts[CHIN] = [cx * W, cy * H + ph / 2]
    pts[NOSE] = [cx * W + yaw * pw, cy * H - ph / 2 + pitch * ph]
    return pts


def geom(**kw):
    return geometry_from_landmarks(fake_face(**kw), (H, W))


def drive(tr: GuidanceTracker, seconds: float, g, status="collecting", t0=0.0, env=None, others=None, detail=None) -> float:
    """Feed the tracker at 5 Hz (face) and 1 Hz (assessment). ``g`` may be a Geometry, None, or a callable t -> Geometry."""
    t = t0
    while t < t0 + seconds:
        gg = g(t) if callable(g) else g
        tr.update(t, gg, others)
        if abs(t - round(t)) < 1e-6:
            tr.set_assessment(status, {**ENV_OK, **(env or {})}, {"voters": 3, "strong": 3, "agree": 3, "min_seconds": 8.0, **(detail or {})}, t)
        t = round(t + 0.2, 6)
    return t


# ------------------------------------------------------------------------------------------ directions (user's perspective)
@pytest.mark.parametrize("kw,expected", [
    (dict(cx=0.84, h=0.3), "Move slightly right."),        # face at the right of the raw image = on the user's left -> user moves right
    (dict(cx=0.16, h=0.3), "Move slightly left."),
    (dict(cy=0.17, h=0.3), "Move slightly down."),         # face too high in the frame -> move down
    (dict(cy=0.83, h=0.3), "Move slightly up."),
    (dict(h=0.13), "Move a little closer."),
    (dict(h=0.86), "Move a little farther away."),
    (dict(yaw=0.32), "Face the camera directly."),
])
def test_instruction_direction(kw, expected) -> None:
    tr = GuidanceTracker()
    drive(tr, 6.0, geom(**kw))
    assert tr.snapshot()["message"] == expected


def test_partially_outside_frame_says_center() -> None:
    tr = GuidanceTracker()
    pts = fake_face(cx=0.93, h=0.45)
    drive(tr, 6.0, geometry_from_landmarks(pts, (H, W)))
    assert tr.snapshot()["code"] in ("center", "move_right")      # either way the user is told to come back toward the middle
    pts2 = fake_face(cx=0.5, cy=0.5, h=0.45)
    pts2[:40, 0] = -20.0                                           # part of the face outside the frame
    tr2 = GuidanceTracker()
    drive(tr2, 6.0, geometry_from_landmarks(pts2, (H, W)))
    assert tr2.snapshot()["code"] == "center"


def test_no_face_and_good_positioning_messages() -> None:
    tr = GuidanceTracker()
    drive(tr, 3.0, None)
    assert tr.snapshot()["message"] == "Position your face in front of the camera."
    t = drive(tr, 4.0, geom(), t0=3.0)
    assert tr.snapshot()["code"] in ("measuring", "collecting"), tr.snapshot()
    drive(tr, 3.0, geom(), status="ok", t0=t)
    assert tr.snapshot()["message"] == "Reliable pulse detected."


def test_multiple_faces_are_reported_and_marked() -> None:
    tr = GuidanceTracker()
    other = [(0.7, 0.55, 0.95, 0.95)]
    drive(tr, 5.0, geom(), others=other)
    s = tr.snapshot()
    assert s["code"] == "multi_face" and "one person" in s["message"].lower() and "more than one face" in s["explain"].lower()
    assert len(s["others"]) == 1 and s["face"] is not None


# ------------------------------------------------------------------------------------------ stability
def test_no_flicker_near_a_threshold() -> None:
    """Face centre wobbling around the 'move' threshold must not make the instruction flip back and forth."""
    tr = GuidanceTracker()
    rng = np.random.default_rng(1)
    changes, last = 0, None
    t = 0.0
    while t < 40:
        cx = 0.5 + 0.27 + 0.05 * np.sin(t * 2.0) + rng.normal(0, 0.01)         # hovers around the enter (0.30) / exit (0.20) thresholds
        tr.update(t, geom(cx=cx, h=0.3))
        if abs(t - round(t)) < 1e-6:
            tr.set_assessment("collecting", ENV_OK, {"min_seconds": 8.0}, t)
        code = tr.snapshot()["code"]
        changes += code != last
        last = code
        t = round(t + 0.2, 6)
    assert changes <= 4, changes


def test_small_jitter_never_triggers_advice() -> None:
    tr = GuidanceTracker()
    rng = np.random.default_rng(2)
    codes = set()
    t = 0.0
    while t < 30:
        tr.update(t, geom(cx=0.5 + rng.normal(0, 0.015), cy=0.5 + rng.normal(0, 0.015), h=0.45 + rng.normal(0, 0.01)))
        if abs(t - round(t)) < 1e-6:
            tr.set_assessment("collecting" if t < 9 else "weak_signal", ENV_OK, {"min_seconds": 8.0}, t)
        codes.add(tr.snapshot()["code"])
        t = round(t + 0.2, 6)
    assert not codes & set(("move_left", "move_right", "move_up", "move_down", "closer", "farther", "turn_front", "center")), codes


def test_instruction_needs_persistence() -> None:
    tr = GuidanceTracker()
    drive(tr, 4.0, geom())
    t = 4.0
    for _ in range(3):                                      # 0.6 s off-centre: shorter than the persistence time
        tr.update(t, geom(cx=0.85, h=0.3))
        t = round(t + 0.2, 6)
    tr.update(t, geom())
    assert tr.snapshot()["code"] not in ("move_right", "move_left")
    t = drive(tr, 3.0, geom(cx=0.85, h=0.3), t0=t)
    assert tr.snapshot()["code"] == "move_right"


def test_returns_to_good_after_repositioning() -> None:
    tr = GuidanceTracker()
    t = drive(tr, 4.0, geom(cx=0.85, h=0.3))
    assert tr.snapshot()["code"] == "move_right"
    drive(tr, 5.0, geom(cx=0.5), t0=t)
    assert tr.snapshot()["code"] in ("measuring", "collecting")


def test_no_nagging_while_a_reliable_pulse_is_measured() -> None:
    """Slightly off-centre and small, but the pulse is reliable: no movement instruction. A severe problem still gets one."""
    tr = GuidanceTracker()
    t = drive(tr, 6.0, geom(cx=0.72, h=0.17), status="ok")
    assert tr.snapshot()["code"] == "reliable"
    drive(tr, 6.0, geom(cx=0.93), status="ok", t0=t)
    assert tr.snapshot()["code"] in ("move_right", "center")


def test_lighting_and_motion_messages() -> None:
    for status, text in (("motion", "Too much movement. Hold still for a moment."), ("dark", "Your face is too dark. Increase the lighting in front of you."),
                         ("unstable_light", "Lighting is changing. Try steady indoor lighting.")):
        tr = GuidanceTracker()
        drive(tr, 6.0, geom(), status=status)
        assert tr.snapshot()["message"] == text, status


def test_signal_messages_are_distinct_from_framing() -> None:
    tr = GuidanceTracker()
    t = drive(tr, 6.0, geom(), status="weak_signal")
    assert tr.snapshot()["code"] == "stabilizing" and tr.snapshot()["stage"] == "stabilizing"
    drive(tr, 14.0, geom(), status="weak_signal", t0=t)
    assert tr.snapshot()["code"] == "weak_long"
    tr2 = GuidanceTracker()
    drive(tr2, 3.0, geom(), env={"seconds": 3.0})
    assert tr2.snapshot()["code"] == "collecting" and "of 8 s" in tr2.snapshot()["message"]
    tr3 = GuidanceTracker()
    drive(tr3, 8.0, None)
    assert tr3.snapshot()["code"] == "no_face_long" and "move a little closer" in tr3.snapshot()["explain"]


def test_quality_dimensions_use_observations_only() -> None:
    qp = QualityParams()
    dims = quality_dimensions(None, "collecting", {"seconds": 3.0}, {}, 8.0, qp)
    assert [d["key"] for d in dims] == ["framing", "tracking", "lighting", "motion", "signal", "agreement", "data"]
    assert next(d for d in dims if d["key"] == "data")["state"] == "pending"
    tr = GuidanceTracker()
    drive(tr, 6.0, geom(cx=0.85, h=0.3))
    dims = quality_dimensions(tr.snapshot(), "ok", ENV_OK, {"voters": 3, "strong": 3, "agree": 3}, 8.0, qp)
    states = {d["key"]: d["state"] for d in dims}
    assert states["framing"] == "warn" and states["signal"] == "good" and states["data"] == "good"      # good signal does not hide a framing issue


# ------------------------------------------------------------------------------------------ safeguards
def test_two_faces_block_a_reading() -> None:
    est = {k: {"bpm": 72.0, "peak_ratio": 9.0} for k in ("rgb_green", "rgb_pos", "rgb_chrom")}
    a = assess(est, {**ENV_OK, "multi_face": 0.8}, QualityParams())
    assert a.status == "multi_face" and a.bpm is None
    assert assess(est, ENV_OK, QualityParams()).status == "ok"                  # unchanged for one face


def test_guidance_params_load_from_config_lists() -> None:
    p = GuidanceParams(**{"off_x": (0.3, 0.2)})
    assert p.off_x == (0.3, 0.2)


# ------------------------------------------------------------------------------------------ real face through the real pipeline
def _run_scenario(spec: str, seconds: float = 7.0):
    import cv2
    from src.cardion_app.mockscenarios import apply_scenario, reframe, reframe_matrix
    from src.cardion_app.pipeline import Pipeline
    from src.common import load_config
    import glob
    cfg = load_config()
    vids = sorted(glob.glob(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data/raw/mcd_rppg/video/*.avi")), key=lambda p: __import__("os").path.getsize(p))
    if not vids:
        pytest.skip("no sample face video")
    cap = cv2.VideoCapture(vids[0])
    frames = []
    while len(frames) < int(seconds * 30):
        ok, bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.cvtColor(cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    m = reframe_matrix(frames[:24], cfg)
    pl = Pipeline(cfg, models=None, use_landmarks=True, guidance=True)
    pl.set_events(False)
    try:
        for i, fr in enumerate(frames):
            out = apply_scenario(reframe(fr, m), spec, i)
            if out is None:
                continue
            pl.push(out, i / 30.0)
            if i % 30 == 29 and pl.seconds() > 2:
                pl.estimate()
    finally:
        pl.close()                      # explicit: MediaPipe can deadlock when the garbage collector closes a landmarker
    return pl


@pytest.mark.parametrize("spec,codes", [
    ("good", ("measuring", "collecting", "reliable", "stabilizing", "disagree", "uneven_light", "weak_long")),
    ("shift_x:0.33", ("move_right",)),
    ("shift_x:-0.33", ("move_left",)),
    ("shift_y:0.32", ("move_up",)),
    ("shift_y:-0.32", ("move_down",)),
    ("scale:0.38", ("closer",)),
    ("scale:1.9", ("farther",)),
    ("two_faces", ("multi_face",)),
    ("noface", ("no_face", "no_face_long")),
])
def test_real_face_scenarios(spec, codes) -> None:
    pl = _run_scenario(spec)
    snap = pl.guide.snapshot()
    assert snap["code"] in codes, (spec, snap["code"], snap["metrics"])
    if spec == "two_faces":
        assert pl.last_assessment.bpm is None and pl.last_assessment.status == "multi_face"
    if spec == "good":
        assert snap["code"] not in ("move_left", "move_right", "move_up", "move_down", "closer", "farther", "center", "turn_front", "multi_face", "no_face")


def test_real_face_dark_and_shaking() -> None:
    assert _run_scenario("dark:0.18", 11.0).guide.snapshot()["code"] == "dark"
    assert _run_scenario("shake:0.06", 11.0).guide.snapshot()["code"] == "hold_still"
    assert _run_scenario("flicker:0.4", seconds=14.0).guide.snapshot()["code"] == "unstable_light"


def test_landmarks_unchanged_by_second_face_support() -> None:
    """num_faces=2 must not change what a single-face frame yields (the estimation path depends on it)."""
    import glob
    import cv2
    from src.capture import LandmarkSkinROI
    from src.common import load_config
    vids = sorted(glob.glob("data/raw/mcd_rppg/video/*.avi"))[:1]
    if not vids:
        pytest.skip("no sample face video")
    cap = cv2.VideoCapture(vids[0])
    cap.read()
    ok, bgr = cap.read()
    fr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = LandmarkSkinROI(load_config())
    faces = roi.detect_all(fr)
    assert len(faces) == 1 and np.array_equal(faces[0], roi.landmarks(fr))
    roi.close()


def test_reliable_message_never_outlives_the_reliability_gate() -> None:
    """Regression (found in the browser): 'Reliable pulse detected.' was still shown for ~1-2 s after the gate had withdrawn the BPM."""
    tr = GuidanceTracker()
    t = drive(tr, 5.0, geom(), status="ok")
    assert tr.snapshot()["code"] == "reliable"
    tr.set_assessment("weak_signal", ENV_OK, {"min_seconds": 8.0}, t)
    assert tr.snapshot()["code"] != "reliable" and tr.snapshot()["stage"] != "reliable"
    tr.set_assessment("ok", ENV_OK, {"min_seconds": 8.0}, t + 1.0)
    assert tr.snapshot()["code"] == "reliable"


def test_state_json_is_serialisable_with_two_faces() -> None:
    """Regression (found in the browser): boxes of other faces were numpy floats and /api/state returned an error."""
    import json
    from src.cardion_app.pipeline import Pipeline
    from src.cardion_app.server import _json_default
    from src.common import load_config
    pl = Pipeline(load_config(), models=None, use_landmarks=False, guidance=False)
    faces = [fake_face(cx=0.35), fake_face(cx=0.78, h=0.4)]
    pl.box = np.array([0.35 * W, 0.5 * H, 200.0])
    pl.last_face_frame = 0
    main, others = pl._pick_face(faces, (H, W))
    json.dumps({"others": others}, default=None)                                         # plain floats: no helper needed
    json.dumps({"x": np.float32(1.5), "y": np.arange(3)}, default=_json_default)
    assert main is not None and len(others) == 1


def test_real_face_swaying_is_hold_still() -> None:
    assert _run_scenario("sway:0.05", 11.0).guide.snapshot()["code"] == "hold_still"
