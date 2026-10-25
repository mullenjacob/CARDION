"""Automatic camera test: detect webcams, open them, read frames, find the face, run the heart-rate pipeline, and report in plain words.

    python -m src.cardion_app --cameratest                  # every webcam found (opens each for ~25 s; sit still, face the camera)
    python -m src.cardion_app --cameratest --mock           # no hardware: a recorded real-face video is fed through the *camera code path*
    python -m src.cardion_app --cameratest --mock --video some.avi

Writes ``out/camera_test_<time>.json`` / ``.md`` (add ``--snapshot`` to also keep one image of the first frame, only on this computer). Missing cameras,
denied permission and in-use devices are reported as warnings with what to do about them - never as a crash. In ``--mock`` mode only
the software path is tested; it says so, because it proves nothing about the real hardware.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from ..common import REPO_ROOT, save_json, timestamp
from .pipeline import Pipeline
from .sources import CameraSource, SourceError, list_cameras

MIN_FPS = 8.0
REAL_VIDEOCAPTURE = cv2.VideoCapture      # kept so the mock can open files while cv2.VideoCapture is patched


def windows_camera_setting() -> str:
    """Windows' camera privacy switch for this user: ``allow``, ``deny`` or ``unknown`` (never changed by CARDION)."""
    if not sys.platform.startswith("win"):
        return "unknown"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam") as k:
            return str(winreg.QueryValueEx(k, "Value")[0]).lower()
    except OSError:
        return "unknown"


class MockCapture:
    """Stands in for ``cv2.VideoCapture``: replays a video as if it were a live 30 fps webcam (real-time pacing, 640x480, looped)."""

    def __init__(self, path: Path, *_args) -> None:
        self.cap = REAL_VIDEOCAPTURE(str(path))
        self.fps_in = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.t0 = time.perf_counter()
        self.n = 0
        self.reframe_m = None
        if os.environ.get("CARDION_MOCK_SCENARIO") or os.environ.get("CARDION_MOCK_SCENARIO_FILE"):     # scenario tests start from a normally framed face
            from ..common import load_config
            from .mockscenarios import reframe_matrix
            probe = []
            for _ in range(24):
                ok, fr = self.cap.read()
                if ok:
                    probe.append(cv2.cvtColor(cv2.resize(fr, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            self.reframe_m = reframe_matrix(probe, load_config())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.t0 = time.perf_counter()                       # start the 30 fps clock only now: setting up the re-framing takes seconds and must not be "caught up" in a burst

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API name
        return self.cap.isOpened()

    def set(self, *_a) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        wait = self.t0 + self.n / 30.0 - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return False, None
        self.n += 1
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        if os.environ.get("CARDION_MOCK_SCENARIO") or os.environ.get("CARDION_MOCK_SCENARIO_FILE"):     # test scenarios (shifted / small / dark / two faces ...)
            from .mockscenarios import apply_scenario, current_spec, reframe
            frame = reframe(frame, self.reframe_m) if self.reframe_m is not None else frame
            if self.n % 10 == 1:
                self.spec = current_spec()
            out = apply_scenario(frame, getattr(self, "spec", "good"), self.n)
            return (False, None) if out is None else (True, out)
        return True, frame

    def release(self) -> None:
        self.cap.release()


class ClosedCapture:
    """A camera index that does not exist (mock mode exposes exactly one camera)."""

    def isOpened(self) -> bool:  # noqa: N802
        return False

    def release(self) -> None:
        return None


def default_mock_video() -> Path | None:
    """A small real-face recording shipped with the research data, if present."""
    d = REPO_ROOT / "data" / "raw" / "mcd_rppg" / "video"
    files = sorted(d.glob("*.avi"), key=lambda p: p.stat().st_size) if d.exists() else []
    return files[0] if files else None


class Report:
    """Collects step results."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.steps.append({"step": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}", flush=True)


def test_one_camera(rep: Report, cfg: dict, index: int, seconds: float, out_dir: Path, label: str, snapshot: bool = False,
                    stop: "threading.Event | None" = None) -> dict:
    """Open one camera, stream for ``seconds``, and judge frames / face / pulse. Returns a summary dict."""
    summary: dict = {"index": index}
    try:
        src = CameraSource(index)
    except SourceError as e:
        rep.add(f"{label}: open", "WARN", str(e))
        return summary
    except Exception as e:  # noqa: BLE001
        rep.add(f"{label}: open", "WARN", f"{type(e).__name__}: {e}")
        return summary
    rep.add(f"{label}: open", "PASS", f"opened (driver reports {src.fps:.0f} fps)")
    try:
        pl = Pipeline(cfg, models=None, use_landmarks=True)
    except Exception as e:  # noqa: BLE001
        src.close()
        rep.add(f"{label}: face tracker", "FAIL", f"{type(e).__name__}: {e}")
        return summary
    n, t_first, lumas, diffs, prev, a, snap = 0, None, [], [], None, None, False
    last_est, t_end = 0.0, time.perf_counter() + seconds
    shape = None
    try:
        while time.perf_counter() < t_end and not (stop is not None and stop.is_set()):
            frame, t, roi = src.read()
            if isinstance(frame, str):
                continue
            if frame is None:
                break
            if t_first is None:
                t_first = t
            shape = frame.shape
            lumas.append(float(frame.mean()))
            small = cv2.resize(frame, (32, 24)).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.abs(small - prev).mean()))
            prev = small
            if snapshot and not snap:
                cv2.imwrite(str(out_dir / f"snapshot_cam{index}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                snap = True
            pl.push(frame, t, roi)
            n += 1
            if t - last_est >= 1.0 and pl.seconds() > 2.0:
                last_est = t
                a = pl.estimate()
    except SourceError as e:
        rep.add(f"{label}: streaming", "WARN", str(e))
    except Exception:  # noqa: BLE001
        rep.add(f"{label}: streaming", "FAIL", traceback.format_exc())
    finally:
        src.close()
        pl.close()                      # free the face tracker now; MediaPipe can deadlock when the garbage collector closes it
    dur = max((time.perf_counter() - (t_end - seconds)), 1e-6)
    fps = n / dur
    summary.update({"frames": n, "fps": round(fps, 1), "resolution": None if shape is None else f"{shape[1]}x{shape[0]}"})
    if n == 0:
        rep.add(f"{label}: frames", "WARN", "no frames could be read (camera in use by another app, or Windows is blocking camera access)")
        return summary
    rep.add(f"{label}: frames", "PASS" if fps >= MIN_FPS else "WARN", f"{n} frames in {dur:.0f} s = {fps:.1f} fps at {summary['resolution']}")
    luma, motion = float(np.mean(lumas)), float(np.mean(diffs)) if diffs else 0.0
    summary.update({"mean_brightness": round(luma, 1), "frame_change": round(motion, 2)})
    if luma < 8:
        rep.add(f"{label}: picture", "WARN", f"the image is almost black (brightness {luma:.0f}/255): lens cover / privacy shutter / camera disabled?")
    elif motion < 0.05:
        rep.add(f"{label}: picture", "WARN", "the image never changes (frozen camera or still picture)")
    else:
        rep.add(f"{label}: picture", "PASS", f"brightness {luma:.0f}/255, live image")
    face = pl.last_env.get("face_fraction") if pl.last_env else None
    summary["face_fraction"] = None if face is None else round(float(face), 2)
    if face is None:
        rep.add(f"{label}: face", "WARN", "not enough data to judge the face tracking")
    elif face >= 0.8:
        rep.add(f"{label}: face", "PASS", f"face tracked in {face * 100:.0f}% of frames")
    else:
        rep.add(f"{label}: face", "WARN", f"face found in only {face * 100:.0f}% of frames: face the camera with good light, 0.5-1 m away")
    if a is None:
        rep.add(f"{label}: pulse pipeline", "WARN", f"ran {n} frames but the run was too short to produce an estimate (needs about 10 s)")
        return summary
    summary.update({"status": a.status, "text": a.text, "bpm": None if a.bpm is None else round(float(a.bpm), 1),
                    "methods": {k: round(float(v["bpm"]), 1) for k, v in pl.last_est.items()}})
    if a.status == "ok":
        rep.add(f"{label}: pulse pipeline", "PASS", f"reliable estimate {a.bpm:.0f} BPM (average of the agreeing methods; simulated events)")
    else:
        rep.add(f"{label}: pulse pipeline", "PASS", f"ran correctly; no reliable pulse yet - '{a.text}' (CARDION deliberately shows no number)")
    return summary


def run_camera_test(cfg: dict, seconds: float = 25.0, index: int | None = None, mock: bool = False, video: str | None = None, snapshot: bool = False) -> int:
    """Run the camera test. Returns 0 unless the software itself failed (a missing camera is a warning)."""
    stamp = timestamp()
    out_dir = REPO_ROOT / "out" / f"camera_test_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    setting = windows_camera_setting()
    rep.add("Windows camera privacy setting", "WARN" if setting == "deny" else "PASS",
            {"deny": "Camera access is OFF: Settings > Privacy & security > Camera (turn on 'Camera access' and 'Let desktop apps access your camera')."}.get(
                setting, f"setting: {setting}"))
    summaries: list[dict] = []
    if mock:
        path = Path(video) if video else default_mock_video()
        if path is None or not path.exists():
            rep.add("mock video", "FAIL", "no video available for the mock camera (pass --video)")
            return _finish(rep, summaries, out_dir, stamp, mock, 1)
        rep.add("mock camera", "PASS", f"replaying {path.name} as a live 30 fps webcam (software path only - this does not test hardware)")
        os.environ["CARDION_MOCK_CAMERA"] = str(path)                      # sources.open_capture() then serves the video as camera 0
    try:
        cams = list_cameras() if index is None else [{"index": index, "name": f"Camera {index}", "width": 0, "height": 0}]
        if not cams:
            rep.add("camera detection", "WARN", "no working camera was found. Plug one in, close other apps that use it, check the Windows camera "
                    "privacy settings, then run this test again. (The built-in demo and video files work without a camera.)")
        else:
            rep.add("camera detection", "PASS", ", ".join(f"{c['name']} ({c['width']}x{c['height']})" for c in cams))
        for c in cams:
            summaries.append(test_one_camera(rep, cfg, c["index"], seconds, out_dir, c["name"], snapshot))
    finally:
        if mock:
            os.environ.pop("CARDION_MOCK_CAMERA", None)
    failed = any(s["status"] == "FAIL" for s in rep.steps)
    return _finish(rep, summaries, out_dir, stamp, mock, 1 if failed else 0)


def _finish(rep: Report, summaries: list[dict], out_dir: Path, stamp: str, mock: bool, code: int) -> int:
    notes = ["Mock mode: only the software path was exercised; no physical camera was opened." if mock else
             "Real camera hardware was opened for this test.", "Events are simulated from the camera video; this is not a medical device."]
    result = {"created": stamp, "mock": mock, "steps": rep.steps, "cameras": summaries, "notes": notes, "ok": code == 0}
    jp = save_json(result, REPO_ROOT / "out" / f"camera_test_{stamp}.json")
    save_json(result, REPO_ROOT / "out" / "camera_test_latest.json")
    lines = [f"# CARDION camera test {stamp}", "", f"**{'No software failures' if code == 0 else 'FAILURES - see below'}**", ""]
    lines += [f"- {s['status']} - {s['step']}: {s['detail'].splitlines()[0]}" for s in rep.steps] + [""] + notes
    (REPO_ROOT / "out" / "camera_test_latest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nSaved: {jp}", flush=True)
    return code
