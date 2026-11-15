"""Automated self-test: exercises the real app (server, demo source, video source, recording, error handling) and writes a report.

    python -m src.cardion_app --selftest        (or double-click ``Run CARDION Self-Test.bat``)

Writes ``out/selftest_report.md`` and ``out/selftest_report.json`` and returns exit code 0 only if every check passed. Camera
hardware is never opened by the self-test (camera code is checked by simulated failures only) - this is stated in the report.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..common import REPO_ROOT, save_json, timestamp
from .engine import Engine
from .server import serve
from .sources import CameraSource, SourceError, VideoSource, list_cameras

TOL_BPM = 3.0


def _http(base: str, path: str, body=None, raw: bytes | None = None, method: str | None = None) -> tuple[int, bytes]:
    headers = {"X-CARDION": "1"}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    if raw is not None:
        data = raw
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _wait_for(base: str, pred, timeout: float) -> dict:
    t0, s = time.time(), {}
    while time.time() - t0 < timeout:
        s = json.loads(_http(base, "/api/state")[1])
        if pred(s):
            return s
        time.sleep(0.4)
    return s


def make_test_video(cfg: dict, path: Path, bpm: float = 84.0, seconds: float = 24.0) -> float:
    """Write a simulated-face video (known pulse) for the video-file test; returns the true BPM."""
    from ..datasets import make_synthetic_clip
    sc = {**cfg, "datasets": {**cfg["datasets"], "synthetic": {**cfg["datasets"]["synthetic"], "size": 192, "clip_seconds": seconds, "fps": 30,
                                                                 "noise_sigma_range": [1.0, 1.0]}}}
    clip = make_synthetic_clip(3, sc, seed=11, hr0=bpm)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (192, 192))
    for f in clip.face:
        w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    w.release()
    return float(clip.meta["hr0_bpm"])


def run_selftest(cfg: dict) -> int:
    """Run all checks; print and save a pass/fail report. Returns a process exit code."""
    results: list[dict] = []
    run_id = os.environ.get("CARDION_RUN_ID") or time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    for stale in ("selftest_report.md", "selftest_report.json"):          # never leave an older PASS lying around if this run crashes
        try:
            (REPO_ROOT / "out" / stale).unlink()
        except OSError:
            pass
    (REPO_ROOT / "out").mkdir(exist_ok=True)
    (REPO_ROOT / "out" / "selftest_report.md").write_text(f"# CARDION self-test run {run_id}\n\nIN PROGRESS or CRASHED - no result for this run yet.\n", encoding="utf-8")

    def check(name: str, fn) -> None:
        t0 = time.time()
        try:
            detail = fn()
            results.append({"check": name, "passed": True, "detail": str(detail), "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            results.append({"check": name, "passed": False, "detail": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)})
        r = results[-1]
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {name} ({r['seconds']} s) {r['detail'][:150]}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="cardion_selftest_"))
    cfg = {**cfg, "app": {**cfg["app"], "sessions_dir": str(Path("out") / "selftest_sessions"),
                                          "validation_dir": str(REPO_ROOT / "out" / "selftest_validation")}}
    engine = Engine(cfg, fast=True)
    srv, port = serve(engine, port=int(cfg["app"].get("port", 8765)) + 100)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    state: dict = {}

    def t_models():
        info = engine.model_info
        assert info.get("snn", {}).get("loaded") and info.get("ann", {}).get("loaded"), f"models not loaded: {info}"
        return "SNN and ANN checkpoints loaded"

    def t_ui():
        code, body = _http(base, "/")
        assert code == 200 and b"CARDION" in body, f"UI page missing ({code})"
        for f in ("/static/app.js", "/static/shared.js", "/static/validate.js", "/static/style.css", "/validate"):
            assert _http(base, f)[0] == 200, f"{f} not served"
        return "UI, script and stylesheet served"

    def t_security():
        req = urllib.request.Request(base + "/api/source", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("POST without X-CARDION header was accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        return "cross-site style POST rejected"

    def t_demo():
        code, body = _http(base, "/api/source", {"type": "demo", "bpm": 72})
        assert code == 200, body
        s = _wait_for(base, lambda s: s["headline"]["status"] == "ok", 90)
        state["demo"] = s
        assert s["headline"]["status"] == "ok", f"demo never became reliable: {s['headline']}"
        err = abs(s["headline"]["bpm"] - s["true_bpm"])
        assert err <= TOL_BPM, f"demo estimate {s['headline']['bpm']:.1f} vs true {s['true_bpm']:.1f}"
        return f"estimate {s['headline']['bpm']:.1f} BPM vs simulated truth {s['true_bpm']:.1f} (|err| {err:.1f} <= {TOL_BPM})"

    def t_methods():
        s = state["demo"]
        got = {k: v["bpm"] for k, v in s["estimates"].items() if v["bpm"] is not None}
        need = {"rgb_green", "rgb_pos", "event_green", "snn", "ann"}
        assert need <= set(got), f"missing methods: {need - set(got)}"
        return ", ".join(f"{k}={v:.0f}" for k, v in got.items())

    def t_events_toggle():
        code, b = _http(base, "/api/events", {"enabled": False})
        assert code == 200 and json.loads(b)["ok"], b
        time.sleep(3)
        s = json.loads(_http(base, "/api/state")[1])
        assert s["events_enabled"] is False and s["estimates"]["rgb_pos"]["bpm"] is not None, "RGB path must keep running with events off"
        code, b = _http(base, "/api/events", {"enabled": True})
        assert json.loads(b)["ok"]
        s = _wait_for(base, lambda s: s["events_enabled"] and (s["estimates"]["snn"]["bpm"] is not None), 60)
        assert s["events_enabled"], "events did not resume"
        return "event/neural methods can be paused and resumed without stopping the RGB path"

    def t_frames():
        code, jp = _http(base, "/frame/rgb.jpg")
        assert code == 200 and jp[:2] == b"\xff\xd8", "no RGB preview frame"
        code, jp = _http(base, "/frame/events.jpg")
        assert code == 200 and jp[:2] == b"\xff\xd8", "no event preview frame"
        return "RGB and event previews served"

    def t_record():
        code, b = _http(base, "/api/record", {"action": "start", "video": True})
        assert code == 200, b
        time.sleep(4)
        code, b = _http(base, "/api/record", {"action": "stop"})
        j = json.loads(b)
        assert code == 200 and j["ok"], b
        d = Path(j["dir"])
        files = sorted(p.name for p in d.iterdir())
        for need in ("estimates.csv", "summary.json", "plots.png"):
            assert need in files, f"{need} missing in {files}"
        assert _http(base, f"/api/session/{d.name}.zip")[0] == 200, "zip export failed"
        return f"session saved: {files}"

    def t_video():
        truth = make_test_video(cfg, tmp / "test_84bpm.avi")
        code, b = _http(base, "/api/upload?name=test_84bpm.avi", raw=(tmp / "test_84bpm.avi").read_bytes())
        assert code == 200, b
        path = json.loads(b)["path"]
        code, b = _http(base, "/api/source", {"type": "video", "path": path, "speed": 0})
        assert code == 200, b
        s = _wait_for(base, lambda s: s["finished"], 180)
        assert s["finished"], "video analysis did not finish"
        ok = [h["bpm"] for h in s["history"] if h["status"] == "ok" and h["bpm"] is not None]
        assert ok, "no reliable estimate from the video"
        err = abs(float(np.median(ok)) - truth)
        assert err <= TOL_BPM, f"video estimate {np.median(ok):.1f} vs true {truth:.1f}"
        return f"median reliable estimate {np.median(ok):.1f} vs known {truth:.1f} BPM ({len(ok)} estimates)"

    def t_bad_video():
        bad = tmp / "not_a_video.avi"
        bad.write_bytes(b"this is not a video")
        code, b = _http(base, "/api/source", {"type": "video", "path": str(bad)})
        assert code == 400 and "Could not open" in json.loads(b)["message"], (code, b)
        code, b = _http(base, "/api/source", {"type": "video", "path": str(tmp / "missing.avi")})
        assert code == 400
        s = json.loads(_http(base, "/api/state")[1])
        assert s["error"], "error not surfaced to the UI"
        return "unreadable and missing files reported without crashing"

    def t_camera_failure():
        try:
            CameraSource(99)
            raise AssertionError("camera index 99 unexpectedly opened")
        except SourceError as e:
            assert "could not be opened" in str(e)
        code, b = _http(base, "/api/source", {"type": "camera", "index": 99})
        assert code == 400 and "Camera 99" in json.loads(b)["message"]
        return "missing camera produces a clear message (no hardware was opened for real)"

    def t_quality_gates():
        from .quality import QualityParams, assess
        p = QualityParams()
        env = {"seconds": 12, "face_fraction": 1.0, "motion": 0.01, "luma": 120, "luma_var": 0.01}
        good = {k: {"bpm": 72.0 + d, "peak_ratio": 8.0} for k, d in (("rgb_green", 0.5), ("rgb_pos", -0.5), ("event_green", 1.0))}
        assert assess(good, env, p).status == "ok"
        assert assess(good, {**env, "seconds": 3}, p).status == "collecting"
        assert assess(good, {**env, "motion": 0.2}, p).status == "motion"
        assert assess(good, {**env, "luma": 10}, p).status == "dark"
        assert assess(good, {**env, "face_fraction": 0.1}, p).status == "no_face"
        weak = {k: {"bpm": 72.0, "peak_ratio": 1.0} for k in good}
        assert assess(weak, env, p).status == "weak_signal"
        split = {"rgb_green": {"bpm": 60.0, "peak_ratio": 8.0}, "rgb_pos": {"bpm": 100.0, "peak_ratio": 8.0}, "event_green": {"bpm": 140.0, "peak_ratio": 8.0}}
        assert assess(split, env, p).status == "disagree"
        return "no number is shown when data, light, motion, signal or agreement is inadequate"

    def t_camera_enum_mocked():
        import unittest.mock as mock
        with mock.patch("cv2.VideoCapture") as vc:
            vc.return_value.isOpened.return_value = False
            assert list_cameras(3) == []
        return "camera enumeration handles absent devices"

    def with_mock_camera(fn):
        """Run ``fn`` with camera 0 replaced by a recorded real-face video (the camera code path, no hardware). Skips if no video exists."""
        import os

        from .cameratest import default_mock_video
        vid = default_mock_video()
        if vid is None:
            return "skipped: no sample face video in data/raw/mcd_rppg/video"
        os.environ["CARDION_MOCK_CAMERA"] = str(vid)
        try:
            return fn()
        finally:
            os.environ.pop("CARDION_MOCK_CAMERA", None)

    def t_camera_live_mock():
        def go():
            code, b = _http(base, "/api/cameras")
            assert json.loads(b)["cameras"], "mock camera was not discovered"
            code, b = _http(base, "/api/source", {"type": "camera", "index": 0})
            assert code == 200, b
            s = _wait_for(base, lambda s: s["frames"] > 150 and s["buffered_s"] > 6, 90)
            assert s["running"] and not s["error"] and s["frames"] > 150, f"camera stream not flowing: {s['message']} {s['error']}"
            assert s["headline"]["status"] in ("ok", "collecting", "weak_signal", "disagree", "motion", "unstable_light"), s["headline"]
            _http(base, "/api/stop", {})
            return f"live-camera path: {s['frames']} frames at {s['fps']} fps, face tracked, status '{s['headline']['text']}'"
        return with_mock_camera(go)

    def t_camera_check_api():
        def go():
            code, b = _http(base, "/api/cameracheck", {"index": 0, "seconds": 8})
            assert code == 200, b
            s = _wait_for(base, lambda s: s["camera_check"]["done"], 90)
            steps = {x["step"]: x["status"] for x in s["camera_check"]["steps"]}
            assert s["camera_check"]["done"] and not any(v == "FAIL" for v in steps.values()), steps
            assert steps.get("Camera 0: frames") == "PASS" and steps.get("Camera 0: face") == "PASS", steps
            return "camera check ran: " + ", ".join(f"{k.split(': ')[-1]}={v}" for k, v in steps.items())
        return with_mock_camera(go)

    def t_guidance_live():
        """The positioning assistant through the real server: a recorded face is framed well, then moved to the side, then joined by a second face."""
        import os
        spec_file = tmp / "scenario.txt"
        spec_file.write_text("good", encoding="utf-8")

        def go():
            os.environ["CARDION_MOCK_SCENARIO_FILE"] = str(spec_file)
            try:
                code, b = _http(base, "/api/source", {"type": "camera", "index": 0})
                assert code == 200, b
                s = _wait_for(base, lambda s: s.get("guidance") and s["guidance"]["code"] in ("measuring", "collecting", "stabilizing", "disagree", "reliable", "uneven_light"), 90)
                assert not s["guidance"]["code"].startswith("move"), s["guidance"]
                spec_file.write_text("shift_x:0.33", encoding="utf-8")
                s = _wait_for(base, lambda s: s.get("guidance") and s["guidance"]["code"] == "move_right", 60)
                assert s["guidance"]["message"] == "Move slightly right.", s["guidance"]
                spec_file.write_text("two_faces", encoding="utf-8")
                s = _wait_for(base, lambda s: s.get("guidance") and s["guidance"]["code"] == "multi_face", 60)
                s = _wait_for(base, lambda s: s["headline"]["status"] == "multi_face", 40)
                assert s["headline"]["bpm"] is None, "a reading was shown with two faces in view"
                assert len(s["quality_dims"]) == 7
                _http(base, "/api/stop", {})
                return "good framing -> no advice; face moved right -> \"Move slightly right.\"; second face -> measurement paused, no BPM shown"
            finally:
                os.environ.pop("CARDION_MOCK_SCENARIO_FILE", None)
        return with_mock_camera(go)

    def t_validation_mode():
        """Validation mode through the real server: refusals, a short timed session on the recorded-face camera, report files, no accuracy claimed."""
        import shutil
        shutil.rmtree(REPO_ROOT / "out" / "selftest_validation", ignore_errors=True)
        body = {"participant": "P901", "reference_device": "Apple Watch", "condition": "Resting, seated", "duration_s": 20, "consent_confirmed": False, "source": {"type": "camera", "index": 0}}

        def go():
            code, r = _http(base, "/api/validation/start", body)
            assert code == 400 and "agreed" in json.loads(r)["message"], "a session started without the consent confirmation"
            code, r = _http(base, "/api/validation/start", {**body, "consent_confirmed": True})
            assert code == 200, r
            code, r = _http(base, "/api/validation/reference", {"action": "add", "value": "999"})
            assert code == 400, "an impossible heart rate was accepted"
            st = None
            for _ in range(80):
                time.sleep(0.5)
                code, raw = _http(base, "/api/validation/state")
                st = json.loads(raw)
                if not st["active"] and st["session"]["state"] != "starting" and st.get("report"):
                    break
            assert st and not st["active"] and st["session"]["state"] == "completed", st and st["session"]
            rep = st["report"]
            assert rep["accuracy"] == {"n_pairs": 0} and "cannot be assessed" in rep["headline"], "accuracy was claimed without any reference reading"
            d = REPO_ROOT / "out" / "selftest_validation" / "P901" / st["session"]["session_id"]
            for name in ("session.json", "estimates.csv", "reference_readings.csv", "audit.jsonl", "report.json", "report.html", "report.png"):
                assert (d / name).exists(), f"{name} missing"
            n_est = st["n_estimates"]
            return f"consent required; bad value refused; 20 s session completed with {n_est} estimates and a report; no accuracy claimed without reference readings"
        return with_mock_camera(go)

    def t_launcher_files():
        for f in ("Launch CARDION.bat", "Run CARDION Self-Test.bat", "Test and Repair CARDION.bat", "scripts/launch.ps1", "scripts/doctor.py"):
            assert (REPO_ROOT / f).is_file(), f"{f} is missing"
        ps1 = (REPO_ROOT / "scripts" / "launch.ps1").read_bytes()
        assert ps1.isascii(), "launch.ps1 must be ASCII-only (Windows PowerShell reads BOM-less files as ANSI)"
        return "launchers and doctor present; launch.ps1 is ASCII"

    for name, fn in (("models load", t_models), ("web UI served", t_ui), ("cross-site POST blocked", t_security), ("quality gating logic", t_quality_gates),
                     ("demo source reaches a reliable, correct estimate", t_demo), ("all methods report an estimate", t_methods),
                     ("event methods pause/resume", t_events_toggle), ("preview frames served", t_frames), ("recording + export", t_record), ("video file analysis matches known pulse", t_video),
                     ("bad/missing video handled", t_bad_video), ("camera failure handled", t_camera_failure),
                     ("camera enumeration (mocked)", t_camera_enum_mocked), ("live-camera code path with a recorded real face", t_camera_live_mock),
                     ("camera check (UI button backend)", t_camera_check_api), ("positioning guidance (moved face, second face)", t_guidance_live), ("validation mode session and report", t_validation_mode),
                     ("launcher files present", t_launcher_files)):
        check(name, fn)
    engine.stop()
    srv.shutdown()
    passed = sum(r["passed"] for r in results)
    out = {"run_id": run_id, "created": timestamp(), "passed": passed, "total": len(results), "all_passed": passed == len(results), "results": results,
           "not_tested": ["real webcam hardware (never opened by the self-test)", "real event-camera hardware (none is used)"]}
    path = save_json(out, REPO_ROOT / "out" / "selftest_report.json")
    from ..common import register_claim
    register_claim("selftest.passed", passed, path, "python -m src.cardion_app --selftest")
    register_claim("selftest.total", len(results), path, "python -m src.cardion_app --selftest")
    lines = [f"# CARDION self-test {out['created']} (run {run_id})", "",
             f"**{'PASS' if out['all_passed'] else 'FAIL'}: {passed}/{len(results)} checks passed**", ""]
    lines += [f"- {'PASS' if r['passed'] else 'FAIL'} - {r['check']} ({r['seconds']} s): {r['detail']}" for r in results]
    lines += ["", "Not covered by this test: " + "; ".join(out["not_tested"])]
    (REPO_ROOT / "out" / "selftest_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:4]), flush=True)
    return 0 if out["all_passed"] else 1
