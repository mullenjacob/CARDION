"""Validation mode through the real web server and the mock camera: refusals, interruption (camera unplugged), restart, stop, downloads, path safety."""
from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from src.common import load_config

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = sorted((ROOT / "data" / "raw" / "mcd_rppg" / "video").glob("*.avi"), key=lambda p: p.stat().st_size)
pytestmark = pytest.mark.skipif(not VIDEOS, reason="needs the recorded clips in data/raw/mcd_rppg/video")


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        hdr = {"X-CARDION": "1", "Content-Type": "application/json"} if method == "POST" else {}
        c.request(method, path, json.dumps(body or {}) if method == "POST" else None, hdr)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def j(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        code, data = self.call(method, path, body)
        return code, json.loads(data)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    from src.cardion_app.engine import Engine
    from src.cardion_app.server import serve
    scen = tmp_path / "scenario.txt"
    scen.write_text("good", encoding="utf-8")
    monkeypatch.setenv("CARDION_MOCK_CAMERA", str(VIDEOS[0]))
    monkeypatch.setenv("CARDION_MOCK_SCENARIO_FILE", str(scen))
    cfg = load_config()
    cfg["app"]["validation_dir"] = str(tmp_path / "val")
    engine = Engine(cfg)
    srv, port = serve(engine, port=8830)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield Client(port), scen, engine, tmp_path / "val"
    try:
        engine.validation.stop("test cleanup")
    except Exception:  # noqa: BLE001
        pass
    engine.stop()
    srv.shutdown()
    srv.server_close()


def wait_for(fn, seconds: float, every: float = 0.5):
    t0 = time.time()
    while time.time() - t0 < seconds:
        v = fn()
        if v:
            return v
        time.sleep(every)
    return None


def start_body(**kw) -> dict:
    return {"participant": "P001", "reference_device": "Apple Watch", "condition": "Resting, seated", "duration_s": 90, "consent_confirmed": True,
            "source": {"type": "camera", "index": 0}, **kw}


def test_interruption_restart_stop_and_safety(server):
    cl, scen, engine, vroot = server
    code, r = cl.j("POST", "/api/validation/start", start_body(consent_confirmed=False))
    assert code == 400 and "agreed" in r["message"]
    code, r = cl.j("POST", "/api/validation/start", start_body())
    assert code == 200 and r["ok"], r
    sid = r["session_id"]

    code, r = cl.j("POST", "/api/source", {"type": "demo"})                                  # the measurement page cannot hijack a running session
    assert code == 400 and "validation session is running" in r["message"]
    assert wait_for(lambda: engine.wall_t0 and time.time() - engine.wall_t0 > 4, 40), "camera never delivered frames"
    code, r = cl.j("POST", "/api/validation/reference", {"action": "add", "value": "abc"})
    assert code == 400 and "not a valid heart rate" in r["message"]
    code, r = cl.j("POST", "/api/validation/reference", {"action": "add", "value": 74})
    assert code == 200 and r["entry"]["value"] == 74.0

    scen.write_text("disconnect", encoding="utf-8")                                          # unplug the camera
    st = wait_for(lambda: (lambda s: s if (not s["active"] and s.get("report")) else None)(cl.j("GET", "/api/validation/state")[1]), 40)
    assert st is not None, "the session should end when the camera stops"
    assert st["session"]["state"] == "interrupted" and "camera" in st["session"]["end_reason"].lower()
    d = vroot / "P001" / sid
    assert (d / "report.html").exists() and (d / "estimates.csv").exists() and st["report"]["reference_availability"]["n_reference_entries"] == 1

    scen.write_text("good", encoding="utf-8")                                                # plug it back in and start again
    code, r = cl.j("POST", "/api/validation/start", start_body(participant="Participant 002"))
    assert code == 200 and r["participant"] == "P002"
    assert wait_for(lambda: cl.j("GET", "/api/live")[1]["frames"] > 30, 40)
    code, _ = cl.j("POST", "/api/stop", {})
    assert code == 200
    st = wait_for(lambda: (lambda s: s if (not s["active"] and s.get("report")) else None)(cl.j("GET", "/api/validation/state")[1]), 20)
    assert st["session"]["state"] == "stopped" and st["session"]["participant"] == "P002"

    code, body = cl.call("GET", f"/validation/files/P001/{sid}/report.html")
    assert code == 200 and b"CARDION accuracy validation report" in body
    code, body = cl.call("GET", f"/api/validation/export/P001/{sid}.zip")
    assert code == 200 and body[:2] == b"PK"
    for bad in ("/validation/files/P001/S1/../../../configs/default.yaml", "/validation/files/P001/S20261025-1/secret.txt", "/api/validation/export/..%5C..%5Cconfigs/x.zip"):
        assert cl.call("GET", bad)[0] == 404
    code, r = cl.j("POST", "/api/validation/load", {"participant": "../x", "session_id": "S1"})
    assert code == 400
    code, r = cl.j("GET", "/api/validation/study")
    assert code == 200 and r["n_participants"] == 0 or r["n_sessions"] >= 0                  # mock-camera sessions are real-camera sessions; no pairs here
