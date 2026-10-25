"""Profile the live app as a browser would see it: preview smoothness, /api/state cost, CPU use, BPM stability, time to first reading.

    python scripts/profile_live.py --label before --seconds 60

Starts the real server (``python -m src.cardion_app``) with a recorded real webcam clip standing in for the camera (30 fps, real-time
pacing, the face re-framed like a normal webcam shot), starts the camera source over HTTP, then measures for ``--seconds``:

* preview: frames per second actually delivered on ``/stream/rgb.mjpg`` and the gaps between frames (a choppy preview = long gaps)
* state: latency and size of ``/api/state`` polled every 0.5 s like the page does
* CPU: processor seconds the server used per wall-clock second (100 % = one full core)
* BPM display: time to the first reliable reading, share of one-per-second estimates that were reliable, how often the shown number
  changed, the size of the changes, and how often the status flipped between reliable and not
* loop stages: milliseconds per frame spent in push / estimate / render (from the engine's own timers)

Output: ``out/profile_live_<label>_<time>.json``.
"""
from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import timestamp  # noqa: E402

VIDEO_DIR = ROOT / "data" / "raw" / "mcd_rppg" / "video"


def cpu_seconds(pid: int) -> float:
    """Kernel+user processor seconds used so far by process ``pid`` (Windows)."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)           # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return float("nan")
    ft = [ctypes.c_ulonglong() for _ in range(4)]
    k32.GetProcessTimes(h, *[ctypes.byref(x) for x in ft])
    k32.CloseHandle(h)
    return (ft[2].value + ft[3].value) / 1e7


def real_server_pid(pid: int) -> int:
    """The virtual-environment ``python.exe`` is a launcher that starts the real interpreter as a child; CPU time must be read from the child."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-CimInstance Win32_Process -Filter \"ParentProcessId={pid}\" | Where-Object Name -like python*).ProcessId"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else pid


def get_json(conn: http.client.HTTPConnection, path: str) -> tuple[dict, float, int]:
    t0 = time.perf_counter()
    conn.request("GET", path)
    r = conn.getresponse()
    body = r.read()
    return json.loads(body), 1000 * (time.perf_counter() - t0), len(body)


def post(port: int, path: str, data: dict) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", path, json.dumps(data), {"X-CARDION": "1", "Content-Type": "application/json"})
    out = json.loads(c.getresponse().read())
    c.close()
    return out


def stream_reader(port: int, stop: threading.Event, stamps: list[float], sizes: list[int]) -> None:
    """Read the MJPEG preview and note when each JPEG finished arriving."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/stream/rgb.mjpg")
    r = c.getresponse()
    buf = b""
    while not stop.is_set():
        try:
            chunk = r.read1(65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a < 0 or b < 0:
                break
            stamps.append(time.perf_counter())
            sizes.append(b + 2 - a)
            buf = buf[b + 2:]
    c.close()


def summarise_gaps(stamps: list[float]) -> dict:
    g = np.diff(stamps) * 1000 if len(stamps) > 2 else np.zeros(1)
    span = (stamps[-1] - stamps[0]) if len(stamps) > 2 else 1.0
    return {"frames": len(stamps), "fps": round((len(stamps) - 1) / span, 1), "gap_ms_median": round(float(np.median(g)), 1),
            "gap_ms_p95": round(float(np.percentile(g, 95)), 1), "gap_ms_max": round(float(g.max()), 1),
            "gaps_over_100ms": int((g > 100).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--video", default=None, help="recorded clip standing in for the webcam (default: a smaller MCD-rPPG recording)")
    ap.add_argument("--scenario", default="good")
    ap.add_argument("--tree", default=str(ROOT), help="source tree to run (e.g. a checkout of an older version for before/after comparisons)")
    a = ap.parse_args()
    video = Path(a.video) if a.video else sorted(VIDEO_DIR.glob("*.avi"), key=lambda p: p.stat().st_size)[0]
    scen = ROOT / "out" / "_profile_scenario.txt"
    scen.write_text(a.scenario, encoding="utf-8")
    env = {**os.environ, "CARDION_MOCK_CAMERA": str(video), "CARDION_MOCK_SCENARIO_FILE": str(scen)}
    py = os.environ.get("CARDION_PY", sys.executable)
    proc = subprocess.Popen([py, "-m", "src.cardion_app", "--no-browser", "--port", str(a.port)], cwd=a.tree, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        for _ in range(120):
            try:
                get_json(conn, "/api/state")
                break
            except OSError:
                conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
                time.sleep(0.5)
        print("server up; starting source", flush=True)
        print(post(a.port, "/api/source", {"type": "camera", "index": 0}), flush=True)
        stop, stamps, sizes = threading.Event(), [], []
        th = threading.Thread(target=stream_reader, args=(a.port, stop, stamps, sizes), daemon=True)
        th.start()
        time.sleep(2.0)
        spid = real_server_pid(proc.pid)
        cpu0, t0 = cpu_seconds(spid), time.perf_counter()
        lat, nbytes, shown, perf = [], [], [], {}
        first_ok = None
        while time.perf_counter() - t0 < a.seconds:
            s, ms, nb = get_json(conn, "/api/state")
            lat.append(ms)
            nbytes.append(nb)
            h = s.get("headline") or {}
            shown.append((time.perf_counter() - t0, h.get("status"), h.get("bpm"), s.get("buffered_s")))
            if first_ok is None and h.get("status") == "ok":
                first_ok = time.perf_counter() - t0 + 2.0
            perf = s.get("perf", {})
            time.sleep(max(0.0, 0.5 - ms / 1000))
        cpu = (cpu_seconds(spid) - cpu0) / (time.perf_counter() - t0)
        stop.set()
        # BPM display behaviour on the 1 Hz estimates (state is polled twice per second: keep changes of the shown value)
        seq = [(t, st, b) for t, st, b, _ in shown]
        vals = [(round(b) if st == "ok" and b is not None else None) for _, st, b in seq]
        changes = [abs(y - x) for x, y in zip(vals, vals[1:]) if x is not None and y is not None and x != y]
        flips = sum(1 for (_, s1, _), (_, s2, _) in zip(seq, seq[1:]) if (s1 == "ok") != (s2 == "ok"))
        reliable_share = float(np.mean([st == "ok" for _, st, _ in seq]))
        status_counts = {k: sum(1 for _, st, _ in seq if st == k) for k in sorted({st for _, st, _ in seq})}
        res = {"label": a.label, "tree": a.tree, "command": f"python scripts/profile_live.py --label {a.label} --seconds {a.seconds}", "clip": video.name,
               "scenario": a.scenario, "cpu_cores_used": round(cpu, 3), "cpu_logical_processors": os.cpu_count(),
               "preview": {**summarise_gaps(stamps), "jpeg_bytes_mean": int(np.mean(sizes)) if sizes else 0},
               "state_poll": {"latency_ms_median": round(float(np.median(lat)), 1), "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
                              "latency_ms_max": round(float(max(lat)), 1), "bytes_mean": int(np.mean(nbytes))},
               "loop_ms_per_frame": perf, "time_to_first_reliable_s": first_ok, "reliable_share_of_polls": round(reliable_share, 3),
               "status_counts_polls": status_counts, "shown_number_changes": len(changes),
               "shown_change_mean_abs_bpm": round(float(np.mean(changes)), 2) if changes else 0.0,
               "shown_change_max_abs_bpm": max(changes) if changes else 0, "reliable_flips": flips,
               "bpm_series": [(round(t, 1), st, None if b is None else round(b, 2)) for t, st, b in seq][::2]}
        out = ROOT / "out" / f"profile_live_{a.label}_{timestamp()}.json"
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "bpm_series"}, indent=1))
        print("saved", out)
        post(a.port, "/api/stop", {})
    finally:
        try:
            post(a.port, "/api/shutdown", {})
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
        proc.kill()


if __name__ == "__main__":
    main()
