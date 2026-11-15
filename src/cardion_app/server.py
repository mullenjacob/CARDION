"""Local web server for the CARDION app (standard library only; binds to 127.0.0.1).

Routes: ``/`` UI, ``/api/state``, ``/api/cameras``, ``/api/source`` (POST), ``/api/upload`` (POST raw body), ``/api/record`` (POST),
``/api/sessions``, ``/api/session/<id>.zip``, ``/stream/rgb.mjpg``, ``/stream/events.mjpg``, ``/api/shutdown`` (POST).
POST requests must carry the header ``X-CARDION: 1`` (blocks cross-site requests from other web pages); the ``Host`` header must be local.
"""
from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..common import REPO_ROOT
from .engine import Engine
from .validation import ValidationError, ValidationManager

log = logging.getLogger("cardion.app")
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 4 * 1024 ** 3


def _json_default(o):
    """numpy numbers/arrays that slip into a state dictionary must never make /api/state fail."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def make_handler(engine: Engine, shutdown_cb):
    """Create a request-handler class bound to ``engine``."""

    val = getattr(engine, "validation", None) or ValidationManager(engine)
    engine.validation = val
    FILE_OK = {"report.html": "text/html; charset=utf-8", "report.png": "image/png", "report.json": "application/json", "pairs.csv": "text/csv", "estimates.csv": "text/csv",
               "reference_readings.csv": "text/csv", "session.json": "application/json", "audit.jsonl": "text/plain"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "CARDION"

        def log_message(self, fmt, *args):  # quiet
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ---------------------------------------------------------------- helpers
        def _local_host(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0].lower()
            return host in ("127.0.0.1", "localhost", "[::1]")

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, default=_json_default).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        # ---------------------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802
            if not self._local_host():
                return self._json({"error": "forbidden host"}, 403)
            u = urlparse(self.path)
            p = u.path
            try:
                if p in ("/", "/index.html"):
                    return self._static("index.html")
                if p in ("/validate", "/validate.html"):
                    return self._static("validate.html")
                if p == "/api/validation/state":
                    return self._json(val.snapshot())
                if p == "/api/validation/participants":
                    return self._json(val.participants())
                if p == "/api/validation/sessions":
                    return self._json({"sessions": val.sessions()})
                if p == "/api/validation/study":
                    return self._json(val.study())
                m = re.fullmatch(r"/validation/exports/(study_report\.html|study_summary\.json|all_pairs\.csv|all_sessions\.csv)", p)
                if m:
                    return self._export_file(m.group(1))
                m = re.fullmatch(r"/validation/files/(P[A-Za-z0-9_-]{0,23})/(S[0-9\-]+)/([a-z_.]+)", p)
                if m and m.group(3) in FILE_OK:
                    return self._validation_file(m.group(1), m.group(2), m.group(3))
                m = re.fullmatch(r"/api/validation/export/(P[A-Za-z0-9_-]{0,23})/(S[0-9\-]+)\.zip", p)
                if m:
                    return self._validation_zip(m.group(1), m.group(2))
                if p.startswith("/static/"):
                    return self._static(p[len("/static/"):])
                if p == "/api/state":
                    return self._json(engine.state())
                if p == "/api/live":                       # small, fast-changing part of the state (polled several times a second)
                    return self._json(engine.live_state())
                if p == "/api/cameras":
                    return self._json({"cameras": engine.discover_cameras()})
                if p == "/api/sessions":
                    return self._json({"sessions": engine.list_sessions()})
                m = re.fullmatch(r"/api/session/(session_[0-9\-]+)\.zip", p)
                if m:
                    return self._zip(m.group(1))
                if p in ("/stream/rgb.mjpg", "/stream/events.mjpg"):
                    return self._mjpeg("rgb" if "rgb" in p else "ev")
                if p in ("/frame/rgb.jpg", "/frame/events.jpg"):
                    jp = engine.jpeg_rgb if "rgb" in p else engine.jpeg_ev
                    if jp is None:
                        return self._json({"error": "no frame yet"}, 404)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jp)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jp)
                    return
                return self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # noqa: BLE001
                log.exception("GET %s failed", p)
                try:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                except Exception:  # noqa: BLE001
                    pass

        def _static(self, rel: str) -> None:
            f = (STATIC / rel).resolve()
            if STATIC.resolve() not in f.parents and f != STATIC.resolve() or not f.is_file():
                return self._json({"error": "not found"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", (mimetypes.guess_type(str(f))[0] or "application/octet-stream") + ("; charset=utf-8" if f.suffix in (".html", ".js", ".css") else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _validation_file(self, pid: str, sid: str, name: str) -> None:
            f = val.root / pid / sid / name
            if not f.is_file():
                return self._json({"error": "not found"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", FILE_OK[name])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _export_file(self, name: str) -> None:
            f = val.root / "exports" / name
            if not f.is_file():
                val.study()                                   # builds the exports on first request
            if not f.is_file():
                return self._json({"error": "not found"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {"html": "text/html; charset=utf-8", "json": "application/json", "csv": "text/csv"}[name.rsplit(".", 1)[1]])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _validation_zip(self, pid: str, sid: str) -> None:
            d = val.root / pid / sid
            if not d.is_dir():
                return self._json({"error": "no such session"}, 404)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in d.iterdir():
                    if f.is_file():
                        z.write(f, f"{pid}_{sid}/{f.name}")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="CARDION_validation_{pid}_{sid}.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _zip(self, sid: str) -> None:
            d = REPO_ROOT / engine.cfg.get("app", {}).get("sessions_dir", "sessions") / sid
            if not d.is_dir():
                return self._json({"error": "no such session"}, 404)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in d.iterdir():
                    z.write(f, f"{sid}/{f.name}")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{sid}.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _mjpeg(self, which: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = -1
            if which == "ev":
                engine.ev_watchers += 1                  # the event picture is only encoded while somebody is looking at it
            try:
                while True:
                    jp, seq = engine.wait_frame(which, seq, 1.0)      # wakes the moment a new picture exists (no polling delay)
                    if jp is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jp)).encode() + b"\r\n\r\n" + jp + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
            finally:
                if which == "ev":
                    engine.ev_watchers = max(0, engine.ev_watchers - 1)

        # ---------------------------------------------------------------- POST
        def do_POST(self) -> None:  # noqa: N802
            if not self._local_host():
                return self._json({"error": "forbidden host"}, 403)
            if self.headers.get("X-CARDION") != "1":
                return self._json({"error": "missing X-CARDION header"}, 403)
            u = urlparse(self.path)
            try:
                if u.path == "/api/upload":
                    return self._upload(parse_qs(u.query))
                body = self._body()
                data = json.loads(body.decode("utf-8")) if body else {}
                if u.path.startswith("/api/validation/"):
                    return self._validation_post(u.path[len("/api/validation/"):], data)
                if u.path == "/api/source":
                    if val.active:
                        return self._json({"ok": False, "message": "A validation session is running. Stop it on the Validation page first."}, 400)
                    res = engine.start(data)
                    return self._json(res, 200 if res["ok"] else 400)
                if u.path == "/api/events":
                    return self._json(engine.set_events(bool(data.get("enabled", True))))
                if u.path == "/api/cameracheck":
                    if data.get("cancel"):
                        engine.cancel_camera_check()
                        return self._json({"ok": True})
                    res = engine.start_camera_check(int(data.get("index", 0)), float(min(60.0, max(4.0, data.get("seconds", 20)))))
                    return self._json(res, 200 if res["ok"] else 400)
                if u.path == "/api/stop":
                    if val.active:
                        val.stop("stopped by operator")
                    else:
                        engine.stop()
                    return self._json({"ok": True})
                if u.path == "/api/record":
                    res = engine.record(data.get("action", ""), bool(data.get("video", False)))
                    return self._json({k: v for k, v in res.items() if k != "summary"} | ({"summary": res["summary"]} if "summary" in res else {}),
                                      200 if res["ok"] else 400)
                if u.path == "/api/open_sessions":
                    d = REPO_ROOT / engine.cfg.get("app", {}).get("sessions_dir", "sessions")
                    d.mkdir(exist_ok=True)
                    if sys.platform.startswith("win"):
                        os.startfile(str(d))  # noqa: S606 - opens the user's own folder
                    return self._json({"ok": True, "path": str(d)})
                if u.path == "/api/shutdown":
                    self._json({"ok": True})
                    threading.Thread(target=shutdown_cb, daemon=True).start()
                    return
                return self._json({"error": "not found"}, 404)
            except json.JSONDecodeError:
                return self._json({"error": "invalid JSON"}, 400)
            except Exception as e:  # noqa: BLE001
                log.exception("POST %s failed", u.path)
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _validation_post(self, action: str, data: dict) -> None:
            """Validation-mode actions; a refused request answers 400 with a plain-language message."""
            try:
                if action == "start":
                    return self._json(val.start(data))
                if action == "stop":
                    return self._json(val.stop("stopped by operator"))
                if action == "reference":
                    op = data.get("action", "add")
                    if op == "add":
                        return self._json(val.add_reference(data.get("value"), data.get("ago_s", 0), data.get("t_s"), data.get("note", "")))
                    if op == "edit":
                        return self._json(val.edit_reference(data.get("id"), data.get("value"), data.get("t_s"), data.get("note")))
                    if op == "delete":
                        return self._json(val.delete_reference(data.get("id")))
                    return self._json({"ok": False, "message": "Unknown reference action."}, 400)
                if action == "import":
                    return self._json(val.import_reference(data.get("path", "")))
                if action == "load":
                    return self._json(val.load(data.get("participant", ""), data.get("session_id", "")))
                if action == "open_folder":
                    val.root.mkdir(parents=True, exist_ok=True)
                    if sys.platform.startswith("win"):
                        os.startfile(str(val.root))  # noqa: S606 - opens the user's own folder
                    return self._json({"ok": True, "path": str(val.root)})
                return self._json({"error": "not found"}, 404)
            except ValidationError as e:
                return self._json({"ok": False, "message": str(e)}, 400)

        def _upload(self, q: dict) -> None:
            name = re.sub(r"[^A-Za-z0-9._-]", "_", (q.get("name") or ["video.mp4"])[0])[-80:]
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_UPLOAD:
                return self._json({"error": "empty or too large upload"}, 400)
            d = REPO_ROOT / engine.cfg.get("app", {}).get("sessions_dir", "sessions") / "uploads"
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{int(time.time())}_{name}"
            left = n
            with open(dest, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            return self._json({"ok": left == 0, "path": str(dest), "bytes": n - left})

    return Handler


class QuietServer(ThreadingHTTPServer):
    """Browsers close streaming connections all the time; that is normal and must not fill the log with tracebacks."""

    def handle_error(self, request, client_address) -> None:  # noqa: D401
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            log.debug("client %s disconnected: %s", client_address, exc)
            return
        log.exception("unexpected error while serving %s", client_address)


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8765) -> tuple[ThreadingHTTPServer, int]:
    """Create the server on the first free port at or after ``port``."""
    srv = None
    for p in range(port, port + 30):
        try:
            holder: dict = {}
            srv = QuietServer((host, p), make_handler(engine, lambda: holder["srv"].shutdown()))
            holder["srv"] = srv
            srv.daemon_threads = True
            return srv, p
        except OSError:
            continue
    raise RuntimeError(f"no free port in {port}-{port + 29}")
