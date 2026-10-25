"""Engine: runs a source through the pipeline on a background thread, serves state/JPEGs, and records sessions.

All public methods are thread-safe and never raise into the web layer; problems are reported in ``state()['message']``.
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..common import REPO_ROOT, repo_path, timestamp
from .pipeline import LABELS, METHODS, Pipeline
from .guidance import quality_dimensions
from .quality import STATUS_TEXT
from .sources import CameraSource, DemoSource, SourceError, VideoSource, list_cameras

log = logging.getLogger("cardion.app")
DISCLAIMER = ("Research prototype - not a medical device. Events are SIMULATED from ordinary camera video; no event-camera "
              "hardware is used.")


def load_models(cfg: dict) -> tuple[dict, dict]:
    """Load SNN/ANN checkpoints from ``models/`` (missing or broken files are skipped); returns (models, info)."""
    import torch
    from ..snn_model import build_model
    mdir = repo_path(cfg.get("app", {}).get("models_dir", "models"))
    info_path = mdir / "models.json"
    meta = json.loads(info_path.read_text(encoding="utf-8-sig")) if info_path.exists() else {}
    models, info = {}, {}
    for kind in ("snn", "ann"):
        f = mdir / f"{kind}.pt"
        if not f.exists():
            info[kind] = {"loaded": False, "reason": "checkpoint not found"}
            continue
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = build_model(kind, ck.get("cfg", cfg))
            m.load_state_dict(ck["state"])
            m.eval()
            models[kind] = m
            info[kind] = {"loaded": True, **meta.get(kind, {})}
        except Exception as e:  # noqa: BLE001 - a corrupt model must not stop the app
            info[kind] = {"loaded": False, "reason": f"{type(e).__name__}: {e}"}
    return models, info


class Engine:
    """Owns the source thread and the latest results."""

    MAX_QUEUE = 150            # camera frames waiting for processing (5 s at 30 fps) before the oldest is dropped (and counted)
    LAG_FRAMES = 45            # processing this far behind the camera (1.5 s) pauses the optional event path

    def __init__(self, cfg: dict, fast: bool = False, use_landmarks: bool = True) -> None:
        self.cfg = cfg
        self._limit_threads(int(cfg.get("app", {}).get("torch_threads", 2)))
        self.fast = fast
        self.use_landmarks = use_landmarks
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.models, self.model_info = load_models(cfg)
        self.pipeline: Pipeline | None = None
        self.message = "Choose a source to begin."
        self.error: str | None = None
        self.source_info: dict = {"type": None, "label": None}
        self.latest: dict = {}
        self.jpeg_rgb: bytes | None = None
        self.jpeg_ev: bytes | None = None
        self.fps = 0.0
        self.frames = 0
        self.cameras: list[dict] = []
        self.rec: dict | None = None
        self.speed = 1.0
        self.finished = False
        self.hr_history: deque = deque(maxlen=600)
        self.true_bpm: float | None = None
        self.notice: str | None = None
        self.last_ok: dict | None = None                  # last reliable reading: {"bpm", "wall"}
        self.perf: dict = {"frames": 0, "read_s": 0.0, "push_s": 0.0, "estimate_s": 0.0, "render_s": 0.0, "dropped": 0}
        self.camera_fps = 0.0                               # measured arrival rate of camera frames
        self.rev = 0                                        # increments with every new one-per-second estimate (lets the page skip redundant redraws)
        self.new_frame = threading.Condition()              # wakes preview streams the moment a new picture exists
        self.frame_seq = self.ev_seq = 0
        self.ev_watchers = 0                                # browsers currently showing the event picture
        self.wall_t0: float | None = None                   # wall-clock time of the first processed frame (row time 0); lets reference readings be aligned
        self.listeners: list = []                           # callables(row) invoked for every one-per-second estimate (validation recorder)
        self.video_sink = None                              # optional object with write(bgr) that also receives the preview frames (opt-in video saving)
        self._outline_for = None
        self._outline = []
        self.camera_check: dict = {"running": False, "steps": [], "done": False}
        self._check_stop = threading.Event()

    @staticmethod
    def _limit_threads(n: int) -> None:
        """PyTorch defaults to one worker per logical core (12-16 here); for these tiny networks that only burns CPU (measured: 9 cores busy and
        slower than real time vs 2 cores and faster, with bit-identical outputs - see ``scripts/bench_threads.py``)."""
        import torch
        if n > 0:
            torch.set_num_threads(n)

    # ------------------------------------------------------------------ sources
    def discover_cameras(self) -> list[dict]:
        """(Re)scan cameras; result is cached in ``self.cameras``."""
        self.cameras = list_cameras()
        return self.cameras

    def start(self, spec: dict) -> dict:
        """Start a source: ``{"type": "demo"|"video"|"camera", ...}``. Returns ``{"ok": bool, "message": str}``."""
        self.stop()
        try:
            kind = spec.get("type")
            if kind == "camera":
                src = CameraSource(int(spec.get("index", 0)))
            elif kind == "video":
                src = VideoSource(spec["path"])
            elif kind == "demo":
                src = DemoSource(self.cfg, bpm=float(spec.get("bpm", 72)), size=192, seconds=20)
                self.true_bpm = src.true_bpm
            else:
                raise SourceError(f"Unknown source type {kind!r}")
        except SourceError as e:
            self._fail(str(e))
            return {"ok": False, "message": str(e)}
        except Exception as e:  # noqa: BLE001
            msg = f"Could not start the source ({type(e).__name__}: {e})"
            self._fail(msg)
            return {"ok": False, "message": msg}
        with self.lock:
            self.error = None
            self.finished = False
            self.speed = float(spec.get("speed", 1.0))
            self.source_info = {"type": kind, "label": src.label, "fps": src.fps}
            if kind != "demo":
                self.true_bpm = None
            self.hr_history.clear()
            self.frames = 0
            self.perf = {"frames": 0, "read_s": 0.0, "push_s": 0.0, "estimate_s": 0.0, "render_s": 0.0, "dropped": 0}
            self.camera_fps = 0.0
            self.wall_t0 = None
            self.jpeg_ev = None
            self.notice = None
            try:
                self.last_ok = None
                old, self.pipeline = self.pipeline, None
                if old is not None:
                    old.close()                                    # free the previous face tracker now (MediaPipe can deadlock if the garbage collector does it)
                self.pipeline = Pipeline(self.cfg, self.models, use_landmarks=self.use_landmarks and kind != "demo", guidance=(kind == "camera"))
            except Exception as e:  # noqa: BLE001
                src.close()
                msg = f"Face tracker could not start ({type(e).__name__}: {e})."
                self._fail(msg)
                return {"ok": False, "message": msg}
            self.stop_flag.clear()
            self.message = "Running."
            self.thread = threading.Thread(target=self._run, args=(src,), daemon=True, name="cardion-source")
            self.thread.start()
        return {"ok": True, "message": f"Started {src.label}"}

    def set_events(self, enabled: bool) -> dict:
        """Switch the event path (emulation, event estimators, SNN/ANN) on or off while running."""
        with self.lock:
            if self.pipeline is None:
                return {"ok": False, "message": "Start a source first."}
            self.pipeline.set_events(bool(enabled))
            self.notice = None if enabled else "Event methods are switched off (RGB methods only)."
        self._t_guard = time.perf_counter()
        return {"ok": True, "message": "Event methods " + ("on" if enabled else "off")}

    def stop(self) -> None:
        """Stop the current source (if any) and finish an active recording."""
        self.stop_flag.set()
        th = self.thread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=5)
            with self.lock:
                if not self.error and not self.finished:
                    self.message = "Stopped."
        self.thread = None
        if self.rec:
            self.record("stop")
        if th is not None and not (th.is_alive()):
            pl = self.pipeline
            if pl is not None:
                pl.close()                                        # free the face tracker now; leaving it to the garbage collector can hang MediaPipe at exit

    def _fail(self, msg: str) -> None:
        with self.lock:
            self.error = msg
            self.message = msg
        log.warning(msg)

    # ------------------------------------------------------------------ camera self-check (button in the UI)
    def start_camera_check(self, index: int = 0, seconds: float = 20.0) -> dict:
        """Run the automatic camera test in the background (stops any running source first); progress is in ``state()['camera_check']``."""
        with self.lock:
            if self.camera_check.get("running"):
                return {"ok": False, "message": "A camera check is already running."}
            self.camera_check = {"running": True, "steps": [], "done": False, "seconds": seconds}
        self.stop()
        self._check_stop.clear()

        def work() -> None:
            from .cameratest import Report, test_one_camera, windows_camera_setting
            rep = Report()
            summary: dict = {}
            out = REPO_ROOT / "out" / f"camera_check_{timestamp()}"
            try:
                out.mkdir(parents=True, exist_ok=True)
                setting = windows_camera_setting()
                rep.add("Windows camera privacy setting", "WARN" if setting == "deny" else "PASS", f"setting: {setting}")
                self.camera_check = {"running": True, "steps": rep.steps, "done": False, "seconds": seconds}
                summary = test_one_camera(rep, self.cfg, index, seconds, out, f"Camera {index}", False, self._check_stop)
            except Exception as e:  # noqa: BLE001 - reported to the UI and the log
                log.exception("camera check failed")
                rep.add("camera check", "FAIL", f"{type(e).__name__}: {e}")
            finally:
                try:
                    (REPO_ROOT / "out" / "camera_check_latest.json").write_text(
                        json.dumps({"steps": rep.steps, "summary": summary, "created": timestamp()}, indent=2), encoding="utf-8")
                except OSError:
                    pass
                self.camera_check = {"running": False, "done": True, "steps": rep.steps, "summary": summary}

        threading.Thread(target=work, daemon=True, name="cardion-camera-check").start()
        return {"ok": True, "message": f"Testing camera {index} for about {seconds:.0f} s - sit still and face the camera."}

    def cancel_camera_check(self) -> None:
        """Ask a running camera check to finish early."""
        self._check_stop.set()

    # ------------------------------------------------------------------ main loop
    def _run(self, src) -> None:
        if src.kind == "camera":
            return self._run_camera(src)
        return self._run_file(src)

    def _run_camera(self, src) -> None:
        """Live camera: one thread only reads frames and draws the preview (so the picture stays smooth whatever the computing load),
        another thread processes every frame in order. A short queue between them absorbs momentary slowness; frames are never skipped
        silently (``perf['dropped']`` counts the rare overflow)."""
        pl = self.pipeline
        q: deque = deque()
        cond = threading.Condition()
        done = threading.Event()
        arrivals: deque = deque(maxlen=60)

        def capture() -> None:
            n, win0 = 0, time.perf_counter()
            try:
                while not self.stop_flag.is_set():
                    frame, t, roi = src.read()
                    if isinstance(frame, str):                           # transient camera failure -> retry
                        continue
                    if frame is None:
                        break
                    now = time.perf_counter()
                    if self.wall_t0 is None:
                        self.wall_t0 = time.time()
                    arrivals.append(now)
                    self._render_rgb(frame, pl)
                    with cond:
                        if len(q) >= self.MAX_QUEUE:
                            q.popleft()
                            self.perf["dropped"] += 1
                        q.append((frame, t, roi))
                        cond.notify()
                    n += 1
                    if n % 90 == 0:
                        self._camera_note(src, 90 / max(time.perf_counter() - win0, 1e-6), len(q))
                        win0 = time.perf_counter()
            except SourceError as e:
                self._fail(str(e))
            except Exception as e:  # noqa: BLE001 - never crash the app
                log.exception("camera capture failure")
                self._fail(f"Camera capture stopped: {type(e).__name__}: {e}")
            finally:
                done.set()
                with cond:
                    cond.notify_all()

        cap_thread = threading.Thread(target=capture, daemon=True, name="cardion-capture")
        cap_thread.start()
        t_wall0, last_est_t, n, last_ev = time.perf_counter(), 0.0, 0, 0.0
        try:
            while not self.stop_flag.is_set():
                with cond:
                    while not q and not done.is_set() and not self.stop_flag.is_set():
                        cond.wait(0.1)
                    if not q:
                        if done.is_set() or self.stop_flag.is_set():
                            break
                        continue
                    frame, t, roi = q.popleft()
                    depth = len(q)
                t_a = time.perf_counter()
                pl.push(frame, t, roi)
                n += 1
                t_b = time.perf_counter()
                if t - last_est_t >= 1.0 and pl.seconds() > 2.0:
                    last_est_t = t
                    self._on_estimate(pl.estimate(), t)
                t_c = time.perf_counter()
                if t_c - last_ev >= 1 / 15 and self.ev_watchers > 0:      # the small event picture needs no more than 15 fps
                    last_ev = t_c
                    self._render_events(pl)
                self._perf_add(push=t_b - t_a, estimate=t_c - t_b, render=time.perf_counter() - t_c)
                self.perf["queue_max"] = max(self.perf.get("queue_max", 0), depth)
                self.frames = n
                self.fps = n / max(time.perf_counter() - t_wall0, 1e-6)
                if arrivals and len(arrivals) > 10:
                    self.camera_fps = (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], 1e-6)
                if n % 30 == 0 and pl.events_enabled and depth > self.LAG_FRAMES:
                    pl.set_events(False)                                    # the event path is optional; the displayed BPM never depends on it
                    with self.lock:
                        self.notice = (f"Event methods paused: this computer is {depth / 30:.1f} s behind the camera. RGB methods continue in real "
                                       "time. Use the event switch to retry.")
        except Exception as e:  # noqa: BLE001 - never crash the app
            log.exception("pipeline failure")
            self._fail(f"Processing stopped: {type(e).__name__}: {e}")
        finally:
            self.stop_flag.set()                                           # make sure the capture thread ends too
            done.set()
            cap_thread.join(timeout=5)
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass
            if self.rec and self.finished:
                self.record("stop")

    def _camera_note(self, src, recent_fps: float, depth: int) -> None:
        """Tell the user when the CAMERA (not this computer) is the slow part, which is common in dim rooms (long exposure)."""
        with self.lock:
            if recent_fps < 0.85 * src.fps and depth < 5:
                self.notice = (f"The camera is delivering only about {recent_fps:.0f} fps (usually because the room is dim). "
                               "More light lets it run faster. Measurements still work.")
            elif self.notice and self.notice.startswith("The camera is delivering only"):
                self.notice = None

    def _run_file(self, src) -> None:
        pl = self.pipeline
        t_wall0 = time.perf_counter()
        last_est_t = 0.0
        n = 0
        try:
            while not self.stop_flag.is_set():
                t_read = time.perf_counter()
                frame, t, roi = src.read()
                if isinstance(frame, str):
                    continue
                if frame is None:
                    with self.lock:
                        self.finished = True
                        self.message = "End of video."
                    break
                if self.wall_t0 is None:
                    self.wall_t0 = time.time()
                t_a = time.perf_counter()
                pl.push(frame, t, roi)
                n += 1
                t_b = time.perf_counter()
                if t - last_est_t >= 1.0 and pl.seconds() > 2.0:
                    last_est_t = t
                    a = pl.estimate()
                    self._on_estimate(a, t)
                t_c = time.perf_counter()
                self._render_rgb(frame, pl)
                self._render_events(pl)
                self._perf_add(read=t_a - t_read, push=t_b - t_a, estimate=t_c - t_b, render=time.perf_counter() - t_c)
                self.frames = n
                self.fps = n / max(time.perf_counter() - t_wall0, 1e-6)
                if src.kind != "camera" and not self.fast and self.speed > 0:
                    target = t / self.speed
                    sleep = target - (time.perf_counter() - t_wall0)
                    if sleep > 0:
                        time.sleep(min(sleep, 0.2))
        except SourceError as e:
            self._fail(str(e))
        except Exception as e:  # noqa: BLE001 - never crash the app
            log.exception("pipeline failure")
            self._fail(f"Processing stopped: {type(e).__name__}: {e}")
        finally:
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass
            if self.rec and self.finished:
                self.record("stop")

    def _perf_add(self, **seconds: float) -> None:
        """Accumulate per-stage loop time (read = waiting for the camera, push = face+events+networks, estimate, render = JPEG encoding)."""
        p = self.perf
        p["frames"] += 1
        for k, v in seconds.items():
            p[k + "_s"] += v
            p[k + "_max_ms"] = max(p.get(k + "_max_ms", 0.0), 1000 * v)

    def perf_summary(self) -> dict:
        """Mean milliseconds per frame for every loop stage (diagnostics for slowness; shown in the state and the profiler)."""
        p, n = self.perf, max(self.perf["frames"], 1)
        out = {"frames": p["frames"]}
        for k in ("read", "push", "estimate", "render"):
            out[k + "_ms"] = round(1000 * p[k + "_s"] / n, 2)
            out[k + "_max_ms"] = round(p.get(k + "_max_ms", 0.0), 1)
        out["busy_ms"] = round(out["push_ms"] + out["estimate_ms"] + out["render_ms"], 2)
        out["dropped_frames"] = int(p.get("dropped", 0))
        out["queue_max_frames"] = int(p.get("queue_max", 0))
        out["camera_fps"] = round(self.camera_fps, 1)
        out["process_cpu_s"] = round(time.process_time(), 2)          # processor seconds used by the whole app so far (all threads)
        return out

    def _on_estimate(self, a, t: float) -> None:
        pl = self.pipeline
        row = {"t": round(t, 2), "wall": round(time.time(), 3), "status": a.status, "bpm": a.bpm, "confidence": a.confidence,
               **{m: pl.last_est.get(m, {}).get("bpm") for m in METHODS},
               **{f"{m}_quality": pl.last_est.get(m, {}).get("peak_ratio") for m in METHODS},
               "face_fraction": pl.last_env.get("face_fraction"), "motion": pl.last_env.get("motion"), "luma": pl.last_env.get("luma"),
               "luma_var": pl.last_env.get("luma_var"), "multi_face": pl.last_env.get("multi_face"),
               "guidance": pl.guide.shown if pl.guide is not None else ""}
        with self.lock:
            self.rev += 1
            if a.status == "ok" and a.bpm is not None:
                self.last_ok = {"bpm": float(a.bpm), "wall": time.time()}
            self.hr_history.append(row)
            if self.rec:
                self.rec["rows"].append(row)
        for fn in list(self.listeners):
            try:
                fn(row)
            except Exception:  # noqa: BLE001 - a recorder problem must never stop the measurement
                log.exception("estimate listener failed")

    def _render_rgb(self, frame: np.ndarray, pl: Pipeline) -> None:
        """Encode the preview picture (with the skin outline of the latest face mask) and wake the browser's stream."""
        vis = frame.copy()
        box, mask = pl.box, pl.mask_full                  # references are replaced atomically by the processing thread
        if box is not None:
            cx, cy, s = box
            if pl.guide is None:                  # live camera: the browser draws a subtler overlay (and the guide); other sources keep this box
                cv2.rectangle(vis, (int(cx - s / 2), int(cy - s / 2)), (int(cx + s / 2), int(cy + s / 2)), (0, 220, 120), 2)
            if mask is not None:
                if self._outline_for is not mask:                 # the outline only changes when the face is re-detected
                    self._outline_for = mask
                    self._outline = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(vis, self._outline, -1, (255, 200, 0), 1)
        h, w = vis.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1.0 or h < 300:
            vis = cv2.resize(vis, (int(w * min(scale, 2.0)), int(h * min(scale, 2.0))))
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        ok, jp = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.new_frame:
            if ok:
                self.jpeg_rgb = jp.tobytes()
                self.frame_seq += 1
            if self.rec is not None and self.rec.get("video") is not None:
                self.rec["video"].write(bgr)
            if self.video_sink is not None:
                try:
                    self.video_sink.write(bgr)
                except Exception:  # noqa: BLE001
                    self.video_sink = None
            self.new_frame.notify_all()

    def _render_events(self, pl: Pipeline) -> None:
        """Encode the small simulated-event picture."""
        ev = pl.event_view(256)
        ok, jp = cv2.imencode(".jpg", cv2.cvtColor(ev, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.new_frame:
                self.jpeg_ev = jp.tobytes()
                self.ev_seq += 1
                self.new_frame.notify_all()

    def wait_frame(self, which: str, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        """Block until a picture newer than ``last_seq`` exists (or the timeout passes); returns (jpeg or None, its sequence number)."""
        with self.new_frame:
            attr, seq = ("jpeg_rgb", "frame_seq") if which == "rgb" else ("jpeg_ev", "ev_seq")
            if getattr(self, seq) == last_seq:
                self.new_frame.wait(timeout)
            return (getattr(self, attr), getattr(self, seq)) if getattr(self, seq) != last_seq else (None, last_seq)

    # ------------------------------------------------------------------ state for the UI
    def live_state(self) -> dict:
        """The small part of the state that changes quickly (headline, positioning guidance, progress); a few hundred bytes."""
        with self.lock:
            pl = self.pipeline
            a = pl.last_assessment if pl else None
            running = bool(self.thread and self.thread.is_alive())
            h = ({"status": a.status, "text": a.text, "bpm": _clean(a.bpm), "spread": _clean(a.spread)} if a
                 else {"status": "collecting", "text": STATUS_TEXT["collecting"], "bpm": None, "spread": None})
            return {"running": running, "finished": self.finished, "message": self.message, "error": self.error, "notice": self.notice,
                    "headline": h, "buffered_s": round(pl.seconds(), 1) if pl else 0.0, "need_s": pl.qp.min_seconds if pl else 8.0,
                    "valid_s": round(pl.valid_seconds(), 1) if pl else 0.0,
                    "guidance": pl.guide.snapshot() if (pl and pl.guide is not None and running) else None,
                    "last_reliable": ({"bpm": round(self.last_ok["bpm"], 1), "age_s": round(time.time() - self.last_ok["wall"], 1)} if self.last_ok else None),
                    "rev": self.rev, "fps": round(self.fps, 1), "camera_fps": round(self.camera_fps, 1), "recording": bool(self.rec),
                    "frames": self.frames, "source": self.source_info}

    def state(self) -> dict:
        """JSON-serialisable snapshot: headline, per-method table, waveforms, history, costs, environment."""
        with self.lock:
            pl = self.pipeline
            a = pl.last_assessment if pl else None
            out = {"running": bool(self.thread and self.thread.is_alive()), "finished": self.finished, "message": self.message,
                   "error": self.error, "source": self.source_info, "fps": round(self.fps, 1), "frames": self.frames,
                   "disclaimer": DISCLAIMER, "models": self.model_info, "cameras": self.cameras, "true_bpm": self.true_bpm,
                   "recording": bool(self.rec), "events_simulated": True, "notice": self.notice,
                   "events_enabled": bool(pl.events_enabled) if pl else True, "camera_check": self.camera_check,
                   "perf": self.perf_summary(), "rev": self.rev, "camera_fps": round(self.camera_fps, 1),
                   "need_s": pl.qp.min_seconds if pl else 8.0, "valid_s": round(pl.valid_seconds(), 1) if pl else 0.0}
            if pl and a:
                est = {m: {"label": LABELS[m], "bpm": _clean(pl.last_est.get(m, {}).get("bpm")),
                           "quality": _clean(pl.last_est.get(m, {}).get("peak_ratio"))} for m in METHODS}
                out["headline"] = {"status": a.status, "text": a.text, "bpm": _clean(a.bpm), "spread": _clean(a.spread),
                                   "confidence": round(a.confidence, 2), "detail": {k: _clean(v) for k, v in a.detail.items()}}
                out["estimates"] = est
                out["env"] = {k: _clean(v) for k, v in pl.last_env.items()}
                out["waveforms"] = {m: [round(float(v), 2) for v in _norm(p)[-300:]] for m, p in pl.last_pulses.items()}
                out["costs"] = _clean_dict(pl.costs())
                out["events_per_s"] = round(pl.events_total / max(pl.rec[-1][0], 1e-6), 1) if pl.rec else 0.0
                out["buffered_s"] = round(pl.seconds(), 1)
                out["quality_dims"] = quality_dimensions(pl.guide.snapshot() if pl.guide else None, a.status, pl.last_env, a.detail,
                                                         pl.qp.min_seconds, pl.qp)
            else:
                out["headline"] = {"status": "collecting", "text": STATUS_TEXT["collecting"], "bpm": None, "spread": None, "confidence": 0.0,
                                   "detail": {"seconds": pl.seconds() if pl else 0.0}}
                out["buffered_s"] = round(pl.seconds(), 1) if pl else 0.0
            out["guidance"] = pl.guide.snapshot() if (pl and pl.guide is not None and out["running"]) else None
            out["last_reliable"] = ({"bpm": round(self.last_ok["bpm"], 1), "age_s": round(time.time() - self.last_ok["wall"], 1)} if self.last_ok else None)
            out["history"] = [{k: _r1(_clean(v)) for k, v in r.items() if k in ("t", "bpm", "status") or k in METHODS} for r in list(self.hr_history)[-300:]]
            return out

    # ------------------------------------------------------------------ recording / export
    def record(self, action: str, with_video: bool = False) -> dict:
        """``start`` a recording session or ``stop`` it (writes CSV, JSON summary, PNG plot)."""
        with self.lock:
            if action == "start":
                if self.rec:
                    return {"ok": False, "message": "Already recording."}
                if not (self.thread and self.thread.is_alive()):
                    return {"ok": False, "message": "Start a source before recording."}
                d = REPO_ROOT / self.cfg.get("app", {}).get("sessions_dir", "sessions") / f"session_{timestamp()}"
                d.mkdir(parents=True, exist_ok=True)
                video = None
                if with_video and self.jpeg_rgb:
                    img = cv2.imdecode(np.frombuffer(self.jpeg_rgb, np.uint8), cv2.IMREAD_COLOR)
                    video = cv2.VideoWriter(str(d / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (img.shape[1], img.shape[0]))
                self.rec = {"dir": d, "rows": [], "t0": time.time(), "source": dict(self.source_info), "video": video}
                return {"ok": True, "message": f"Recording to {d.name}", "dir": str(d)}
            if action == "stop":
                if not self.rec:
                    return {"ok": False, "message": "Not recording."}
                rec, self.rec = self.rec, None
        return self._finish_recording(rec)

    def _finish_recording(self, rec: dict) -> dict:
        d: Path = rec["dir"]
        rows = rec["rows"]
        if rec.get("video") is not None:
            rec["video"].release()
        cols = (["t", "status", "bpm", "confidence"] + METHODS + [f"{m}_quality" for m in METHODS]
                + ["face_fraction", "motion", "luma", "luma_var", "multi_face", "guidance"])
        with open(d / "estimates.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows([{k: ("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()} for r in rows])
        ok_rows = [r for r in rows if r["status"] == "ok" and r["bpm"] is not None]
        summary = {"disclaimer": DISCLAIMER, "events": "simulated from camera video", "source": rec["source"], "duration_s": round(time.time() - rec["t0"], 1),
                   "estimates_total": len(rows), "reliable_estimates": len(ok_rows),
                   "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},   # why readings were withheld
                   "confidence_note": "the confidence column is an internal heuristic, not a calibrated probability",
                   "mean_reliable_bpm": float(np.mean([r["bpm"] for r in ok_rows])) if ok_rows else None,
                   "min_reliable_bpm": float(np.min([r["bpm"] for r in ok_rows])) if ok_rows else None,
                   "max_reliable_bpm": float(np.max([r["bpm"] for r in ok_rows])) if ok_rows else None,
                   "known_true_bpm": self.true_bpm, "models": self.model_info, "costs": _clean_dict(self.pipeline.costs()) if self.pipeline else {}}
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        try:
            _plot_session(rows, d / "plots.png", self.true_bpm)
        except Exception as e:  # noqa: BLE001
            log.warning("plot failed: %s", e)
        return {"ok": True, "message": f"Saved session to {d}", "dir": str(d), "summary": summary}

    def list_sessions(self) -> list[dict]:
        """Saved sessions, newest first."""
        root = REPO_ROOT / self.cfg.get("app", {}).get("sessions_dir", "sessions")
        out = []
        for d in sorted(root.glob("session_*"), reverse=True) if root.exists() else []:
            s = d / "summary.json"
            out.append({"id": d.name, "summary": json.loads(s.read_text()) if s.exists() else None})
        return out


def _clean(v):
    """Make numbers JSON-safe (NaN/inf -> None, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _r1(v):
    """One decimal for display payloads (the saved CSV keeps full precision)."""
    return round(v, 1) if isinstance(v, float) else v


def _clean_dict(d: dict) -> dict:
    return {k: (_clean_dict(v) if isinstance(v, dict) else _clean(v)) for k, v in d.items()}


def _norm(p: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std waveform for plotting."""
    p = np.asarray(p, float)
    return (p - p.mean()) / (p.std() + 1e-9)


def _plot_session(rows: list[dict], path: Path, true_bpm: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    t = [r["t"] for r in rows]
    for m in METHODS:
        v = [r.get(m) if r.get(m) is not None else np.nan for r in rows]
        if np.isfinite(v).any():
            ax.plot(t, v, lw=1, alpha=0.7, label=LABELS[m])
    ok = [(r["t"], r["bpm"]) for r in rows if r["status"] == "ok" and r["bpm"] is not None]
    if ok:
        ax.plot(*zip(*ok), "k-", lw=2.5, label="CARDION reliable estimate")
    if true_bpm:
        ax.axhline(true_bpm, color="g", ls="--", label="known simulated rate")
    ax.set_xlabel("time (s)"); ax.set_ylabel("BPM"); ax.legend(fontsize=6, ncol=2)
    ax.set_title("CARDION session (events simulated; not a medical device)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

