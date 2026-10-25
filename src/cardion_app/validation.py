"""Accuracy-validation sessions: record CARDION's readings next to hand-entered (or imported) reference-monitor readings, then report how close they are.

One session = one participant, one reference device, one condition, one timed measurement. Everything is stored under
``validation_data/<participant>/<session>/`` with an anonymous participant code (``P001``); no names, no faces, and no video unless the
operator explicitly ticks the box. The session folder holds:

    session.json              settings, state, reference entries (with edit history)
    estimates.csv             CARDION's one-per-second rows (status, BPM, each method, quality values)
    reference_readings.csv    the reference values and how each was timed/edited
    audit.jsonl               append-only log of every action (start, add, edit, delete, import, stop)
    pairs.csv, report.json, report.html, report.png   written when the session ends or the entries change

The maths lives in ``validation_metrics.py``; this module is the book-keeping around it.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from ..common import REPO_ROOT, timestamp
from . import validation_metrics as vm
from .pipeline import METHODS

log = logging.getLogger("cardion.validation")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
DEVICES = ["Apple Watch", "ECG chest strap", "Fingertip pulse oximeter", "Other heart-rate monitor"]
CONDITIONS = ["Resting, seated", "Small head movements", "Dim light", "Bright light", "After light exercise", "Other"]
MAX_AGO_S = 120.0
EST_COLS = (["t", "wall_iso", "status", "bpm", "confidence"] + METHODS + [f"{m}_quality" for m in METHODS]
            + ["face_fraction", "motion", "luma", "luma_var", "multi_face", "guidance"])
REF_COLS = ["id", "session_t_s", "bpm", "source", "entered_at_iso", "entered_session_t_s", "note", "edited", "deleted", "original_bpm", "original_t_s"]


class ValidationError(ValueError):
    """A request the validation recorder refuses; the message is shown to the user as is."""


def normalise_participant(text: str) -> str:
    """'Participant 001', 'p1', '001' -> 'P001'; other safe codes are kept as typed. Raises ValidationError for anything else."""
    s = str(text or "").strip()
    m = re.fullmatch(r"(?i)(?:participant|p)?[\s_-]*0*(\d{1,4})", s)
    if m:
        return f"P{int(m.group(1)):03d}"
    if ID_RE.match(s):
        return s
    raise ValidationError("Use an anonymous code such as P001 (letters, digits, - and _ only; no names).")


def default_prompts(duration_s: float) -> list[float]:
    """Times (s) at which the screen asks for a reference reading: first once the first estimate is possible, then every 15 s, not in the last 5 s."""
    out, t = [], 20.0
    while t <= duration_s - 5.0:
        out.append(t)
        t += 15.0
    return out


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


def _num(v):
    return "" if v is None or (isinstance(v, float) and v != v) else v


class Session:
    """In-memory state of one validation session (mirrors the files on disk)."""

    def __init__(self, directory: Path, meta: dict) -> None:
        self.dir = directory
        self.meta = meta
        self.entries: list[dict] = []
        self.rows: list[dict] = []
        self.next_id = 1
        self.report: dict | None = None
        self.csv_fh = None
        self.csv_wr = None


class ValidationManager:
    """Owns the (at most one) current validation session and everything on disk below ``root``."""

    def __init__(self, engine, root: Path | None = None) -> None:
        self.engine = engine
        self.root = Path(root) if root else REPO_ROOT / engine.cfg.get("app", {}).get("validation_dir", "validation_data")
        self.lock = threading.RLock()
        self.s: Session | None = None
        self.active = False
        self.watch: threading.Thread | None = None
        self.video = None

    # ------------------------------------------------------------------ helpers
    def _audit(self, action: str, **info) -> None:
        if self.s is None:
            return
        with open(self.s.dir / "audit.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": _iso(time.time()), "action": action, **info}, default=str) + "\n")

    def _save(self) -> None:
        """Rewrite session.json and reference_readings.csv from memory (small files, so rewriting is simplest and always consistent)."""
        s = self.s
        if s is None:
            return
        tmp = s.dir / "session.json.tmp"
        tmp.write_text(json.dumps({**s.meta, "entries": s.entries}, indent=1, default=str), encoding="utf-8")
        tmp.replace(s.dir / "session.json")
        with open(s.dir / "reference_readings.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=REF_COLS)
            wr.writeheader()
            for e in s.entries:
                wr.writerow({"id": e["id"], "session_t_s": round(e["t"], 2), "bpm": e["value"], "source": e["source"], "entered_at_iso": e["entered_at"],
                             "entered_session_t_s": _num(e.get("entered_t")), "note": e.get("note", ""), "edited": e.get("edited", False),
                             "deleted": e.get("deleted", False), "original_bpm": _num(e.get("original_value")), "original_t_s": _num(e.get("original_t"))})

    def _elapsed(self) -> float:
        t0 = self.engine.wall_t0
        return max(0.0, time.time() - t0) if t0 else 0.0

    # ------------------------------------------------------------------ start / stop
    def start(self, p: dict) -> dict:
        """Start a timed session. ``p``: participant, reference_device, device_note, condition, visit, duration_s, consent_confirmed, save_video, prompt_times, source."""
        with self.lock:
            if self.active:
                raise ValidationError("A session is already running. Stop it first.")
            if not p.get("consent_confirmed"):
                raise ValidationError("Please confirm that the participant has agreed to take part (see the study guide) before starting.")
            pid = normalise_participant(p.get("participant", ""))
            device = str(p.get("reference_device") or "").strip() or DEVICES[0]
            duration = float(p.get("duration_s") or 60.0)
            if not 20.0 <= duration <= 900.0:
                raise ValidationError("Session length must be between 20 s and 15 min.")
            tol = float(p.get("tolerance_s") or vm.DEFAULT_TOL_S)
            spec = dict(p.get("source") or {"type": "camera", "index": 0})
            sid = f"S{timestamp()}"
            pdir = self.root / pid
            visit = str(p.get("visit") or "").strip() or dt.date.today().isoformat()
            n_before = len(list(pdir.glob("S*"))) if pdir.exists() else 0
            meta = {"participant": pid, "session_id": sid, "reference_device": device, "device_note": str(p.get("device_note") or "")[:200],
                    "condition": str(p.get("condition") or CONDITIONS[0])[:80], "visit": visit[:40], "session_number_for_participant": n_before + 1,
                    "duration_s": duration, "tolerance_s": tol, "source_type": spec.get("type"), "save_video": bool(p.get("save_video")),
                    "prompt_times": [float(x) for x in (p.get("prompt_times") or default_prompts(duration))],
                    "consent_confirmed": True, "created": _iso(time.time()), "state": "starting", "end_reason": None,
                    "notes": "Anonymous code only. No names, no face recognition. Reference values are typed by the operator.",
                    "app_version": "session7"}
            d = pdir / sid
            d.mkdir(parents=True, exist_ok=True)
            res = self.engine.start(spec)
            if not res.get("ok"):
                try:
                    d.rmdir()
                except OSError:
                    pass
                raise ValidationError(res.get("message", "Could not start the source."))
            self.s = Session(d, meta)
            meta["source_label"] = self.engine.source_info.get("label")
            meta["true_bpm_simulated"] = self.engine.true_bpm if spec.get("type") == "demo" else None
            meta["min_seconds"] = self.engine.pipeline.qp.min_seconds if self.engine.pipeline else 8.0
            self.s.csv_fh = open(d / "estimates.csv", "w", newline="", encoding="utf-8")
            self.s.csv_wr = csv.DictWriter(self.s.csv_fh, fieldnames=EST_COLS, extrasaction="ignore")
            self.s.csv_wr.writeheader()
            if meta["save_video"]:
                self.video = cv2.VideoWriter(str(d / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
                self.engine.video_sink = _ResizeSink(self.video)
            meta["state"] = "recording"
            self.active = True
            self.engine.listeners.append(self._on_row)
            self._save()
            self._audit("start", **{k: meta[k] for k in ("participant", "reference_device", "condition", "visit", "duration_s", "source_type", "save_video")})
            self.watch = threading.Thread(target=self._watchdog, daemon=True, name="cardion-validation-watch")
            self.watch.start()
            return {"ok": True, "message": f"Session {sid} started for {pid}.", "session_id": sid, "participant": pid}

    def _on_row(self, row: dict) -> None:
        """Called by the engine for every one-per-second estimate."""
        with self.lock:
            s = self.s
            if s is None or not self.active:
                return
            r = dict(row)
            if "wall_t0" not in s.meta and self.engine.wall_t0:
                s.meta["wall_t0"] = self.engine.wall_t0                  # lets reference files with real timestamps be matched later
            s.rows.append(r)
            out = {k: _num(r.get(k)) for k in EST_COLS}
            out["wall_iso"] = _iso(r["wall"]) if r.get("wall") else ""
            s.csv_wr.writerow(out)
            s.csv_fh.flush()

    def _watchdog(self) -> None:
        """Ends the session when its planned length is reached, or when the source stops under it (camera unplugged, video ended)."""
        while True:
            time.sleep(0.5)
            with self.lock:
                if not self.active:
                    return
                s = self.s
                running = bool(self.engine.thread and self.engine.thread.is_alive())
                elapsed = self._elapsed()
                if elapsed >= s.meta["duration_s"] and self.engine.wall_t0:
                    reason = "completed"
                elif not running and (time.time() - (self.engine.wall_t0 or time.time()) > 2.0 or self.engine.error):
                    reason = "interrupted: " + (self.engine.error or self.engine.message or "the source stopped")
                else:
                    continue
            threading.Thread(target=self.stop, args=(reason,), daemon=True).start()
            return

    def stop(self, reason: str = "stopped by operator") -> dict:
        """End the session, stop the source (releases the camera), and write the report."""
        with self.lock:
            if not self.active or self.s is None:
                return {"ok": False, "message": "No session is running."}
            self.active = False
            s = self.s
            try:
                self.engine.listeners.remove(self._on_row)
            except ValueError:
                pass
            s.meta["state"] = "completed" if reason == "completed" else ("interrupted" if reason.startswith("interrupted") else "stopped")
            s.meta["end_reason"] = reason
            s.meta["ended"] = _iso(time.time())
            s.meta["ended_session_t_s"] = round(self._elapsed(), 2)
        self.engine.stop()                                             # outside the lock: it joins the source threads
        with self.lock:
            self.engine.video_sink = None
            if self.video is not None:
                self.video.release()
                self.video = None
            if s.csv_fh:
                s.csv_fh.close()
                s.csv_fh = None
            self._audit("stop", reason=reason, rows=len(s.rows))
            self._save()
            self.recompute()
            return {"ok": True, "message": f"Session ended ({reason}).", "report": self.s.report}

    # ------------------------------------------------------------------ reference entries
    def _session_time_limit(self) -> float:
        s = self.s
        return (s.meta.get("ended_session_t_s") if not self.active else self._elapsed()) or 0.0

    def add_reference(self, value, ago_s: float = 0.0, t_s=None, note: str = "") -> dict:
        """Add one reference reading. ``ago_s`` = it was shown on the monitor this many seconds ago; ``t_s`` = explicit session time of the reading."""
        with self.lock:
            s = self.s
            if s is None:
                raise ValidationError("Start a session first.")
            if not vm.valid_bpm(value):
                raise ValidationError(f"'{value}' is not a valid heart rate. Enter a number between {vm.MIN_BPM:.0f} and {vm.MAX_BPM:.0f}.")
            value = float(value)
            now_t = self._elapsed() if self.active else self._session_time_limit()
            if self.active and not self.engine.wall_t0:
                raise ValidationError("Wait until the camera is delivering pictures, then enter the reading.")
            if t_s not in (None, ""):
                try:
                    t = float(t_s)
                except (TypeError, ValueError):
                    raise ValidationError("The time must be a number of seconds from the start of the session.")
                source = "typed-time"
            else:
                try:
                    ago = float(ago_s or 0.0)
                except (TypeError, ValueError):
                    raise ValidationError("'Seconds ago' must be a number.")
                if not 0.0 <= ago <= MAX_AGO_S:
                    raise ValidationError(f"'Seconds ago' must be between 0 and {MAX_AGO_S:.0f}.")
                t = now_t - ago
                source = "live" if ago == 0 else "live-delayed"
            if t < -0.5:
                raise ValidationError("That time is before the session started.")
            if t > now_t + 5.0:
                raise ValidationError("That time is in the future.")
            e = {"id": s.next_id, "t": max(0.0, round(t, 2)), "value": value, "source": source, "entered_at": _iso(time.time()),
                 "entered_t": round(now_t, 2), "note": str(note or "")[:200], "edited": False, "deleted": False, "history": []}
            if source == "live-delayed":
                e["note"] = (e["note"] + f" (read {ago:.0f} s earlier)").strip()
            s.next_id += 1
            s.entries.append(e)
            self._audit("add_reference", id=e["id"], value=value, t=e["t"], source=source)
            self._save()
            if not self.active:
                self.recompute()
            return {"ok": True, "entry": e}

    def edit_reference(self, entry_id: int, value=None, t_s=None, note=None) -> dict:
        """Correct a value and/or its time; the original is kept in the entry's history and in the audit log."""
        with self.lock:
            e = self._entry(entry_id)
            changes = []
            if value not in (None, ""):
                if not vm.valid_bpm(value):
                    raise ValidationError(f"'{value}' is not a valid heart rate ({vm.MIN_BPM:.0f}-{vm.MAX_BPM:.0f}).")
                if float(value) != e["value"]:
                    e.setdefault("original_value", e["value"])
                    changes.append(("value", e["value"], float(value)))
                    e["value"] = float(value)
            if t_s not in (None, ""):
                try:
                    t = float(t_s)
                except (TypeError, ValueError):
                    raise ValidationError("The time must be a number of seconds from the start of the session.")
                if t < 0 or t > self._session_time_limit() + 5.0:
                    raise ValidationError("That time is outside the session.")
                if round(t, 2) != e["t"]:
                    e.setdefault("original_t", e["t"])
                    changes.append(("time_s", e["t"], round(t, 2)))
                    e["t"] = round(t, 2)
            if note is not None and note != e.get("note"):
                changes.append(("note", e.get("note", ""), str(note)[:200]))
                e["note"] = str(note)[:200]
            if not changes:
                return {"ok": True, "entry": e, "message": "Nothing changed."}
            e["edited"] = True
            for f, old, new in changes:
                e["history"].append({"at": _iso(time.time()), "field": f, "from": old, "to": new})
                self._audit("edit_reference", id=e["id"], field=f, old=old, new=new)
            self._save()
            if not self.active:
                self.recompute()
            return {"ok": True, "entry": e}

    def delete_reference(self, entry_id: int) -> dict:
        """Mark an entry deleted (it stays in the files, flagged, and is excluded from the maths)."""
        with self.lock:
            e = self._entry(entry_id)
            e["deleted"] = True
            e["history"].append({"at": _iso(time.time()), "field": "deleted", "from": False, "to": True})
            self._audit("delete_reference", id=e["id"], value=e["value"], t=e["t"])
            self._save()
            if not self.active:
                self.recompute()
            return {"ok": True}

    def _entry(self, entry_id) -> dict:
        s = self.s
        if s is None:
            raise ValidationError("No session.")
        for e in s.entries:
            if e["id"] == int(entry_id):
                return e
        raise ValidationError(f"There is no reference entry number {entry_id}.")

    # ------------------------------------------------------------------ importing reference data from a file
    def import_reference(self, path: str) -> dict:
        """Add reference values from a file: CSV (columns time_s or timestamp, and bpm) or an Apple Health ``export.xml``. Needs a session with a start time."""
        with self.lock:
            s = self.s
            if s is None:
                raise ValidationError("Start (or open) a session first; imported values are matched to its clock.")
            t0 = self.engine.wall_t0 if self.active else s.meta.get("wall_t0")
            if self.active and t0:
                s.meta["wall_t0"] = t0
            if not t0:
                raise ValidationError("This session has no recorded start time to match against.")
            tmax = (self._session_time_limit() if not self.active else self._elapsed()) + vm.DEFAULT_TOL_S
            p = Path(path)
            if not p.exists():
                raise ValidationError("The file could not be found.")
            samples = _read_reference_file(p, t0, tmax)
            added = skipped = 0
            have = {(round(e["t"], 1), e["value"]) for e in s.entries if not e.get("deleted")}
            for t, v in samples:
                if not vm.valid_bpm(v) or not (-0.5 <= t <= tmax) or (round(t, 1), float(v)) in have:
                    skipped += 1
                    continue
                s.entries.append({"id": s.next_id, "t": max(0.0, round(t, 2)), "value": float(v), "source": f"imported:{p.name}", "entered_at": _iso(time.time()),
                                  "entered_t": None, "note": "", "edited": False, "deleted": False, "history": []})
                s.next_id += 1
                added += 1
            self._audit("import_reference", file=p.name, added=added, skipped=skipped)
            self._save()
            if not self.active:
                self.recompute()
            return {"ok": True, "added": added, "skipped": skipped,
                    "message": f"Imported {added} reference reading(s) from {p.name}" + (f"; {skipped} outside the session or invalid were ignored." if skipped else ".")}

    # ------------------------------------------------------------------ reports
    def recompute(self) -> dict | None:
        """(Re)build the report and write pairs.csv / report.json / report.html / report.png into the session folder."""
        s = self.s
        if s is None:
            return None
        from .validation_report import write_report_files
        rep = vm.session_report(s.entries, s.rows, s.meta.get("tolerance_s", vm.DEFAULT_TOL_S), s.meta.get("min_seconds", 8.0))
        rep["headline"] = vm.headline(rep)
        rep["meta"] = {k: v for k, v in s.meta.items() if k != "notes"}
        rep["entries"] = s.entries
        s.report = rep
        try:
            write_report_files(s.dir, rep, s.rows)
        except Exception:  # noqa: BLE001 - the numbers are still available even if a plot fails
            log.exception("could not write the report files")
        return rep

    def load(self, participant: str, session_id: str) -> dict:
        """Re-open a finished session from disk (to add or correct reference readings afterwards, e.g. from the Health app)."""
        with self.lock:
            if self.active:
                raise ValidationError("Stop the running session first.")
            pid = normalise_participant(participant)
            if not re.fullmatch(r"S[0-9\-]+", session_id or ""):
                raise ValidationError("Unknown session.")
            d = self.root / pid / session_id
            f = d / "session.json"
            if not f.exists():
                raise ValidationError("Session not found.")
            data = json.loads(f.read_text(encoding="utf-8"))
            entries = data.pop("entries", [])
            s = Session(d, data)
            s.entries = entries
            s.next_id = max([e["id"] for e in entries], default=0) + 1
            with open(d / "estimates.csv", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    s.rows.append({k: _parse(v) for k, v in r.items()} | {"status": r["status"]})
            self.s = s
            self._audit("reopen")
            self.recompute()
            return {"ok": True, "message": f"Opened {pid}/{session_id}"}

    # ------------------------------------------------------------------ state for the page
    def snapshot(self) -> dict:
        """Everything the validation page needs to draw itself (small; polled about twice a second)."""
        with self.lock:
            s = self.s
            if s is None:
                return {"session": None, "active": False}
            elapsed = self._elapsed() if self.active else s.meta.get("ended_session_t_s", 0.0)
            tol = s.meta.get("tolerance_s", vm.DEFAULT_TOL_S)
            pairs, unpaired = vm.pair_readings(s.entries, s.rows, tol)
            live = []
            for e in vm.active_entries(s.entries):
                pr = next((p for p in pairs if p["entry_id"] == e["id"]), None)
                up = next((u for u in unpaired if u["entry_id"] == e["id"]), None)
                waiting = pr is None and self.active and elapsed < e["t"] + tol
                live.append({**{k: e[k] for k in ("id", "t", "value", "source", "edited")}, "pair": pr, "unpaired_reason": None if (pr or waiting) else (up or {}).get("reason"),
                             "waiting": waiting})
            prompts = []
            for pt in s.meta.get("prompt_times", []):
                answered = any((not e.get("deleted")) and e.get("entered_t") is not None and pt - 3 <= e["entered_t"] <= pt + 20 for e in s.entries)
                state = "done" if answered else "upcoming" if elapsed < pt else "due" if elapsed < pt + 15 else "missed"
                prompts.append({"t": pt, "state": state})
            rows = s.rows
            last = rows[-1] if rows else None
            av = vm.availability(rows, s.meta.get("min_seconds", 8.0)) if rows else {}
            hist = [{"t": r["t"], "bpm": r["bpm"] if r.get("status") == "ok" else None, "status": r.get("status")} for r in rows[-300:]]
            return {"session": {k: v for k, v in s.meta.items() if k != "notes"}, "active": self.active, "elapsed_s": round(elapsed, 1),
                    "duration_s": s.meta["duration_s"], "n_estimates": len(rows), "n_reliable": av.get("n_reliable", 0),
                    "reliable_share": av.get("reliable_share_whole_session"), "time_to_first_reliable_s": av.get("time_to_first_reliable_s"),
                    "last": ({"t": last["t"], "status": last.get("status"), "bpm": last.get("bpm")} if last else None),
                    "entries": live, "n_entries": len(vm.active_entries(s.entries)), "n_pairs": len(pairs), "prompts": prompts, "history": hist,
                    "all_entries": [{k: e.get(k) for k in ("id", "t", "value", "source", "edited", "deleted", "note")} for e in s.entries],
                    "report": s.report if not self.active else None, "folder": str(s.dir)}

    def participants(self) -> dict:
        """Existing anonymous codes with how many sessions each has, and the next free code."""
        out = []
        if self.root.exists():
            for d in sorted(self.root.glob("P*")):
                if d.is_dir():
                    out.append({"participant": d.name, "sessions": len(list(d.glob("S*")))})
        nums = [int(m.group(1)) for x in out if (m := re.fullmatch(r"P(\d+)", x["participant"]))]
        return {"participants": out, "next": f"P{(max(nums) + 1 if nums else 1):03d}", "devices": DEVICES, "conditions": CONDITIONS}

    def sessions(self) -> list[dict]:
        """All saved sessions (one line each) for the study overview."""
        out = []
        for f in sorted(self.root.glob("P*/S*/session.json")) if self.root.exists() else []:
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rp = f.parent / "report.json"
            rep = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
            out.append({"participant": m.get("participant"), "session_id": m.get("session_id"), "state": m.get("state"), "condition": m.get("condition"),
                        "reference_device": m.get("reference_device"), "visit": m.get("visit"), "source_type": m.get("source_type"),
                        "headline": rep.get("headline"), "n_pairs": (rep.get("accuracy") or {}).get("n_pairs", 0), "mae_bpm": (rep.get("accuracy") or {}).get("mae_bpm")})
        return out

    def study(self) -> dict:
        """Participant-level summary over every saved real-camera session (demo and video sessions are excluded); writes exports/ files."""
        from .validation_report import write_study_files
        sess, excluded = [], 0
        for f in sorted(self.root.glob("P*/S*/report.json")) if self.root.exists() else []:
            try:
                rep = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            meta = rep.get("meta", {})
            if meta.get("source_type") != "camera":
                excluded += 1
                continue
            sess.append({"participant": meta["participant"], "session_id": meta["session_id"], "condition": meta.get("condition"), "visit": meta.get("visit"),
                         "reference_device": meta.get("reference_device"), "pairs": rep.get("pairs", []),
                         "n_reference_entries": rep["reference_availability"]["n_reference_entries"],
                         "reliable_share_after_warmup": rep["cardion_availability"].get("reliable_share_after_warmup"),
                         "time_to_first_reliable_s": rep["cardion_availability"].get("time_to_first_reliable_s")})
        summ = vm.study_summary(sess)
        summ["excluded_non_camera_sessions"] = excluded
        by_cond: dict[str, list[dict]] = {}
        for x in sess:
            by_cond.setdefault(x["condition"] or "unspecified", []).append(x)
        summ["by_condition"] = {c: {k: v for k, v in vm.study_summary(xs).items() if k in ("n_sessions", "n_participants", "n_pairs", "paired_share", "participant_mean_of_mae",
                                                                                         "mean_reliable_share_after_warmup")} for c, xs in by_cond.items()}
        write_study_files(self.root, summ, sess)
        return summ


class _ResizeSink:
    """Writes preview frames at a fixed size (VideoWriter needs constant dimensions)."""

    def __init__(self, writer) -> None:
        self.w = writer

    def write(self, bgr) -> None:
        self.w.write(cv2.resize(bgr, (640, 480)) if bgr.shape[1] != 640 or bgr.shape[0] != 480 else bgr)


def _parse(v: str):
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


# ---------------------------------------------------------------------- reference files
def _parse_time(text: str) -> float | None:
    """Wall-clock seconds from an ISO-like or Apple Health date string; naive values are taken as local time."""
    t = str(text).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M"):
        try:
            d = dt.datetime.strptime(t.replace("Z", "+0000"), fmt)
            return d.timestamp() if d.tzinfo else d.replace(tzinfo=None).timestamp()
        except ValueError:
            continue
    return None


def _read_reference_file(p: Path, t0: float, tmax: float) -> list[tuple[float, float]]:
    """Return ``[(session_seconds, bpm)]`` from a CSV or an Apple Health export.xml (only records near the session are kept)."""
    if p.suffix.lower() == ".xml":
        return _read_apple_health(p, t0, tmax)
    if p.suffix.lower() not in (".csv", ".txt"):
        raise ValidationError("Supported reference files: .csv (columns: time_s or timestamp, and bpm) and Apple Health export.xml.")
    out = []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
        bcol = next((cols[c] for c in ("bpm", "heart_rate", "hr", "value", "heartrate") if c in cols), None)
        tcol = next((cols[c] for c in ("time_s", "t", "seconds", "session_t_s") if c in cols), None)
        wcol = next((cols[c] for c in ("timestamp", "datetime", "time", "date", "start", "startdate") if c in cols), None)
        if bcol is None or (tcol is None and wcol is None):
            raise ValidationError("The CSV needs a 'bpm' column and either 'time_s' (seconds from session start) or 'timestamp' (date and time).")
        for r in rd:
            try:
                v = float(r[bcol])
            except (TypeError, ValueError):
                continue
            if tcol is not None:
                try:
                    out.append((float(r[tcol]), v))
                except (TypeError, ValueError):
                    continue
            else:
                w = _parse_time(r[wcol])
                if w is not None:
                    out.append((w - t0, v))
    return out


def _read_apple_health(p: Path, t0: float, tmax: float) -> list[tuple[float, float]]:
    """Stream an Apple Health export.xml (can be gigabytes) and keep heart-rate records that fall inside the session window."""
    out = []
    for _, el in ET.iterparse(str(p), events=("end",)):
        if el.tag == "Record" and el.attrib.get("type") == "HKQuantityTypeIdentifierHeartRate":
            w = _parse_time(el.attrib.get("endDate") or el.attrib.get("startDate", ""))
            if w is not None and -2.0 <= w - t0 <= tmax:
                try:
                    out.append((w - t0, float(el.attrib["value"])))
                except (KeyError, ValueError):
                    pass
        if el.tag in ("Record", "Workout", "ActivitySummary"):
            el.clear()
    return out
