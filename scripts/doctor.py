"""CARDION doctor: diagnose the installation, repair routine problems automatically, and write a complete log.

Uses only the Python standard library at start-up, so it still runs when the scientific packages are broken. Each package is
imported in its own child process, so a crashing DLL or a full traceback is captured instead of killing the doctor.

    python scripts/doctor.py                    # diagnose only
    python scripts/doctor.py --repair           # diagnose, fix what is safe to fix, re-check
    python scripts/doctor.py --repair --deep    # also loads the models and starts the face tracker

Exit codes: 0 = healthy (or repaired), 1 = problems remain (see the log), 10 = the virtual environment is beyond repair and the
launcher should re-create it. Everything (full tracebacks, command output) is written to ``logs/doctor_<time>.log``; a readable
summary goes to ``out/doctor_report.md`` and ``out/doctor_report.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXIT_OK, EXIT_PROBLEMS, EXIT_RECREATE, EXIT_RELOCATE = 0, 1, 10, 11     # 11 = the install path is too long: rebuild the environment at a short path
MAX_ENV_PATH = 240       # Windows MAX_PATH is 260; DLL loading (LoadLibrary) fails above it even when "long paths" are enabled for other programs
LONG_PATH_MARKERS = ("winerror 206", "filename or extension is too long", "path too long")


def is_long_path_error(text: str, prefix_len: int = 0) -> bool:
    """True if an import error is caused by Windows path-length limits (WinError 206, or a DLL that failed to load from a very deep folder)."""
    low = text.lower()
    if any(m in low for m in LONG_PATH_MARKERS):
        return True
    return "dll load failed" in low and prefix_len > 100

# import name -> pip distribution name (as pinned in requirements-app.txt)
IMPORTS = {"numpy": "numpy", "scipy": "scipy", "cv2": "opencv-contrib-python", "mediapipe": "mediapipe", "torch": "torch",
           "snntorch": "snntorch", "yaml": "pyyaml", "matplotlib": "matplotlib"}
PROJECT_MODULES = ["src.common", "src.rppg_classical", "src.dvs_emulate", "src.snn_model", "src.ann_model", "src.cardion_app.quality",
                   "src.cardion_app.sources", "src.cardion_app.pipeline", "src.cardion_app.engine", "src.cardion_app.server",
                   "src.cardion_app.selftest"]
REQUIRED_FILES = ["configs/default.yaml", "configs/quality_calibrated.json", "models/snn.pt", "models/ann.pt", "models/models.json",
                  "models/face_landmarker.task", "src/cardion_app/static/index.html", "src/cardion_app/static/app.js",
                  "src/cardion_app/static/style.css", "requirements-app.txt"]
WRITABLE_DIRS = ["out", "sessions", "logs"]
LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class Doctor:
    """Runs checks, records results, and appends everything to a log file."""

    def __init__(self, root: Path, repair: bool, deep: bool, uv: str | None, allow_recreate: bool, log_path: Path | None = None, quick: bool = False) -> None:
        self.root, self.repair, self.deep, self.uv, self.allow_recreate = root, repair, deep, uv, allow_recreate
        self.results: list[dict] = []
        self.actions: list[str] = []
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        (root / "logs").mkdir(exist_ok=True)
        self.log_path = log_path or root / "logs" / ("doctor_quick.log" if quick else f"doctor_{stamp}.log")      # quick runs overwrite one file
        self.log_fh = open(self.log_path, "w" if quick else "a", encoding="utf-8", errors="replace")
        self.py = sys.executable
        self.env = {**os.environ, "PYTHONPATH": str(root), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    # ------------------------------------------------------------------ logging
    def log(self, text: str) -> None:
        self.log_fh.write(text.rstrip("\n") + "\n")
        self.log_fh.flush()

    def record(self, name: str, status: str, detail: str, fixed: bool = False) -> None:
        """Store one result (status PASS/WARN/FAIL) and echo it."""
        self.results.append({"check": name, "status": status, "detail": detail, "auto_repaired": fixed})
        tag = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[status] + (" (repaired)" if fixed else "")
        line = f"[{tag}] {name}: {detail.splitlines()[0] if detail else ''}"
        print(line, flush=True)
        self.log(line)
        if "\n" in detail:
            self.log(detail)

    def run(self, cmd: list[str], timeout: float = 600, label: str = "", cwd: Path | None = None, live: bool = False) -> tuple[int, str]:
        """Run a command, capture merged stdout+stderr, log it fully, return (exit code, output).

        ``live=True`` also echoes progress lines to the console (used for long installs so the window is not silent).
        """
        self.log(f"$ {' '.join(map(str, cmd))}   [{label}]")
        lines: list[str] = []
        try:
            p = subprocess.Popen([str(c) for c in cmd], cwd=cwd or self.root, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            timer = threading.Timer(timeout, p.kill)
            timer.start()
            try:
                for line in p.stdout:                                  # type: ignore[union-attr]
                    lines.append(line)
                    if live and line.strip():
                        print("    " + line.rstrip()[:160], flush=True)
                p.wait()
            finally:
                timer.cancel()
            code = p.returncode
        except OSError as e:
            lines.append(f"{type(e).__name__}: {e}\n")
            code = -2
        out = "".join(lines)
        self.log(f"exit={code}\n{out}")
        return code, out
    # ------------------------------------------------------------------ checks
    def check_python(self) -> None:
        v = sys.version_info
        ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
        detail = f"Python {v.major}.{v.minor}.{v.micro} at {self.py}"
        if ok:
            self.record("python version", "PASS", detail)
        else:
            self.record("python version", "FAIL", detail + " - CARDION needs Python 3.10-3.12 (the launcher installs 3.11 itself).")
            self.need_recreate = True

    def check_paths(self) -> None:
        n = len(str(self.root))
        long_ok = True
        if sys.platform.startswith("win"):
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                    long_ok = bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
            except OSError:
                long_ok = False
        env_n = len(sys.prefix)
        longest = self.longest_env_path()
        if longest > MAX_ENV_PATH:
            self.record("environment path length", "FAIL", f"The Python environment ({sys.prefix}, {env_n} characters) contains files with paths up to {longest} "
                        f"characters; Windows cannot load DLLs above {MAX_ENV_PATH + 20} (WinError 206). The environment must live in a shorter folder.")
            self.relocate = True
        else:
            self.record("environment path length", "PASS", f"environment folder {env_n} characters; longest file path {longest} (limit about 260)")
        if n > 100 and not long_ok:
            self.record("folder path length", "WARN", f"The CARDION folder path is {n} characters long. CARDION keeps its Python environment in a short folder "
                        "automatically, but a short project folder such as C:\\CARDION is more robust.")
        else:
            self.record("folder path length", "PASS", f"{n} characters; long paths {'enabled' if long_ok else 'not needed'}")
        if os.environ.get("ONEDRIVE") and "OneDrive" in str(self.root):
            self.record("folder location", "WARN", "The folder is inside OneDrive; syncing can lock files. A plain folder such as C:\\CARDION is safer.")

    @staticmethod
    def longest_env_path() -> int:
        """Length of the longest file path inside the running interpreter's environment."""
        best = 0
        for base, _dirs, files in os.walk(sys.prefix):
            if files:
                best = max(best, len(base) + 1 + max(len(f) for f in files))
        return best

    def check_files(self) -> None:
        for d in WRITABLE_DIRS:
            p = self.root / d
            try:
                p.mkdir(exist_ok=True)
                t = p / f".write_test_{os.getpid()}"
                t.write_text("x")
                t.unlink()
                self.record(f"folder writable: {d}/", "PASS", str(p))
            except OSError as e:
                self.record(f"folder writable: {d}/", "FAIL", f"{e}. Choose a folder you may write to (not Program Files).")
        for rel in REQUIRED_FILES:
            p = self.root / rel
            if p.is_file() and p.stat().st_size > 0:
                if rel.endswith(".json"):
                    try:
                        json.loads(p.read_text(encoding="utf-8-sig"))
                    except ValueError as e:
                        self.record(f"file {rel}", "FAIL", f"not valid JSON: {e}")
                        continue
                self.record(f"file {rel}", "PASS", f"{p.stat().st_size} bytes")
                continue
            fixed = False
            if self.repair and rel == "models/face_landmarker.task":
                fixed = self.download(LANDMARKER_URL, p)
            if fixed:
                self.record(f"file {rel}", "PASS", "was missing; downloaded again from Google's MediaPipe model storage", fixed=True)
            else:
                self.record(f"file {rel}", "FAIL", "missing or empty. Re-extract CARDION.zip (nothing else was changed), or restore this file from the "
                            "original download. Trained models cannot be regenerated automatically.")

    def download(self, url: str, dest: Path) -> bool:
        """Download ``url`` to ``dest`` (atomic); returns success. Logs the error."""
        try:
            import urllib.request
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh)
            if tmp.stat().st_size < 100_000:
                raise OSError("downloaded file is suspiciously small")
            tmp.replace(dest)
            self.actions.append(f"downloaded {dest.name}")
            return True
        except Exception:  # noqa: BLE001
            self.log("download failed:\n" + traceback.format_exc())
            return False

    def import_status(self) -> dict[str, str]:
        """Import every required package in its own process; returns {module: error text or ''}."""
        out = {}
        for mod in IMPORTS:
            t0 = time.time()
            code, txt = self.run([self.py, "-I", "-c", f"import {mod}"], timeout=180, label=f"import {mod}")
            out[mod] = "" if code == 0 else (txt.strip() or f"exit code {code}")
            self.log(f"   import {mod}: {'ok' if code == 0 else 'FAILED'} in {time.time() - t0:.1f}s")
        return out

    def check_imports(self) -> None:
        bad = self.import_status()
        if self.relocate or any(is_long_path_error(e, len(sys.prefix)) for e in bad.values()):
            self.relocate = True
            self.log("Import errors look like Windows path-length problems; NOT reinstalling packages in this folder (that cannot help).")
        elif any(bad.values()) and self.repair:
            bad = self.repair_packages(bad)
        for mod, err in bad.items():
            if err:
                self.record(f"import {mod}", "FAIL", f"{mod} cannot be imported:\n{err}")
            else:
                self.record(f"import {mod}", "PASS", self.version_of(mod))
        if any(bad.values()):
            self.need_recreate = True

    def version_of(self, mod: str) -> str:
        try:
            from importlib import metadata
            return f"version {metadata.version(IMPORTS[mod])}"
        except Exception:  # noqa: BLE001
            return "imports"

    # ------------------------------------------------------------------ repairs
    def installers(self) -> list[tuple[str, list[str]]]:
        """Installer command prefixes for the current interpreter, best first: uv (fast), then pip (works if uv is blocked or absent)."""
        out: list[tuple[str, list[str]]] = []
        if self.uv and Path(self.uv).exists():
            out.append(("uv", [self.uv, "pip", "install", "--python", self.py]))
        code, _ = self.run([self.py, "-m", "pip", "--version"], timeout=60, label="pip available?")
        if code == 0:
            out.append(("pip", [self.py, "-m", "pip", "install", "--disable-pip-version-check"]))
        return out

    def install(self, args_uv: list[str], args_pip: list[str], label: str) -> bool:
        """Try each installer until one exits with 0. Output is shown live and logged."""
        for name, prefix in self.installers():
            code, _ = self.run(prefix + (args_uv if name == "uv" else args_pip), timeout=3600, label=f"{label} via {name}", live=True)
            if code == 0:
                return True
            print(f"[repair] {name} could not finish ({label}); see the log. Trying the next installer if there is one ...", flush=True)
        return False

    def repair_packages(self, bad: dict[str, str]) -> dict[str, str]:
        """Complete the pinned installation (idempotent: only missing/broken parts are fetched), then force-reinstall what still fails."""
        req = str(self.root / "requirements-app.txt")
        state_f = self.root / "logs" / "repair_state.json"
        signature = "|".join(f"{m}:{(e.strip().splitlines() or [''])[-1][:120]}" for m, e in sorted(bad.items()) if e)
        try:
            state = json.loads(state_f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("signature") == signature and state.get("attempts", 0) >= 2:
            print("[repair] the same package error has already survived two repair attempts; reinstalling again would not help. See the log.", flush=True)
            self.log(f"repair skipped: signature {signature!r} already failed {state['attempts']} repair attempts")
            return bad
        state_f.write_text(json.dumps({"signature": signature, "attempts": state.get("attempts", 0) + 1 if state.get("signature") == signature else 1}), encoding="utf-8")
        print("[repair] completing the package installation (needs internet; first time: 5-15 minutes) ...", flush=True)
        if not self.installers():
            self.log("no installer (uv or pip) available; cannot repair packages")
            return bad
        self.install(["-r", req], ["-r", req], "complete install")
        self.actions.append("re-ran the pinned package installation")
        bad = self.import_status()
        for mod in IMPORTS:                                    # dependency order: numpy first, so one root cause is not "fixed" six times
            if not bad.get(mod):
                continue
            if not self.import_ok(mod):
                self.reinstall_one(mod, clean_cache=False)
            if not self.import_ok(mod):
                self.reinstall_one(mod, clean_cache=True)      # a corrupted download cache would otherwise restore the same broken files
        bad = self.import_status()
        if not any(bad.values()):
            try:
                state_f.unlink()
            except OSError:
                pass
        elif any(is_long_path_error(e, len(sys.prefix)) for e in bad.values()):
            self.relocate = True
        return bad

    def import_ok(self, mod: str) -> bool:
        return self.run([self.py, "-I", "-c", f"import {mod}"], timeout=180, label=f"re-check import {mod}")[0] == 0

    def reinstall_one(self, mod: str, clean_cache: bool) -> None:
        dist = IMPORTS[mod]
        pin = self.pin_of(dist) or dist
        req = str(self.root / "requirements-app.txt")
        print(f"[repair] force-reinstalling {pin}{' (after clearing its download cache)' if clean_cache else ''} ...", flush=True)
        if clean_cache and self.uv and Path(self.uv).exists():
            self.run([self.uv, "cache", "clean", dist], timeout=300, label=f"clear uv cache for {dist}")
        self.install(["-r", req, "--reinstall-package", dist], ["--force-reinstall", "--no-deps", "--no-cache-dir", pin], f"reinstall {dist}")
        self.actions.append(f"force-reinstalled {pin}" + (" after clearing its cache" if clean_cache else ""))
    def pin_of(self, dist: str) -> str | None:
        for line in (self.root / "requirements-app.txt").read_text(encoding="utf-8-sig").splitlines():
            if line.strip().lower().startswith(dist.lower() + "=="):
                return line.strip()
        return None

    # ------------------------------------------------------------------ project checks
    def check_project(self) -> None:
        for mod in PROJECT_MODULES:
            code, txt = self.run([self.py, "-c", f"import {mod}"], timeout=240, label=f"import {mod}")
            if code == 0:
                self.record(f"project module {mod}", "PASS", "imports")
            else:
                self.record(f"project module {mod}", "FAIL", f"cannot import {mod}:\n{txt.strip()}")
        code, txt = self.run([self.py, "-c", "from src.common import load_config; c=load_config(); print(sorted(c)[:5])"], label="config")
        self.record("configuration", "PASS" if code == 0 else "FAIL", "configs/default.yaml parsed" if code == 0 else txt.strip())

    def check_deep(self) -> None:
        code = ("from src.common import load_config\nfrom src.cardion_app.engine import load_models\n"
                "m, info = load_models(load_config())\nprint(info)\nassert info['snn']['loaded'] and info['ann']['loaded'], info\n")
        c, txt = self.run([self.py, "-c", code], timeout=240, label="load models")
        self.record("models load (SNN, ANN)", "PASS" if c == 0 else "FAIL", "both checkpoints loaded" if c == 0 else txt.strip())
        code = ("import numpy as np\nfrom src.common import load_config\nfrom src.cardion_app.pipeline import Pipeline\n"
                "p = Pipeline(load_config(), models=None, use_landmarks=True)\nf = (np.random.rand(240,320,3)*255).astype('uint8')\n"
                "p.push(f, 0.0, None)\nprint('face tracker ok')\n")
        c, txt = self.run([self.py, "-c", code], timeout=240, label="face tracker")
        self.record("face tracker starts", "PASS" if c == 0 else "FAIL", "MediaPipe face tracker initialised" if c == 0 else txt.strip())

    def check_port(self) -> None:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 8765))
            self.record("port 8765", "PASS", "free")
        except OSError:
            self.record("port 8765", "WARN", "in use (another CARDION window?); the app will pick the next free port automatically")
        finally:
            s.close()

    def check_camera_permission_hint(self) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam") as k:
                val = winreg.QueryValueEx(k, "Value")[0]
            if str(val).lower() == "deny":
                self.record("Windows camera permission", "WARN", "Camera access is switched OFF for this user. Windows Settings > Privacy & security > "
                            "Camera: turn on 'Camera access' and 'Let desktop apps access your camera'. (CARDION never changes this setting.)")
            else:
                self.record("Windows camera permission", "PASS", f"camera access setting: {val}")
        except OSError:
            self.record("Windows camera permission", "PASS", "setting not present (default allowed)")

    # ------------------------------------------------------------------ quick check and full test
    def quick(self) -> int:
        """Fast health check used on every launch (a few seconds): required files exist and all packages import in ONE process."""
        missing = [r for r in REQUIRED_FILES if not (self.root / r).is_file()]
        code, txt = self.run([self.py, "-I", "-c", "import " + ", ".join(IMPORTS)], timeout=180, label="quick import check")
        if missing:
            print(f"Quick check: missing files: {missing}", flush=True)
        if code != 0:
            first = (txt.strip().splitlines() or ["(no output)"])[-1]
            print(f"Quick check: a package cannot be imported: {first}", flush=True)
            if is_long_path_error(txt, len(sys.prefix)):
                print("Quick check: this is a Windows path-length problem (the environment folder is too deep).", flush=True)
                return EXIT_RELOCATE
        return EXIT_OK if (code == 0 and not missing) else EXIT_PROBLEMS

    def full(self, fails_before: int) -> int:
        """Doctor + self-test + camera tests (mock, then real hardware if a webcam exists); writes out/test_and_repair_report.md."""
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.env["CARDION_RUN_ID"] = run_id
        started = time.time()
        sections: list[tuple[str, int, str, str]] = []           # title, exit code, report body, verdict
        steps = [("Application self-test (server, demo, video, recording, models)", [self.py, "-m", "src.cardion_app", "--selftest"], 1800, "out/selftest_report.md"),
                 ("Camera pipeline test with a recorded real face (no hardware)", [self.py, "-m", "src.cardion_app", "--cameratest", "--mock", "--seconds", "30"], 600, "out/camera_test_latest.md"),
                 ("Real webcam test (your camera light will turn on for ~20 s)", [self.py, "-m", "src.cardion_app", "--cameratest", "--seconds", "20"], 600, "out/camera_test_latest.md")]
        for title, cmd, tmo, rep in steps:
            print(f"\n=== {title} ===", flush=True)
            report = self.root / rep
            for stale in (report, report.with_suffix(".json")):          # an older report must never be shown as this run's result
                try:
                    stale.unlink()
                except OSError:
                    pass
            code, out = self.run(cmd, timeout=tmo, label=title, live=True)
            fresh = report.exists() and report.stat().st_mtime >= started
            if fresh:
                body = report.read_text(encoding="utf-8", errors="replace")
                verdict = "PASS" if code == 0 else "FAIL"
            else:
                body = ("NO REPORT WAS PRODUCED BY THIS RUN - the test crashed before it could write one. Last output (full text in "
                        f"{self.log_path}):\n\n```\n{out[-3000:]}\n```")
                verdict = "FAIL (crashed, no report)"
            sections.append((title, code, body, verdict))
        ok = fails_before == 0 and all(c == 0 for _, c, _, _ in sections)
        md = [f"# CARDION test and repair report - run {run_id}", "", f"Started {dt.datetime.fromtimestamp(started):%Y-%m-%d %H:%M:%S}, finished {dt.datetime.now():%H:%M:%S}. "
              "Everything below was produced by THIS run; older results are deleted before each test starts.", "",
              f"## Overall: {'ALL PASSED' if ok else 'PROBLEMS FOUND'}", "",
              f"- Installation check: {'PASS' if fails_before == 0 else 'FAIL'} (details: out/doctor_report.md)"]
        md += [f"- {t}: {v}" for t, _, _, v in sections]
        md += ["", "Failures above are software failures. A missing webcam, a webcam Windows blocks, no face in view, or too little pulse data are WARNINGS "
               "inside the camera section and never count as failures.",
               "Camera-pulse numbers are only meaningful when you sit still, face the camera in good light. Not a medical device.", ""]
        for t, _, body, _ in sections:
            md += [f"## {t}", "", body, ""]
        text = "\n".join(md)
        (self.root / "out" / f"test_and_repair_{run_id}.md").write_text(text, encoding="utf-8")
        (self.root / "out" / "test_and_repair_report.md").write_text(text, encoding="utf-8")
        print(f"\nOverall: {'ALL PASSED' if ok else 'PROBLEMS FOUND'}. Report: {self.root / 'out' / 'test_and_repair_report.md'}", flush=True)
        return EXIT_OK if ok else EXIT_PROBLEMS
    # ------------------------------------------------------------------ driver
    need_recreate = False
    relocate = False

    def run_all(self) -> int:
        self.log(f"CARDION doctor {dt.datetime.now().isoformat(timespec='seconds')}  root={self.root}  python={self.py}  repair={self.repair} deep={self.deep}")
        self.log(f"platform={sys.platform} version={sys.version}")
        for step in (self.check_python, self.check_paths, self.check_files, self.check_imports):
            try:
                step()
            except Exception:  # noqa: BLE001
                self.record(step.__name__, "FAIL", "the check itself crashed:\n" + traceback.format_exc())
        if not any(r["status"] == "FAIL" and r["check"].startswith("import ") for r in self.results):
            for step in (self.check_project, self.check_port, self.check_camera_permission_hint) + ((self.check_deep,) if self.deep else ()):
                try:
                    step()
                except Exception:  # noqa: BLE001
                    self.record(step.__name__, "FAIL", "the check itself crashed:\n" + traceback.format_exc())
        fails = [r for r in self.results if r["status"] == "FAIL"]
        self.write_report(fails)
        if not fails:
            return EXIT_OK
        if self.relocate and self.allow_recreate:
            return EXIT_RELOCATE
        if self.need_recreate and self.allow_recreate and any(r["check"].startswith(("import ", "python")) for r in fails):
            return EXIT_RECREATE
        return EXIT_PROBLEMS

    def write_report(self, fails: list[dict]) -> None:
        passed = sum(r["status"] == "PASS" for r in self.results)
        warns = sum(r["status"] == "WARN" for r in self.results)
        md = [f"# CARDION doctor report {dt.datetime.now():%Y-%m-%d %H:%M:%S}", "",
              f"**{'HEALTHY' if not fails else 'PROBLEMS FOUND'}** - {passed} passed, {warns} warnings, {len(fails)} failed", ""]
        if self.actions:
            md += ["Automatic repairs performed: " + "; ".join(self.actions), ""]
        for r in self.results:
            first = r["detail"].splitlines()[0] if r["detail"] else ""
            md.append(f"- {r['status']}{' (auto-repaired)' if r['auto_repaired'] else ''} - {r['check']}: {first}")
        md += ["", f"Full technical log (tracebacks, command output): `{self.log_path}`"]
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        (out / "doctor_report.md").write_text("\n".join(md), encoding="utf-8")
        (out / "doctor_report.json").write_text(json.dumps({"passed": passed, "warnings": warns, "failed": len(fails), "actions": self.actions,
                                                            "results": self.results, "log": str(self.log_path)}, indent=2), encoding="utf-8")
        print(f"\nDoctor: {passed} passed, {warns} warnings, {len(fails)} failed. Log: {self.log_path}", flush=True)
        for r in fails:
            print(f"  FAILED: {r['check']}: {r['detail'].splitlines()[0]}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--repair", action="store_true", help="fix routine problems automatically")
    ap.add_argument("--deep", action="store_true", help="also load the models and start the face tracker")
    ap.add_argument("--uv", default=None, help="path to uv.exe (preferred installer)")
    ap.add_argument("--no-recreate", action="store_true", help="never ask the launcher to rebuild the environment")
    ap.add_argument("--quick", action="store_true", help="fast health check for every launch (exit 0/1)")
    ap.add_argument("--full", action="store_true", help="after the checks also run the self-test and camera tests and write out/test_and_repair_report.md")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.full:                                      # a previous run's report must never be mistaken for this run's
        try:
            (Path(a.root) / "out" / "test_and_repair_report.md").unlink()
        except OSError:
            pass
    d = Doctor(Path(a.root).resolve(), a.repair, a.deep, a.uv, not a.no_recreate, Path(a.log) if a.log else None, quick=a.quick)
    try:
        if a.quick:
            sys.exit(d.quick())
        code = d.run_all()
        if a.full and code in (EXIT_OK, EXIT_PROBLEMS):
            code = d.full(0 if code == EXIT_OK else 1)
    except Exception:  # noqa: BLE001
        d.log("DOCTOR CRASHED\n" + traceback.format_exc())
        print(f"The doctor itself crashed. Details: {d.log_path}", flush=True)
        code = EXIT_PROBLEMS
    sys.exit(code)


if __name__ == "__main__":
    main()
