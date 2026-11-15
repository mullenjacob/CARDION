"""Regression tests for the Windows start-up failure ("[CARDION] Problem: Traceback (most recent call last):") and the repair system.

Fast tests run everywhere. Tests marked ``windows`` need Windows PowerShell; the ``slow`` test builds a disposable copy of the project in
a folder whose name contains a space, creates a private environment there, sabotages a package, and runs the real ``Launch CARDION.bat``
(set ``CARDION_SLOW_TESTS=1``; needs the uv cache or internet; takes a few minutes).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
WIN = sys.platform.startswith("win")


# ---------------------------------------------------------------------------------------------- the root cause, reproduced
@pytest.mark.skipif(not WIN, reason="Windows PowerShell 5.1 behaviour")
def test_old_launcher_pattern_hides_the_traceback(tmp_path: Path) -> None:
    """Documents the bug: under ``$ErrorActionPreference='Stop'`` a redirected native stderr becomes a terminating error whose message is
    only the first stderr line."""
    ps = tmp_path / "old.ps1"
    ps.write_text(textwrap.dedent(f"""
        $ErrorActionPreference = "Stop"
        try {{ & "{sys.executable}" -c "import notamodule_xyz" 2>$null; "no error" }} catch {{ "CAUGHT: [" + $_.Exception.Message + "]" }}
    """), encoding="ascii")
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)], capture_output=True, text=True).stdout
    assert "CAUGHT: [Traceback (most recent call last):]" in out, out


def test_launcher_never_redirects_native_stderr() -> None:
    """The fixed launcher must not use PowerShell stream redirection on programs, must not stop on stderr, and must stay ASCII."""
    src = (ROOT / "scripts" / "launch.ps1").read_text(encoding="ascii")
    code = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
    assert not re.search(r"\s2>\s*(\$null|&1)", code), "native stderr redirection found in launch.ps1"
    assert '$ErrorActionPreference = "Continue"' in src
    assert "Start-Process" in src and "-RedirectStandardError" in src          # captured through files, not through PowerShell streams
    assert "launcher_error_" in src                                              # a failure always writes the full error to a log file


def test_bat_files_report_log_location() -> None:
    for name in ("Launch CARDION.bat", "Run CARDION Self-Test.bat", "Test and Repair CARDION.bat"):
        text = (ROOT / name).read_text(encoding="ascii")
        assert "scripts\\launch.ps1" in text and "pause" in text.lower(), name
    assert "logs" in (ROOT / "Launch CARDION.bat").read_text(encoding="ascii")


# ---------------------------------------------------------------------------------------------- doctor
def run_doctor(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "doctor.py"), "--root", str(root), *args], capture_output=True, text=True,
                          timeout=900, env={**os.environ, "PYTHONUTF8": "1"})


def test_doctor_quick_passes_on_healthy_install() -> None:
    if not (ROOT / "models" / "face_landmarker.task").exists():
        pytest.skip("face-landmarker model not downloaded yet (it is not redistributed; the launcher fetches it on first run)")
    r = run_doctor(ROOT, "--quick")
    assert r.returncode == 0, r.stdout + r.stderr


def test_doctor_reports_unrepairable_problem_with_full_log(tmp_path: Path) -> None:
    """A project folder without its model files: the doctor must FAIL, name the file, and keep a complete log (no truncated traceback)."""
    for sub in ("configs", "src/cardion_app/static", "models"):
        (tmp_path / sub).mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "default.yaml", tmp_path / "configs")
    shutil.copy(ROOT / "requirements-app.txt", tmp_path)
    r = run_doctor(tmp_path, "--repair", "--no-recreate")
    assert r.returncode == 1, r.stdout
    report = json.loads((tmp_path / "out" / "doctor_report.json").read_text(encoding="utf-8"))
    failed = {x["check"] for x in report["results"] if x["status"] == "FAIL"}
    assert "file models/snn.pt" in failed and "file models/models.json" in failed
    log = Path(report["log"]).read_text(encoding="utf-8")
    assert "models/snn.pt" in log and "Traceback" in log          # project-module import failures keep their whole traceback
    assert "ModuleNotFoundError" in log


def test_doctor_creates_missing_folders(tmp_path: Path) -> None:
    run_doctor(tmp_path, "--no-recreate")
    for d in ("out", "sessions", "logs"):
        assert (tmp_path / d).is_dir()


# ---------------------------------------------------------------------------------------------- the crash found by the real-camera test
def test_pipeline_survives_a_camera_that_never_sees_a_face() -> None:
    """Real webcam, nobody in front of it: the pipeline used to raise IndexError (grid_t0 = tt[0]); it must report 'no face' instead."""
    from src.cardion_app.pipeline import Pipeline
    from src.common import load_config
    pl = Pipeline(load_config(), models=None, use_landmarks=True)
    rng = np.random.default_rng(0)
    a = None
    for i in range(330):
        pl.push((rng.random((240, 320, 3)) * 60 + 80).astype("uint8"), i / 30.0)
        if i % 30 == 29 and pl.seconds() > 2:
            a = pl.estimate()
    pl.close()
    assert a is not None and a.status == "no_face" and a.bpm is None


def test_camera_test_runs_on_a_mock_camera() -> None:
    """The whole camera test (open, frames, face, pulse pipeline, report) on a recorded face - skipped if the sample video is absent."""
    from src.cardion_app.cameratest import default_mock_video, run_camera_test
    from src.common import load_config
    if default_mock_video() is None:
        pytest.skip("no sample face video")
    assert run_camera_test(load_config(), seconds=6.0, mock=True) == 0
    res = json.loads((ROOT / "out" / "camera_test_latest.json").read_text(encoding="utf-8"))
    status = {s["step"]: s["status"] for s in res["steps"]}
    assert status["Camera 0: frames"] == "PASS" and status["Camera 0: face"] == "PASS", status
    assert res["mock"] is True and any("software path only" in s["detail"] for s in res["steps"])


def test_camera_failures_are_warnings_not_crashes(monkeypatch) -> None:
    from src.cardion_app import cameratest, sources
    from src.common import load_config
    monkeypatch.setattr(cameratest, "list_cameras", lambda *a, **k: [])
    assert cameratest.run_camera_test(load_config(), seconds=2.0) == 0              # no camera: warning only
    monkeypatch.setattr(cameratest, "list_cameras", lambda *a, **k: [{"index": 7, "name": "Camera 7", "width": 0, "height": 0}])
    assert cameratest.run_camera_test(load_config(), seconds=2.0, index=7) == 0     # cannot be opened (permission / in use): warning with advice
    res = json.loads((ROOT / "out" / "camera_test_latest.json").read_text(encoding="utf-8"))
    assert any(s["status"] == "WARN" and "could not be opened" in s["detail"] for s in res["steps"])
    assert sources.SourceError


# ---------------------------------------------------------------------------------------------- end-to-end launcher (slow, Windows)
@pytest.mark.slow
@pytest.mark.skipif(not WIN or os.environ.get("CARDION_SLOW_TESTS") != "1", reason="set CARDION_SLOW_TESTS=1 (Windows, builds a venv)")
def test_launcher_repairs_a_sabotaged_environment_in_a_folder_with_spaces(tmp_path_factory) -> None:
    """Disposable copy in ``<tmp>/CARDION test dir/CARDION``: first-time install, then numpy is broken the way a half-finished install
    breaks it, and ``Launch CARDION.bat`` must repair it itself and leave complete logs."""
    base = tmp_path_factory.mktemp("cardion launcher test") / "CARDION"
    shutil.copytree(ROOT, base, ignore=shutil.ignore_patterns(".venv", ".git", "out", "sessions", "logs", "data", "tools", "paper", "third_party",
                                                               "__pycache__", ".pytest_cache", "*.zip"))
    if (ROOT / "tools" / "uv" / "uv.exe").exists():
        (base / "tools" / "uv").mkdir(parents=True)
        shutil.copy(ROOT / "tools" / "uv" / "uv.exe", base / "tools" / "uv" / "uv.exe")
    env = {**os.environ, "CARDION_NO_PAUSE": "1"}
    bat = str(base / "Launch CARDION.bat")

    def launch() -> subprocess.CompletedProcess:
        return subprocess.run(["cmd", "/c", "call", bat, "-CheckOnly", "-NoPause"], capture_output=True, text=True, env=env, timeout=1800, cwd=base)

    r = launch()                                                    # 1. first-time installation
    assert r.returncode == 0, r.stdout[-3000:]
    env_dir = Path(re.search(r"Python environment: (.+)", r.stdout).group(1).strip())      # inside the project, or ~/.cardion/env_* for long folders
    numpy_init = env_dir / "Lib" / "site-packages" / "numpy" / "__init__.py"
    numpy_init.unlink()                                             # unlink first: uv hard-links from its cache, never edit in place
    numpy_init.write_text('raise ImportError("sabotage-marker: numpy DLL load failed (simulated)")\n')
    r = launch()                                                    # 2. broken environment: must repair itself
    assert r.returncode == 0, r.stdout[-3000:]
    assert "sabotage-marker" not in numpy_init.read_text()
    logs = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (base / "logs").glob("doctor_2*.log"))
    assert "sabotage-marker" in logs and "Traceback (most recent call last)" in logs      # the complete traceback was kept
    (base / "models" / "snn.pt").unlink()                           # 3. an unrepairable problem: clear message + complete error log
    r = launch()
    assert r.returncode == 1
    err_logs = list((base / "logs").glob("launcher_error_*.log"))
    assert err_logs and "models/snn.pt" in err_logs[-1].read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------- long installation paths (WinError 206)
def test_long_path_errors_are_recognised_but_other_errors_are_not() -> None:
    """The user's failure: 'DLL load failed ... The filename or extension is too long' must lead to relocation, not endless reinstalling."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import doctor
    assert doctor.is_long_path_error("FileNotFoundError: [WinError 206] The filename or extension is too long.")
    assert doctor.is_long_path_error("ImportError: DLL load failed while importing _imaging: The filename or extension is too long.")
    assert doctor.is_long_path_error("ImportError: DLL load failed while importing x", prefix_len=200)      # DLL failure inside a very deep folder
    assert not doctor.is_long_path_error("ImportError: DLL load failed while importing x", prefix_len=40)
    assert not doctor.is_long_path_error("ModuleNotFoundError: No module named 'numpy'")


def test_launcher_moves_environment_out_of_long_folders() -> None:
    src = (ROOT / "scripts" / "launch.ps1").read_text(encoding="ascii")
    assert "Use-ShortEnv" in src and ".cardion" in src and "-eq 11" in src
    assert (ROOT / "Move CARDION to a short folder.bat").is_file()


def test_full_report_never_shows_a_stale_result(tmp_path: Path) -> None:
    """A selftest that crashes at import time must not leave an older PASS report to be displayed as this run's result."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import doctor
    d = doctor.Doctor(tmp_path, repair=False, deep=False, uv=None, allow_recreate=False)
    (tmp_path / "out").mkdir()
    stale = tmp_path / "out" / "selftest_report.md"
    stale.write_text("# old\n**PASS: 16/16 checks passed**\n", encoding="utf-8")
    d.py = sys.executable
    code = d.full(0)                    # tmp_path has no src package: every step crashes -> there are no fresh reports
    text = (tmp_path / "out" / "test_and_repair_report.md").read_text(encoding="utf-8")
    assert code == 1 and "PROBLEMS FOUND" in text
    assert "16/16" not in text and "NO REPORT WAS PRODUCED BY THIS RUN" in text
