"""Entry point: ``python -m src.cardion_app`` starts the local app and opens it in the default browser.

    python -m src.cardion_app                 # normal launch
    python -m src.cardion_app --no-browser    # headless (tests)
    python -m src.cardion_app --selftest      # run the automated self-test and write a pass/fail report
    python -m src.cardion_app --cameratest    # check webcams end to end (add --mock to use a recorded video instead of hardware)

Everything printed (and any crash with its complete traceback) is also written to ``logs/<mode>_<time>.log``. This module imports
only the standard library until the log is open, so even an import error of a heavy package is recorded in full.
"""
from __future__ import annotations

import datetime as dt
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class _Tee:
    """File-like object that writes to the original stream and to the log file."""

    def __init__(self, stream, fh) -> None:
        self.stream, self.fh = stream, fh

    def write(self, s: str) -> int:
        try:
            if self.stream is not None:
                self.stream.write(s)
        except Exception:  # noqa: BLE001 - a closed console must not crash the app
            pass
        try:
            self.fh.write(s)
            self.fh.flush()
        except Exception:  # noqa: BLE001
            pass
        return len(s)

    def flush(self) -> None:
        try:
            if self.stream is not None:
                self.stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def open_log(kind: str) -> Path:
    """Create ``logs/<kind>_<time>.log`` and mirror stdout/stderr into it. Returns the path."""
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{kind}_{dt.datetime.now():%Y%m%d-%H%M%S}.log"
    fh = open(path, "a", encoding="utf-8", errors="replace")
    fh.write(f"CARDION {kind} log  {dt.datetime.now().isoformat(timespec='seconds')}\npython {sys.version}\nexecutable {sys.executable}\n"
             f"root {ROOT}\nargv {sys.argv}\n\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return path


def run(argv: list[str], log_path: Path) -> int:
    """Parse arguments and run the requested mode; returns a process exit code."""
    import argparse
    import logging
    import threading
    import webbrowser

    from ..common import add_common_args, resolve_cfg, set_seed, setup_logging

    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cameratest", action="store_true", help="check cameras: frames, face, pulse pipeline; writes out/camera_test_*.json/md")
    ap.add_argument("--mock", action="store_true", help="with --cameratest: feed a recorded/simulated video through the camera code path")
    ap.add_argument("--video", default=None, help="with --cameratest --mock: video file to use")
    ap.add_argument("--index", type=int, default=None, help="with --cameratest: only this camera index")
    ap.add_argument("--seconds", type=float, default=25.0, help="with --cameratest: capture length per camera")
    ap.add_argument("--snapshot", action="store_true", help="with --cameratest: keep one image of the first frame (default: no images are saved)")
    ap.add_argument("--autostart", choices=["demo"], default=None, help="start a source immediately")
    args = ap.parse_args(argv)
    cfg = resolve_cfg(args)
    setup_logging("cardion.app", logging.INFO)
    set_seed(cfg["seed"])
    if args.selftest:
        from .selftest import run_selftest
        return run_selftest(cfg)
    if args.cameratest:
        from .cameratest import run_camera_test
        return run_camera_test(cfg, seconds=args.seconds, index=args.index, mock=args.mock, video=args.video, snapshot=args.snapshot)
    from .engine import Engine
    from .server import serve
    engine = Engine(cfg)
    srv, port = serve(engine, port=args.port or int(cfg["app"].get("port", 8765)))
    url = f"http://127.0.0.1:{port}/"
    print(f"CARDION is running at {url}  (close this window or press Ctrl+C to quit)", flush=True)
    print(f"Log file: {log_path}", flush=True)
    if args.autostart == "demo":
        engine.start({"type": "demo", "bpm": 72})
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        srv.server_close()
    return 0


def main() -> None:
    """CLI entry point: opens the log first, then runs; any crash is written in full to the log and named on screen."""
    argv = sys.argv[1:]
    kind = "selftest" if "--selftest" in argv else "cameratest" if "--cameratest" in argv else "app"
    log_path = open_log(kind)
    try:
        code = run(argv, log_path)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        code = 0
    except BaseException:  # noqa: BLE001 - report everything, with the full traceback
        traceback.print_exc()
        print(f"\nCARDION stopped because of an error. The complete error report is saved in:\n  {log_path}", flush=True)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
