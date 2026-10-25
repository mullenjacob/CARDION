"""Record a webcam clip for the supplementary (qualitative) author dataset.

RECORDING PROTOCOL (please follow; it takes about 2 minutes per clip)
1. Sit 0.5-1 m from the camera, face centred, head supported, shoulders relaxed.
2. Light the face from the front with steady, diffuse light (daylight from a window in front of you is best;
   avoid a lamp that flickers, and avoid strong side light or a bright window behind you).
3. Neutral expression, mouth closed, do not talk, look at the lens, keep still. Glasses off if comfortable.
4. If your camera app allows it, lock exposure / white balance / focus before recording (auto adjustments
   create colour changes that look like pulse).
5. Record ~60 s per clip; make 3-5 clips (e.g. rest, after light exercise, different light).
6. Optionally note your pulse from a pulse oximeter/watch at the START and END in ``--ref-bpm-start/--ref-bpm-end``.
   Without a reference the clips are demo/qualitative only and never enter the quantitative tables.
Clips contain your face: do not publish them without thinking about it.

    python -m src.record --subject me01 --seconds 60                # camera 0
    python -m src.record --subject test --source synthetic --seconds 5   # dry run, no camera
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time

import cv2

from .common import REPO_ROOT, add_common_args, resolve_cfg, set_seed, setup_logging, timestamp

log = logging.getLogger("cardion.record")


def open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer lossless FFV1; fall back to high-quality MJPG."""
    for code in ("FFV1", "MJPG"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
        if w.isOpened():
            log.info("writing with codec %s", code)
            return w
    raise IOError(f"could not open a video writer for {path}")


def main() -> None:
    """CLI entry point."""
    from .capture import WebcamCapture
    from .datasets import make_synthetic_clip

    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--subject", required=True, help="short subject id, e.g. me01")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--source", default="camera", choices=["camera", "synthetic"])
    ap.add_argument("--ref-bpm-start", type=float, default=None)
    ap.add_argument("--ref-bpm-end", type=float, default=None)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    out_dir = REPO_ROOT / cfg["datasets"]["author_root"] / f"{args.subject}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(args.seconds * args.fps)

    if args.source == "synthetic":
        scfg = {**cfg, "datasets": {**cfg["datasets"], "synthetic": {**cfg["datasets"]["synthetic"], "clip_seconds": args.seconds, "fps": args.fps, "size": 240}}}
        clip = make_synthetic_clip(0, scfg)
        frames_iter = iter(clip.face)
        grab = lambda: next(frames_iter, None)  # noqa: E731
        size, cam = (240, 240), None
    else:
        cam = WebcamCapture(args.camera, 640, 480, args.fps)
        grab = cam.read
        size = (640, 480)
        for i in range(args.countdown, 0, -1):
            log.info("recording starts in %d ...", i)
            time.sleep(1)
        for _ in range(10):  # let auto-exposure settle
            cam.read()
    writer = open_writer(str(out_dir / "vid.avi"), args.fps, size)
    stamps = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        frame = grab()
        if frame is None:
            log.warning("source ended at frame %d", i)
            break
        stamps.append(time.perf_counter() - t0)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if args.source == "camera" and not args.no_preview:
            cv2.imshow("recording (q to stop)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    writer.release()
    if cam:
        cam.close()
        cv2.destroyAllWindows()
    with open(out_dir / "timestamps.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "t_s"])
        wr.writerows(enumerate(stamps))
    real_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and args.source == "camera" else float(args.fps)
    meta = {"subject": args.subject, "frames": len(stamps), "nominal_fps": args.fps, "measured_fps": real_fps,
            "source": args.source, "ref_bpm_start": args.ref_bpm_start, "ref_bpm_end": args.ref_bpm_end,
            "created": timestamp(), "note": "supplementary / qualitative only unless reference BPM provided"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %d frames (measured %.1f fps) to %s", len(stamps), real_fps, out_dir)


if __name__ == "__main__":
    main()
