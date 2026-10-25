"""Measure what the positioning guidance thresholds should be, from real recorded faces (no new data, no training).

    python scripts/calibrate_guidance.py geometry     # face size / head-pose / light-asymmetry statistics of real faces
    python scripts/calibrate_guidance.py scenarios    # what the guidance measures on each synthetic reposition of a real face
    python scripts/calibrate_guidance.py sweep        # does the pulse estimate degrade when the face is smaller / off-centre / dark?

``sweep`` compares each repositioned run with the same clip at its normal position (agreement of the video pulse methods and the
reliability gate); it needs no reference monitor and tells which framing problems really damage the signal. Results go to
``out/guidance_calibration_<command>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cardion_app.guidance import GuidanceParams, geometry_from_landmarks, side_luma  # noqa: E402
from src.cardion_app.mockscenarios import apply_scenario, reframe, reframe_matrix  # noqa: E402
from src.cardion_app.pipeline import Pipeline  # noqa: E402
from src.common import load_config, timestamp  # noqa: E402

VIDEO_DIR = ROOT / "data" / "raw" / "mcd_rppg" / "video"


def clips(n: int) -> list[Path]:
    """The ``n`` smallest recordings (quick to process)."""
    return sorted(VIDEO_DIR.glob("*.avi"), key=lambda p: p.stat().st_size)[:n]


def read_frames(path: Path, seconds: float | None = None, reframed: bool = True) -> tuple[list[np.ndarray], float]:
    """RGB frames resized to 640x480 (what the camera path delivers) and the frame rate."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok or (seconds and len(out) >= seconds * fps):
            break
        out.append(cv2.cvtColor(cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    m = reframe_matrix(out[:24], load_config())            # a normally framed webcam shot (face centred, ~45% of the height)
    return [reframe(f, m) for f in out], fps


def cmd_geometry(args: argparse.Namespace, cfg: dict) -> dict:
    from src.capture import LandmarkSkinROI
    roi = LandmarkSkinROI(cfg)
    rows = []
    for p in clips(args.clips):
        frames, _ = read_frames(p, 4.0)
        for fr in frames[:: max(len(frames) // 4, 1)]:
            faces = roi.detect_all(fr)
            if not faces:
                continue
            m = roi.mask_from_landmarks(faces[0], fr.shape[:2])
            g = geometry_from_landmarks(faces[0], fr.shape[:2], side_luma(fr, m, float(np.nonzero(m)[1].mean())))
            lm = max(sum(g.luma_lr) / 2, 1.0) if g.luma_lr else None
            rows.append({"clip": p.name, "cx": g.cx, "cy": g.cy, "h": g.h, "w": g.w, "yaw": abs(g.yaw), "pitch": g.pitch,
                         "uneven": abs(g.luma_lr[0] - g.luma_lr[1]) / lm if lm else None, "outside": g.outside})
    keys = ["cx", "cy", "h", "w", "yaw", "pitch", "uneven", "outside"]
    summ = {k: {q: float(np.nanpercentile([r[k] for r in rows if r[k] is not None], v)) for q, v in (("p5", 5), ("p50", 50), ("p95", 95), ("max", 100))} for k in keys}
    print(json.dumps(summ, indent=1))
    return {"n_faces_measured": len(rows), "summary": summ, "rows": rows}


def cmd_scenarios(args: argparse.Namespace, cfg: dict) -> dict:
    from src.capture import LandmarkSkinROI
    roi = LandmarkSkinROI(cfg)
    frames, _ = read_frames(clips(1)[0], 2.0)
    base = frames[len(frames) // 2]
    specs = ["good", "shift_x:0.25", "shift_x:-0.25", "shift_y:0.25", "shift_y:-0.25", "scale:0.6", "scale:0.4", "scale:0.3", "scale:1.5", "scale:2.0",
             "turn:0.2", "turn:0.35", "turn:0.5", "dark:0.3", "uneven:0.4", "two_faces", "noface"]
    out = []
    for s in specs:
        fr = apply_scenario(base, s, 3)
        faces = roi.detect_all(fr)
        if not faces:
            out.append({"spec": s, "faces": 0})
            print(f"{s:14s} faces=0")
            continue
        g = geometry_from_landmarks(faces[0], fr.shape[:2])
        row = {"spec": s, "faces": len(faces), "cx": round(g.cx, 3), "cy": round(g.cy, 3), "h": round(g.h, 3), "yaw": round(g.yaw, 3),
               "pitch": round(g.pitch, 3), "outside": round(g.outside, 3), "touching": g.touching}
        out.append(row)
        print(f"{s:14s} " + " ".join(f"{k}={v}" for k, v in row.items() if k != "spec"))
    return {"rows": out}


def run_clip(cfg: dict, frames: list[np.ndarray], fps: float, spec: str) -> dict:
    """Push one repositioned clip through the real pipeline (RGB methods only, as fast as possible); report the last estimate."""
    pl = Pipeline(cfg, models=None, use_landmarks=True, guidance=True)
    pl.set_events(False)
    last = None
    statuses = []
    for i, fr in enumerate(frames):
        pl.push(apply_scenario(fr, spec, i), i / fps)
        if i % int(fps) == int(fps) - 1 and pl.seconds() > 2:
            last = pl.estimate()
            statuses.append(last.status)
    pl.close()
    est = {k: v["bpm"] for k, v in pl.last_est.items()}
    return {"spec": spec, "final_status": last.status if last else None, "bpm": last.bpm if last else None,
            "ok_fraction_after_10s": float(np.mean([s == "ok" for s in statuses[10:]])) if len(statuses) > 10 else None,
            "rgb": {k: est.get(k) for k in ("rgb_green", "rgb_pos", "rgb_chrom")}, "guide_shown": pl.guide.shown if pl.guide else None,
            "face_h": pl.guide.sm.get("h") if pl.guide and pl.guide.sm else None}


def cmd_sweep(args: argparse.Namespace, cfg: dict) -> dict:
    specs = ["good", "scale:0.7", "scale:0.5", "scale:0.4", "scale:0.3", "scale:0.22", "shift_x:0.15", "shift_x:0.3", "shift_y:0.2", "scale:1.4", "dark:0.5", "dark:0.25"]
    results = []
    for p in clips(args.clips):
        frames, fps = read_frames(p, args.seconds)
        base = None
        for s in specs:
            r = run_clip(cfg, frames, fps, s)
            r["clip"] = p.name
            if s == "good":
                base = r
            r["rgb_pos_vs_normal_bpm"] = (abs(r["rgb"]["rgb_pos"] - base["rgb"]["rgb_pos"]) if r["rgb"]["rgb_pos"] is not None and base and base["rgb"]["rgb_pos"] is not None else None)
            results.append(r)
            print(f"{p.name[:18]:18s} {s:13s} status={r['final_status']} bpm={r['bpm'] and round(r['bpm'], 1)} ok_frac={r['ok_fraction_after_10s']} "
                  f"d_pos={r['rgb_pos_vs_normal_bpm'] and round(r['rgb_pos_vs_normal_bpm'], 1)} face_h={r['face_h'] and round(r['face_h'], 2)} guide={r['guide_shown']}", flush=True)
    return {"specs": specs, "clips": [p.name for p in clips(args.clips)], "seconds": args.seconds, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["geometry", "scenarios", "sweep"])
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config()
    res = {"geometry": cmd_geometry, "scenarios": cmd_scenarios, "sweep": cmd_sweep}[a.command](a, cfg)
    out = ROOT / "out" / f"guidance_calibration_{a.command}_{timestamp()}.json"
    out.write_text(json.dumps({"command": f"python scripts/calibrate_guidance.py {a.command} --clips {a.clips} --seconds {a.seconds}", **res}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
