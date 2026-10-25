"""Does a lower effective frame rate change how often CARDION gives a reliable reading, or what it says?

    python scripts/fps_effect.py --clips 6 --seconds 45

Real recorded faces are re-framed like a webcam shot and fed to the RGB pipeline (the part that produces the displayed number) three ways with
*true* time stamps: every frame (30 fps), two of every three (20 fps) and every second frame (15 fps). Frame dropping is what a camera or an
overloaded loop does. For each clip it reports the share of one-per-second estimates that were reliable after the 8 s warm-up, the median
difference of the shown BPM from the 30 fps run, and how many seconds both runs agreed to be reliable. Output: ``out/fps_effect_<time>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_guidance import clips, read_frames  # noqa: E402
from src.cardion_app.pipeline import Pipeline  # noqa: E402
from src.common import load_config, timestamp  # noqa: E402

PATTERNS = {"30fps": lambda i: True, "20fps": lambda i: i % 3 != 2, "15fps": lambda i: i % 2 == 0}


def run(cfg: dict, frames: list[np.ndarray], keep) -> list[tuple[float, str, float | None]]:
    """(t, status, bpm) once per second for the frames kept by ``keep`` (time stamps stay at the true capture times)."""
    pl = Pipeline(cfg, models=None, use_landmarks=True, guidance=False)
    pl.set_events(False)
    out, last = [], 0.0
    for i, fr in enumerate(frames):
        if not keep(i):
            continue
        t = i / 30.0
        pl.push(fr, t)
        if t - last >= 1.0 and pl.seconds() > 2.0:
            last = t
            a = pl.estimate()
            out.append((t, a.status, a.bpm))
    pl.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=45.0)
    a = ap.parse_args()
    cfg = load_config()
    res = []
    for p in clips(a.clips + 4)[4:]:                      # skip the very smallest files (used by the profiler)
        frames, _ = read_frames(p, a.seconds)
        runs = {name: run(cfg, frames, k) for name, k in PATTERNS.items()}
        base = {round(t): (s, b) for t, s, b in runs["30fps"]}
        row = {"clip": p.name}
        for name, r in runs.items():
            post = [(t, s, b) for t, s, b in r if t >= 8.0]
            share = float(np.mean([s == "ok" for _, s, _ in post])) if post else None
            both = [(b, base[round(t)][1]) for t, s, b in r if s == "ok" and round(t) in base and base[round(t)][0] == "ok"]
            diff = [abs(x - y) for x, y in both]
            row[name] = {"reliable_share_after_warmup": share, "seconds_reliable_in_both": len(both), "median_abs_diff_vs_30fps_bpm": float(np.median(diff)) if diff else None}
        res.append(row)
        print(p.name, {k: (round(v["reliable_share_after_warmup"], 2) if v["reliable_share_after_warmup"] is not None else None) for k, v in row.items() if k != "clip"}, flush=True)
    summary = {name: {"mean_reliable_share": float(np.mean([r[name]["reliable_share_after_warmup"] for r in res if r[name]["reliable_share_after_warmup"] is not None])),
                      "median_of_median_abs_diff_bpm": float(np.median([r[name]["median_abs_diff_vs_30fps_bpm"] for r in res if r[name]["median_abs_diff_vs_30fps_bpm"] is not None] or [np.nan]))}
               for name in PATTERNS}
    print(json.dumps(summary, indent=1))
    out = ROOT / "out" / f"fps_effect_{timestamp()}.json"
    out.write_text(json.dumps({"command": f"python scripts/fps_effect.py --clips {a.clips} --seconds {a.seconds}", "per_clip": res, "summary": summary}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
