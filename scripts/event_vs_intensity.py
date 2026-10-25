"""Sanity check of the event representation on real video (validation clips only).

The running sum of net events over the skin ROI should track the log of the ROI-mean intensity (that is what a DVS measures).
We correlate the two after the pulse bandpass (0.7-4 Hz) and report, per threshold, how much of the *pulse-band* intensity
signal the events preserve. A low value means the event path loses the pulse before any estimator or network sees it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import add_common_args, out_path, register_claim, resolve_cfg, save_json  # noqa: E402
from src.datasets import clip_specs, load_clip  # noqa: E402
from src.rppg_classical import bandpass, detrend  # noqa: E402
from src.train import event_meta, event_tensor  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max-clips", type=int, default=24)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    rows = {th: [] for th in args.thresholds}
    for spec in clip_specs("mcd", args.split, cfg, args.max_clips):
        c = load_clip(spec, cfg)
        g = np.log(np.maximum(c.rgb[:, 1], 1.0))
        gi = bandpass(detrend(g - g.mean(), 100.0), c.fs, 0.7, 4.0)
        for th in args.thresholds:
            x = event_tensor(c, cfg, th).astype(np.float32)[: len(g)]
            m = event_meta(c, cfg, th)["mask32"].astype(bool)
            tr = np.cumsum(x[:, 1][:, m].sum(axis=1) / max(m.sum(), 1))
            ge = bandpass(detrend(tr, 100.0), c.fs, 0.7, 4.0)
            n = min(len(ge), len(gi))
            rows[th].append(float(np.corrcoef(ge[:n], gi[:n])[0, 1]))
    summary = {str(th): {"mean_r": float(np.mean(v)), "median_r": float(np.median(v)), "n": len(v)} for th, v in rows.items()}
    path = save_json({"split": args.split, "summary": summary, "per_clip": rows}, out_path(cfg, f"event_vs_intensity_{args.split}", "json"))
    for th, s in summary.items():
        print(f"theta={th}: pulse-band correlation event-trace vs log-intensity: mean {s['mean_r']:.3f}, median {s['median_r']:.3f} over {s['n']} clips")
        for k in ("mean_r", "median_r"):
            register_claim(f"event_vs_intensity.{args.split}.th{th}.{k}", s[k], path, f"python scripts/event_vs_intensity.py --split {args.split}")


if __name__ == "__main__":
    main()
