"""Development-only analysis on the TRAIN subjects: reference-PPG quality and RGB baselines under peak vs SHS HR picking.

Never run with --split test. Used to choose ``rppg.hr_method`` and ``rppg.ref_min_peak_ratio`` without touching val/test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import add_common_args, out_path, register_claim, resolve_cfg, save_json  # noqa: E402
from src.datasets import clip_specs, load_clip  # noqa: E402
from src.rppg_classical import bandpass, detrend, estimate, sliding_hr_q  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--max-clips", type=int, default=72)
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    specs = clip_specs("mcd", args.split, cfg, args.max_clips)
    ratios, rows = [], {"peak": {}, "shs": {}}
    refs_all = []
    for s in specs:
        c = load_clip(s, cfg)
        p = bandpass(detrend(c.ppg - c.ppg.mean(), 100.0), c.fs, 0.7, 4.0)
        _, ref, q = sliding_hr_q(p, c.fs, 10, 1, (0.75, 3.0), "peak")
        ratios.append(q); refs_all.append(ref)
        for hm in ("peak", "shs"):
            cfg["rppg"]["hr_method"] = hm
            for m in ("green", "pos", "chrom"):
                bpm = estimate(c.rgb, c.fs, m, cfg)[2]
                k = min(len(bpm), len(ref))
                rows[hm].setdefault(m, []).append(np.abs(bpm[:k] - ref[:k]))
        qs = np.concatenate(ratios)
    allq = np.concatenate(ratios)
    print(f"{len(specs)} clips, {len(allq)} windows; ref HR range {np.concatenate(refs_all).min():.0f}-{np.concatenate(refs_all).max():.0f}")
    print("reference peak-ratio percentiles 5/25/50/75:", np.percentile(allq, [5, 25, 50, 75]).round(1))
    for thr in (0, 2, 3, 4, 6, 8):
        print(f"ref ratio >= {thr}: {np.mean(allq >= thr):.2f} of windows kept")
    thr = float(cfg["rppg"]["ref_min_peak_ratio"])
    summary = {}
    for hm in ("peak", "shs"):
        for m in ("green", "pos", "chrom"):
            errs = np.concatenate([e[(q[: len(e)] >= thr)] for e, q in zip(rows[hm][m], ratios)])
            summary[f"{hm}.{m}"] = {"mae": float(errs.mean()), "median": float(np.median(errs)), "within5": float(np.mean(errs <= 5)), "n": int(len(errs))}
            print(f"{hm:5s} {m:6s} MAE over valid windows (ratio>={thr}): {errs.mean():6.2f}  median {np.median(errs):5.2f}  within 5 BPM {np.mean(errs <= 5):.2f}")
    path = save_json({"split": args.split, "summary": summary, "ref_min_peak_ratio": thr, "ratio_percentiles": np.percentile(allq, [5, 25, 50, 75]).tolist()},
                     out_path(cfg, f"dev_baselines_{args.split}", "json"))
    for k, s in summary.items():
        for f, v in s.items():
            register_claim(f"dev.{args.split}.{k}.{f}", v, path, f"python scripts/dev_baselines.py --split {args.split}")
    register_claim("config.ref_min_peak_ratio", thr, path, "configs/default.yaml rppg.ref_min_peak_ratio")


if __name__ == "__main__":
    main()

