"""Run the live-app pipeline (real MediaPipe face tracking, streaming estimators, reliability gate) on real MCD-rPPG videos.

Compares the app's per-second estimates with the contact-PPG reference over the same 10 s windows, for the *validation* subjects only.
Reports coverage of the reliability gate and the error of the numbers it would show.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import add_common_args, out_path, register_claim, resolve_cfg, save_json  # noqa: E402
from src.capture import video_frames  # noqa: E402
from src.cardion_app.engine import load_models  # noqa: E402
from src.cardion_app.pipeline import Pipeline  # noqa: E402
from src.datasets import clip_specs, mcd_effective_fs  # noqa: E402
from src.rppg_classical import reference_windows  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test (frozen rule, run once)")
    ap.add_argument("--max-clips", type=int, default=4)
    ap.add_argument("--prefix", default=None, help="claim-key prefix (default app_check.<split>)")
    ap.add_argument("--seconds", type=float, default=70.0, help="process this much video from the start (+5 s skip like the dataset excerpt)")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    if args.split == "test" and not args.final:
        raise SystemExit("mcd:test is protected; pass --final for the single frozen evaluation")
    models, info = load_models(cfg)
    root = ROOT / cfg["datasets"]["mcd_root"]
    rows, errs_all, shown_all, n_all = [], [], 0, 0
    per_clip, costs_all, roi_rates = [], [], []
    for spec in clip_specs("mcd", args.split, cfg, 100)[:: max(1, 24 // args.max_clips)][: args.max_clips]:
        cid = spec.split(":")[1]
        pid, step = cid.split("_")
        stem = f"{pid}_FullHDwebcam_{step}"
        ppg = np.loadtxt(root / "ppg_sync" / f"{stem}.txt")[:, 0]
        fs = mcd_effective_fs(root / "meta" / f"{stem}.txt", len(ppg))
        frames, _, _ = video_frames(root / "video" / f"{stem}.avi")
        pl = Pipeline(cfg, models, use_landmarks=True)
        start = int(5 * fs)
        n_frames = int(args.seconds * fs)
        ref, valid = reference_windows(ppg[start:start + n_frames], fs, cfg)
        wd = {}
        for i, f in enumerate(frames):
            if i < start:
                continue
            j = i - start
            if j >= n_frames:
                break
            pl.push(f, j / fs)
            if j % int(fs) == int(fs) - 1 and pl.seconds() > 8:
                a = pl.estimate()
                w = int(round(pl.rec[-1][0])) - int(cfg["rppg"]["hr_window_s"])        # reference window index (start second)
                if 0 <= w < len(ref):
                    for m, e in pl.last_est.items():
                        d = wd.setdefault(m, {"bpm": [], "ref": [], "valid": [], "q": []})
                        d["bpm"].append(round(float(e["bpm"]), 1)); d["ref"].append(round(float(ref[w]), 1)); d["valid"].append(bool(valid[w]))
                        d["q"].append(round(float(e["peak_ratio"]), 2))
                if 0 <= w < len(ref) and valid[w]:
                    n_all += 1
                    if a.status == "ok":
                        shown_all += 1
                        errs_all.append(abs(a.bpm - ref[w]))
                        rows.append({"clip": spec, "t": pl.rec[-1][0], "bpm": a.bpm, "ref": float(ref[w])})
        costs_all.append(pl.costs())
        roi_rates.append(pl.events_total / max(pl.rec[-1][0], 1e-6) / (3 * max(int(pl.mask_crop.sum()), 1)))
        per_clip.append({"spec": spec, "windows_data": wd})
        print(spec, f"fs={fs:.2f}", "shown so far", shown_all, "of", n_all, flush=True)
    errs = np.array(errs_all)
    res = {"coverage": shown_all / max(n_all, 1), "mae_shown": float(errs.mean()) if len(errs) else None, "within5": float(np.mean(errs <= 5)) if len(errs) else None,
           "n_windows": n_all, "n_shown": shown_all, "clips": args.max_clips}
    for part, key in (("roi", "shared_roi"), ("dvs", "dvs_emulation"), ("est", "estimators"), ("snn", "snn"), ("ann", "ann")):
        res[f"cost_{part}_ms_per_s"] = float(np.mean([c[key]["ms_per_s"] for c in costs_all]))
    res["cost_snn_sops_per_s"] = float(np.mean([c["snn"]["sops_per_s"] for c in costs_all]))
    res["cost_ann_macs_per_s"] = float(np.mean([c["ann"]["macs_per_s"] for c in costs_all]))
    res["events_per_roi_pixel_per_s"] = float(np.mean(roi_rates))
    path = save_json({"summary": res, "rows": rows, "dataset": "mcd", "split": args.split, "per_clip": per_clip, "tag": "app"}, out_path(cfg, f"app_real_video_check_{args.split}", "json"))
    print(res)
    for k, v in res.items():
        if v is not None:
            register_claim(f"{args.prefix or 'app_check.' + args.split}.{k}", v, path, "python scripts/app_real_video_check.py")


if __name__ == "__main__":
    main()
