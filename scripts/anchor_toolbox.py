"""Correctness anchor: compare src.rppg_classical POS/CHROM with the rPPG-Toolbox reference code.

Both implementations receive the *identical* mean-RGB trace (fed to the Toolbox as 1x1 "frames", so its
spatial averaging is a no-op). We report waveform correlation, per-clip HR from each, and MAE to the
contact-PPG reference over the same windows.

    python scripts/anchor_toolbox.py --clips synthetic:0 synthetic:1 ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rppg_toolbox_ref" / "pkg"))

if not hasattr(np, "mat"):  # numpy>=2 removed np.mat, which the reference POS_WANG uses (compat shim only)
    np.mat = np.asmatrix  # type: ignore[attr-defined]

from evaluation.post_process import calculate_metric_per_video  # noqa: E402  (Toolbox reference)
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN  # noqa: E402
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402

from src.common import add_common_args, out_path, register_claim, resolve_cfg, save_json, set_seed, setup_logging  # noqa: E402
from src.datasets import load_clip  # noqa: E402
from src.rppg_classical import chrom, pos, ppg_reference_hr, sliding_hr  # noqa: E402

log = logging.getLogger("cardion.anchor")


def run_clip(spec: str, cfg: dict) -> dict:
    """Compare ours vs Toolbox on one clip; returns a result row."""
    clip = load_clip(spec, cfg)
    fs, rgb = clip.fs, clip.rgb
    frames = rgb.reshape(-1, 1, 1, 3)
    r = cfg["rppg"]
    band = tuple(r["hr_band_hz"])
    ours = {"pos": pos(rgb, fs, r["pos_window_s"], tuple(r["band_hz"])), "chrom": chrom(rgb, fs, r["pos_window_s"], tuple(r["band_hz"]))}
    ref = {"pos": POS_WANG(frames, fs), "chrom": CHROME_DEHAAN(frames, fs)}
    t_ref, hr_ref = ppg_reference_hr(clip.ppg, fs, r["hr_window_s"], r["hr_stride_s"], band)
    row: dict = {"clip": spec, "n_frames": len(rgb), "fs": fs}
    for m in ("pos", "chrom"):
        a, b = ours[m], ref[m][:len(ours[m])]
        n = min(len(a), len(b))
        corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
        _, bpm_o = sliding_hr(a[:n], fs, r["hr_window_s"], r["hr_stride_s"], band)
        _, bpm_t = sliding_hr(b[:n], fs, r["hr_window_s"], r["hr_stride_s"], band)
        k = min(len(bpm_o), len(hr_ref))
        # Toolbox whole-video protocol: one FFT HR per video, 0.75-2.5 Hz as recommended in its README
        hr_lbl, hr_tb, _, _ = calculate_metric_per_video(b[:n], clip.ppg[:n], fs=fs, diff_flag=False, hr_method="FFT")
        _, hr_mine, _, _ = calculate_metric_per_video(a[:n], clip.ppg[:n], fs=fs, diff_flag=False, hr_method="FFT")
        row[m] = {"waveform_corr_ours_vs_toolbox": corr,
                  "window_mae_ours": float(np.mean(np.abs(bpm_o[:k] - hr_ref[:k]))),
                  "window_mae_toolbox_pulse": float(np.mean(np.abs(bpm_t[:k] - hr_ref[:k]))),
                  "video_hr_ours": float(hr_mine), "video_hr_toolbox": float(hr_tb), "video_hr_label": float(hr_lbl)}
        log.info("%s %s: corr=%.4f winMAE ours=%.2f toolbox=%.2f | video HR ours=%.1f toolbox=%.1f label=%.1f",
                 spec, m, corr, row[m]["window_mae_ours"], row[m]["window_mae_toolbox_pulse"], hr_mine, hr_tb, hr_lbl)
    return row


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--clips", nargs="+", default=[f"synthetic:{i}" for i in range(5)])
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    rows = [run_clip(c, cfg) for c in args.clips]
    summary = {m: {"mean_waveform_corr": float(np.mean([r[m]["waveform_corr_ours_vs_toolbox"] for r in rows])),
                   "mean_window_mae_ours": float(np.mean([r[m]["window_mae_ours"] for r in rows])),
                   "mean_window_mae_toolbox": float(np.mean([r[m]["window_mae_toolbox_pulse"] for r in rows]))}
               for m in ("pos", "chrom")}
    log.info("SUMMARY %s", summary)
    path = save_json({"rows": rows, "summary": summary, "clips": args.clips}, out_path(cfg, "anchor_toolbox", "json"))
    cmd = "python scripts/anchor_toolbox.py --clips " + " ".join(args.clips)
    src = "+".join(sorted({c.split(":")[0] for c in args.clips}))
    for m in ("pos", "chrom"):
        register_claim(f"anchor.{src}.{m}.waveform_corr", summary[m]["mean_waveform_corr"], path, cmd)
        register_claim(f"anchor.{src}.{m}.window_mae_ours", summary[m]["mean_window_mae_ours"], path, cmd)
        register_claim(f"anchor.{src}.{m}.window_mae_toolbox", summary[m]["mean_window_mae_toolbox"], path, cmd)
        register_claim(f"anchor.{src}.{m}.n_clips", len(args.clips), path, cmd)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
