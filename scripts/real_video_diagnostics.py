"""Diagnostics on real MCD-rPPG videos (validation subjects only) that explain the event-path gap.

1. Illumination flicker: dominant frequency/amplitude of the frame-to-frame log-intensity change (aliased mains flicker).
2. ROI/crop check: RGB green/POS from the full-resolution tracked ROI versus the 64x64 face crop used by the event emulator.
3. Event fidelity: correlation (pulse band) of the integrated net-event trace with the crop-mean log intensity and with the full-resolution one.
4. Pulse strength: PPG-locked amplitude of the green channel and in-band SNR.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, out_path, register_claim, save_json  # noqa: E402
from src.datasets import clip_specs, load_clip  # noqa: E402
from src.dvs_emulate import prepare_face  # noqa: E402
from src.rppg_classical import bandpass, detrend, estimate_q, reference_windows  # noqa: E402
from src.train import event_meta, event_tensor  # noqa: E402


def bp(x: np.ndarray, fs: float) -> np.ndarray:
    """Pulse-band (0.7-4 Hz) filtered, detrended signal."""
    return bandpass(detrend(x - x.mean(), 100.0), fs, 0.7, 4.0)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    cfg = load_config()
    rows = []
    for spec in clip_specs("mcd", args.split, cfg):
        c = load_clip(spec, cfg)
        fs = c.fs
        g = np.log(np.maximum(c.rgb[:, 1], 1.0))
        # 1. flicker
        f, p = signal.welch(np.diff(g), fs=fs, nperseg=256)
        band = (f >= 0.7) & (f <= 4.0)
        fl = (f >= 5.0) & (f <= 14.5)
        k = int(np.argmax(p[fl]))
        flicker_hz, flicker_amp = float(f[fl][k]), float(np.sqrt(p[fl][k] * (f[1] - f[0]) * 2) / (2 * np.sin(np.pi * f[fl][k] / fs)))
        # 4. pulse strength
        ref, valid = reference_windows(c.ppg, fs, cfg)
        pp = bp(c.ppg, fs)
        pp = pp / pp.std()
        lag = max(range(-15, 16), key=lambda l: abs(np.corrcoef(np.roll(pp, l), bp(g, fs))[0, 1]))
        pr = np.roll(pp, lag)
        coef = np.polyfit(pr, bp(g, fs), 1)
        resid = bp(g, fs) - np.polyval(coef, pr)
        # 2. crop vs full ROI
        face, mask = prepare_face(c, 64)
        m = mask.astype(bool)
        crop_rgb = face[:, m].astype(float).mean(axis=1)
        maes = {}
        for name, rgb in (("full", c.rgb), ("crop", crop_rgb)):
            for meth in ("green", "pos"):
                b = estimate_q(rgb, fs, meth, cfg)[2]
                kk = min(len(b), len(ref))
                maes[f"{name}_{meth}"] = float(np.abs(b[:kk] - ref[:kk])[valid[:kk]].mean())
        # 3. event fidelity
        x = event_tensor(c, cfg, args.threshold).astype(np.float32)[: len(g)]
        m32 = event_meta(c, cfg, args.threshold)["mask32"].astype(bool)
        ev = bp(np.cumsum(x[:, 1][:, m32].sum(axis=1) / max(m32.sum(), 1)), fs)
        crop_g = bp(np.log(np.maximum(face[..., 1][:, m].mean(axis=1), 1)), fs)
        rows.append({"clip": spec, "flicker_hz": flicker_hz, "flicker_amp": flicker_amp, "pulse_amp": float(abs(coef[0])),
                     "inband_snr_db": float(10 * np.log10(coef[0] ** 2 / resid.var())), "event_vs_crop": float(np.corrcoef(ev, crop_g)[0, 1]),
                     "event_vs_full": float(np.corrcoef(ev, bp(g, fs))[0, 1]), "crop_vs_full": float(np.corrcoef(crop_g, bp(g, fs))[0, 1]), **maes})
        print(spec, {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items() if k != "clip"}, flush=True)
    keys = [k for k in rows[0] if k != "clip"]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows]))} for k in keys}
    summary["flicker_near_10hz_fraction"] = {"mean": float(np.mean([abs(r["flicker_hz"] - 10.0) < 0.6 for r in rows]))}
    path = save_json({"split": args.split, "threshold": args.threshold, "summary": summary, "rows": rows}, out_path(cfg, f"real_video_diagnostics_{args.split}", "json"))
    cmd = f"python scripts/real_video_diagnostics.py --split {args.split} --threshold {args.threshold}"
    for k, d in summary.items():
        for stat, v in d.items():
            register_claim(f"diag_real.{args.split}.{k}.{stat}", v, path, cmd)
    print(summary)


if __name__ == "__main__":
    main()
