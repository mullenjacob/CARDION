"""Controlled noise / flicker study on the synthetic generator (known pulse, no training involved).

For every (sensor-noise sigma, flicker amplitude) condition it generates clips, then measures HR error of RGB green / POS and of the
classical event estimators at several contrast thresholds. Shows where the event path breaks as the video gets noisier or flickers,
the two properties that distinguish real compressed webcam footage from the clean simulator.

    python scripts/noise_study.py --clips 6 --workers 3
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import add_common_args, load_config, out_path, register_claim, resolve_cfg, save_json  # noqa: E402

SIGMAS = [0.5, 2.0, 4.0, 8.0]
FLICKERS = [0.0, 0.1]
THRESHOLDS = [0.02, 0.05, 0.1]


def one(job: tuple) -> dict:
    """Evaluate one clip under one condition; returns per-method MAE (BPM)."""
    sigma, flick, cid, overrides = job
    from src.datasets import make_synthetic_clip
    from src.dvs_emulate import prepare_face
    from src.rppg_classical import bandpass, detrend, estimate_q, reference_windows, sliding_hr_q
    from src.rppg_events import pos_from_traces
    from src.train import emulate_event_frames
    cfg = load_config(overrides=overrides + [f"datasets.synthetic.noise_sigma_range=[{sigma},{sigma}]", f"datasets.synthetic.flicker_amp={flick}",
                                             "datasets.synthetic.clip_seconds=40"])
    clip = make_synthetic_clip(cid, cfg)
    ref, valid = reference_windows(clip.ppg, clip.fs, cfg)
    r = cfg["rppg"]
    out = {}

    def err(b: np.ndarray) -> float:
        k = min(len(b), len(ref))
        return float(np.abs(b[:k] - ref[:k])[valid[:k]].mean())

    for m in ("green", "pos"):
        out[f"rgb_{m}"] = err(estimate_q(clip.rgb, clip.fs, m, cfg)[2])
    face, mask = prepare_face(clip, 64)
    m32 = (np.asarray([[mask[2 * i:2 * i + 2, 2 * j:2 * j + 2].mean() for j in range(32)] for i in range(32)]) > 0.5)
    for th in THRESHOLDS:
        x, _, _ = emulate_event_frames(face, mask, clip.fs, cfg, th, cfg["seed"])
        x = x.astype(np.float32)
        tr = np.cumsum(x[:, :, m32].sum(axis=2) / max(m32.sum(), 1), axis=0)
        g = bandpass(detrend(tr[:, 1], 100.0), clip.fs, 0.7, 4.0)
        out[f"event_green@{th}"] = err(sliding_hr_q(g, clip.fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), r["hr_method"])[1])
        p = pos_from_traces(tr, clip.fs, r["pos_window_s"], tuple(r["band_hz"]))
        out[f"event_pos@{th}"] = err(sliding_hr_q(p, clip.fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), r["hr_method"])[1])
    return {"sigma": sigma, "flicker": flick, "clip": cid, **out}


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--amp-scale", type=float, default=1.0, help="scale the simulated pulse amplitude (0.2 ~ real MCD-rPPG video)")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    jobs = [(s, f, c, args.overrides + [f"datasets.synthetic.amp_scale={args.amp_scale}"]) for s in SIGMAS for f in FLICKERS for c in range(args.clips)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            rows.append(r)
            if i % 6 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    methods = [k for k in rows[0] if k not in ("sigma", "flicker", "clip")]
    summary = {f"sigma{s}_flicker{f}": {m: float(np.mean([r[m] for r in rows if r["sigma"] == s and r["flicker"] == f])) for m in methods}
               for s in SIGMAS for f in FLICKERS}
    path = save_json({"summary": summary, "rows": rows, "sigmas": SIGMAS, "flickers": FLICKERS, "thresholds": THRESHOLDS, "clips": args.clips},
                     out_path(cfg, "noise_study", "json", tag=f"amp{args.amp_scale}"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(FLICKERS), figsize=(9, 3.4), sharey=True)
    for ax, f in zip(axes, FLICKERS):
        for m in ("rgb_green", "rgb_pos", "event_green@0.02", "event_green@0.1", "event_pos@0.02", "event_pos@0.1"):
            ax.plot(SIGMAS, [summary[f"sigma{s}_flicker{f}"][m] for s in SIGMAS], marker="o", label=m, ls="--" if m.startswith("rgb") else "-")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("sensor noise sigma (8-bit counts)"); ax.set_title(f"illumination flicker {f:.0%}", fontsize=9)
    axes[0].set_ylabel("HR MAE (BPM)"); axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "fig_noise_study", "png", tag=f"amp{args.amp_scale}", stamp=False), dpi=150)
    cmd = f"python scripts/noise_study.py --clips {args.clips} --amp-scale {args.amp_scale}"
    for cond, d in summary.items():
        for m, v in d.items():
            register_claim(f"noise.amp{args.amp_scale}.{cond}.{m}", v, path, cmd, note=f"mean over {args.clips} synthetic 40 s clips")
    print("wrote", path)


if __name__ == "__main__":
    main()
