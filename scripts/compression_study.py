"""Does video compression explain why real webcam video is harder for the event path than the clean simulator?

Simulated faces with a *weak, real-video-like pulse* (amp_scale 0.2) are pasted on a 640x480 background, encoded with OpenCV codecs
(MJPG ~1.2 Mbps; MPEG-4 'mp4v', OpenCV default = ~0.2 Mbps, harsher than the dataset's 1-3 Mbps MPEG-4), decoded again and analysed with
the RGB and classical event estimators. Everything else (ROI, estimators, thresholds) is unchanged. A known pulse makes the error exact.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, out_path, register_claim, save_json  # noqa: E402

CODECS = ["none", "MJPG", "mp4v"]
THRESHOLDS = [0.02, 0.05]


def one(job: tuple) -> dict:
    """One clip, one codec: per-method HR MAE and the achieved bitrate."""
    cid, codec, sigma = job
    from src.datasets import make_synthetic_clip
    from src.rppg_classical import bandpass, detrend, estimate_q, reference_windows, sliding_hr_q
    from src.rppg_events import pos_from_traces
    from src.train import emulate_event_frames
    cfg = load_config(overrides=["datasets.synthetic.size=160", "datasets.synthetic.clip_seconds=40", "datasets.synthetic.amp_scale=0.2",
                                 f"datasets.synthetic.noise_sigma_range=[{sigma},{sigma}]"])
    clip = make_synthetic_clip(cid, cfg)
    rng = np.random.default_rng(cid)
    bg = 70 + cv2.GaussianBlur(rng.normal(0, 6, (480, 640, 3)).astype(np.float32), (0, 0), 9)
    frames = []
    for f in clip.face:
        c = bg.copy()
        c[160:320, 240:400] = f
        frames.append(np.clip(c + rng.normal(0, 1.5, c.shape), 0, 255).astype(np.uint8))
    bitrate = None
    if codec != "none":
        path = Path(tempfile.mkdtemp()) / f"c_{cid}_{codec}.avi"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30.0, (640, 480))
        for f in frames:
            w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        bitrate = path.stat().st_size * 8 / (len(frames) / 30.0) / 1e6
        cap = cv2.VideoCapture(str(path))
        frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB) for _ in range(len(frames))]
        cap.release()
    crop = np.stack([f[160:320, 240:400] for f in frames])
    mask = clip.face_mask.astype(bool)
    rgb = crop[:, mask].astype(np.float64).mean(axis=1)
    ref, valid = reference_windows(clip.ppg, clip.fs, cfg)
    r = cfg["rppg"]

    def err(b: np.ndarray) -> float:
        k = min(len(b), len(ref))
        return float(np.abs(b[:k] - ref[:k])[valid[:k]].mean())

    out = {"clip": cid, "codec": codec, "bitrate_mbps": bitrate}
    for m in ("green", "pos"):
        out[f"rgb_{m}"] = err(estimate_q(rgb, clip.fs, m, cfg)[2])
    face = np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in crop])
    m64 = cv2.resize(clip.face_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
    m32 = cv2.resize(clip.face_mask, (32, 32), interpolation=cv2.INTER_NEAREST).astype(bool)
    for th in THRESHOLDS:
        x = emulate_event_frames(face, m64, clip.fs, cfg, th, cfg["seed"])[0].astype(np.float32)
        tr = np.cumsum(x[:, :, m32].sum(axis=2) / max(m32.sum(), 1), axis=0)
        g = bandpass(detrend(tr[:, 1], 100.0), clip.fs, 0.7, 4.0)
        out[f"event_green@{th}"] = err(sliding_hr_q(g, clip.fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), r["hr_method"])[1])
        p = pos_from_traces(tr, clip.fs, r["pos_window_s"], tuple(r["band_hz"]))
        out[f"event_pos@{th}"] = err(sliding_hr_q(p, clip.fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), r["hr_method"])[1])
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config()
    jobs = [(c, k, args.sigma) for k in CODECS for c in range(args.clips)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    methods = [k for k in rows[0] if k not in ("clip", "codec", "bitrate_mbps")]
    summary = {codec: {m: float(np.mean([r[m] for r in rows if r["codec"] == codec])) for m in methods} for codec in CODECS}
    for codec in CODECS:
        br = [r["bitrate_mbps"] for r in rows if r["codec"] == codec and r["bitrate_mbps"]]
        summary[codec]["bitrate_mbps"] = float(np.mean(br)) if br else None
    path = save_json({"summary": summary, "rows": rows, "sigma": args.sigma}, out_path(cfg, "compression_study", "json"))
    cmd = f"python scripts/compression_study.py --clips {args.clips} --sigma {args.sigma}"
    for codec, d in summary.items():
        for m, v in d.items():
            if v is not None:
                register_claim(f"compression.{codec}.{m}", v, path, cmd, note=f"mean over {args.clips} simulated 40 s clips, weak pulse (amp_scale 0.2), sigma {args.sigma}")
        print(codec, {k: (round(v, 2) if v is not None else None) for k, v in d.items()})


if __name__ == "__main__":
    main()
