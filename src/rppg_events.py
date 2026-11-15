"""Unsupervised pulse extraction from event streams (the event-domain analogue of POS).

Per-channel signed event counts over the face ROI are integrated into a log-intensity-like signal
(an event camera outputs d/dt of log I, so its running sum is a log-intensity trace up to drift). The
trace is detrended, then POS-style projected: ``log`` traces play the role of normalised ``C_n`` (a small log
change equals a small relative change). ``bin_ms`` sets the temporal bin: the frame period for the main
comparison, or finer (e.g. 10 ms) to study latency. Note the information limit documented in
``dvs_emulate``: events synthesised from 30 fps video contain no content above 15 Hz.
"""
from __future__ import annotations

import argparse
import logging
import math

import numpy as np

from .common import add_common_args, resolve_cfg, set_seed, setup_logging
from .events import EventStream, binned_rates
from .rppg_classical import bandpass, detrend, sliding_hr

log = logging.getLogger("cardion.rppg_events")


def event_log_traces(streams: list[EventStream], mask: np.ndarray, bin_s: float) -> tuple[np.ndarray, float]:
    """Integrate per-channel signed counts into (T,3) traces and return them with their sample rate."""
    cols = []
    for s in streams:
        _, on, off = binned_rates(s, bin_s, mask)
        cols.append(np.cumsum(on - off))
    tr = np.stack(cols, axis=1)
    return tr / max(float(mask.sum()), 1.0), 1.0 / bin_s


def pos_events(streams: list[EventStream], mask: np.ndarray, bin_s: float, win_s: float = 1.6,
               band: tuple[float, float] = (0.7, 4.0)) -> tuple[np.ndarray, float]:
    """POS-style chrominance projection on integrated event traces; returns (pulse, fs)."""
    tr, fs = event_log_traces(streams, mask, bin_s)
    return pos_from_traces(tr, fs, win_s, band), fs


def pos_from_traces(tr: np.ndarray, fs: float, win_s: float = 1.6, band: tuple[float, float] = (0.7, 4.0)) -> np.ndarray:
    """POS-style projection of (T,3) integrated net-event traces (also used by the live demo)."""
    tr = np.stack([detrend(tr[:, c], 100.0 * max(fs / 30.0, 1.0)) if len(tr) > 8 else tr[:, c] for c in range(3)], axis=1)
    n, l = len(tr), math.ceil(win_s * fs)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    out = np.zeros(n)
    stride = max(1, l // 8)
    for end in range(l, n, stride):
        seg = tr[end - l:end]
        seg = seg - seg.mean(axis=0)                      # log-domain traces: subtract mean instead of dividing
        s = proj @ seg.T
        h = s[0] + (np.std(s[0]) / (np.std(s[1]) + 1e-12)) * s[1]
        out[end - l:end] += h - h.mean()
    return bandpass(out, fs, band[0], min(band[1], 0.45 * fs))


def green_events(streams: list[EventStream], mask: np.ndarray, bin_s: float,
                 band: tuple[float, float] = (0.7, 4.0)) -> tuple[np.ndarray, float]:
    """Single-channel baseline: integrated green-channel net events, bandpassed."""
    tr, fs = event_log_traces(streams, mask, bin_s)
    return bandpass(detrend(tr[:, 1], 100.0 * max(fs / 30.0, 1.0)), fs, band[0], min(band[1], 0.45 * fs)), fs


def main() -> None:
    """CLI: event-domain HR vs RGB POS on a clip (frame-rate bins)."""
    from .datasets import load_clip
    from .dvs_emulate import emulate_rgb, prepare_face
    from .rppg_classical import estimate, ppg_reference_hr

    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--clip", default="synthetic:0")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    r = cfg["rppg"]
    clip = load_clip(args.clip, cfg)
    face, mask = prepare_face(clip, int(cfg["dvs"]["crop_size"]))
    streams = emulate_rgb(face, clip.fs, cfg, threshold=args.threshold, seed=cfg["seed"])
    _, hr_ref = ppg_reference_hr(clip.ppg, clip.fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]))
    _, _, bpm_rgb = estimate(clip.rgb, clip.fs, "pos", cfg)
    log.info("RGB POS MAE %.2f BPM", np.mean(np.abs(bpm_rgb[:len(hr_ref)] - hr_ref[:len(bpm_rgb)])))
    for ms in cfg["event_rppg"]["window_ms"]:
        pulse, fs = pos_events(streams, mask, ms / 1000.0)
        _, bpm = sliding_hr(pulse, fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]))
        k = min(len(bpm), len(hr_ref))
        log.info("event POS bin=%.1f ms (fs=%.0f Hz): MAE %.2f BPM", ms, fs, np.mean(np.abs(bpm[:k] - hr_ref[:k])))


if __name__ == "__main__":
    main()
