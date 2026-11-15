"""Classical unsupervised rPPG: POS (Wang 2017) and CHROM (de Haan & Jeanne 2013).

All functions take a ``(T, 3)`` mean-RGB trace and a sample rate. The POS/CHROM windowing mirrors the
rPPG-Toolbox reference implementation (kept in ``third_party/rppg_toolbox_ref``) so the two can be
compared numerically in ``scripts/anchor_toolbox.py``.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

from .common import add_common_args, out_path, resolve_cfg, set_seed, setup_logging

log = logging.getLogger("cardion.rppg")


def detrend(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Tarvainen smoothness-priors detrending (same maths as the Toolbox, sparse solve)."""
    n = len(x)
    if n < 4:
        return x - np.mean(x)
    d = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    a = sparse.identity(n) + (lam ** 2) * (d.T @ d)
    trend = spsolve(a.tocsc(), x)
    return x - trend


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass (second-order sections)."""
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64))


def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6, band: tuple[float, float] = (0.7, 4.0)) -> np.ndarray:
    """Plane-orthogonal-to-skin pulse signal, overlap-added with stride 1 like the reference."""
    n = rgb.shape[0]
    l = math.ceil(win_s * fs)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h_out = np.zeros(n)
    for end in range(l, n):
        start = end - l
        seg = rgb[start:end]
        cn = seg / np.mean(seg, axis=0)
        s = proj @ cn.T
        h = s[0] + (np.std(s[0]) / (np.std(s[1]) + 1e-12)) * s[1]
        h_out[start:end] += h - np.mean(h)
    return bandpass(detrend(h_out, 100.0), fs, band[0], band[1], order=1)


def chrom(rgb: np.ndarray, fs: float, win_s: float = 1.6, band: tuple[float, float] = (0.7, 4.0)) -> np.ndarray:
    """CHROM pulse signal (50%-overlap Hann-windowed segments, as in the reference)."""
    n = rgb.shape[0]
    win = math.ceil(win_s * fs)
    win += win % 2
    half = win // 2
    nwin = math.floor((n - half) / half)
    sos = signal.butter(3, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    out = np.zeros(half * (nwin + 1))
    start = 0
    for _ in range(nwin):
        seg = rgb[start:start + win]
        norm = seg / np.mean(seg, axis=0)
        xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        xf, yf = signal.sosfiltfilt(sos, xs), signal.sosfiltfilt(sos, ys)
        s_win = (xf - (np.std(xf) / (np.std(yf) + 1e-12)) * yf) * signal.windows.hann(win)
        out[start:start + half] += s_win[:half]
        out[start + half:start + win] = s_win[half:]
        start += half
    if len(out) < n:  # pad the tail so the pulse aligns with the input length
        out = np.pad(out, (0, n - len(out)), mode="edge")
    return out[:n]


def green(rgb: np.ndarray, fs: float, band: tuple[float, float] = (0.7, 4.0)) -> np.ndarray:
    """Green-channel baseline (detrended, bandpassed)."""
    return bandpass(detrend(rgb[:, 1] - rgb[:, 1].mean(), 100.0), fs, band[0], band[1], order=3)


METHODS = {"pos": pos, "chrom": chrom, "green": green}


def fft_hr(x: np.ndarray, fs: float, band: tuple[float, float] = (0.75, 3.0), method: str = "peak") -> tuple[float, float]:
    """Dominant in-band frequency of ``x`` in BPM, with its spectral peak ratio.

    Zero-pads 4x beyond the next power of two (finer grid than the Toolbox). ``method='peak'`` takes the largest in-band
    periodogram bin; ``method='shs'`` takes the largest *sub-harmonic sum* ``P(f) + 0.5 P(2f) + 0.25 P(3f)``, which reduces
    octave errors when the fundamental is present but weaker than the second harmonic (power ratio above ~0.5); it cannot recover a fundamental
    that is absent. Returns ``(bpm, peak_ratio)`` where ``peak_ratio`` is the
    peak power over the mean in-band power (a signal-quality proxy).
    """
    nfft = 1 << (4 * len(x) - 1).bit_length()
    f, p = signal.periodogram(x, fs=fs, nfft=nfft, detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    fm, pm = f[m], p[m]
    if method == "shs":
        score = pm.copy()
        for k, w in ((2, 0.5), (3, 0.25)):
            score += w * np.interp(k * fm, f, p, right=0.0)
    elif method == "peak":
        score = pm
    else:
        raise ValueError(f"unknown hr method {method!r}")
    k = int(np.argmax(score))
    return float(fm[k] * 60.0), float(pm[k] / (np.mean(pm) + 1e-18))


def sliding_hr(x: np.ndarray, fs: float, win_s: float = 10.0, stride_s: float = 1.0,
               band: tuple[float, float] = (0.75, 3.0), method: str = "peak") -> tuple[np.ndarray, np.ndarray]:
    """HR (BPM) on sliding windows; returns (window-centre times in s, bpm)."""
    t, bpm, _ = sliding_hr_q(x, fs, win_s, stride_s, band, method)
    return t, bpm


def sliding_hr_q(x: np.ndarray, fs: float, win_s: float = 10.0, stride_s: float = 1.0,
                 band: tuple[float, float] = (0.75, 3.0), method: str = "peak") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`sliding_hr` but also returns the per-window spectral peak ratio."""
    w, s = int(round(win_s * fs)), max(1, int(round(stride_s * fs)))
    starts = np.arange(0, len(x) - w + 1, s)
    res = np.array([fft_hr(x[i:i + w], fs, band, method) for i in starts]).reshape(-1, 2)
    return (starts + w / 2) / fs, res[:, 0], res[:, 1]


def ppg_reference_hr(ppg: np.ndarray, fs: float, win_s: float, stride_s: float,
                     band: tuple[float, float] = (0.75, 3.0)) -> tuple[np.ndarray, np.ndarray]:
    """Reference HR from a contact PPG waveform using the exact same windowing as the estimates."""
    p = bandpass(detrend(ppg - np.mean(ppg), 100.0), fs, 0.7, 4.0, order=3)
    return sliding_hr(p, fs, win_s, stride_s, band)


def reference_windows(ppg: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reference HR per window from the contact PPG and a validity mask (peak ratio >= ``rppg.ref_min_peak_ratio``).

    The same mask is applied to every method, so a corrupted oximeter segment never counts against (or for) an estimator.
    """
    r = cfg["rppg"]
    p = bandpass(detrend(ppg - np.mean(ppg), 100.0), fs, 0.7, 4.0, order=3)
    _, bpm, ratio = sliding_hr_q(p, fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), "peak")
    return bpm, ratio >= float(r.get("ref_min_peak_ratio", 0.0))


def estimate(rgb: np.ndarray, fs: float, method: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``method`` and return (pulse, window times, bpm)."""
    pulse, t, bpm, _ = estimate_q(rgb, fs, method, cfg)
    return pulse, t, bpm


def estimate_q(rgb: np.ndarray, fs: float, method: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`estimate` but also returns the spectral peak ratio of every window (signal-quality proxy)."""
    r = cfg["rppg"]
    pulse = METHODS[method](rgb, fs, band=tuple(r["band_hz"])) if method == "green" else \
        METHODS[method](rgb, fs, r["pos_window_s"], tuple(r["band_hz"]))
    t, bpm, ratio = sliding_hr_q(pulse, fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), r.get("hr_method", "peak"))
    return pulse, t, bpm, ratio


def main() -> None:
    """CLI: estimate HR on a synthetic clip or a real video with ground truth, and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .datasets import load_clip

    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--clip", default="synthetic:0", help="'synthetic:<id>' or 'ubfc:subject<N>'")
    ap.add_argument("--methods", nargs="+", default=["pos", "chrom"], choices=list(METHODS))
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])

    clip = load_clip(args.clip, cfg)
    log.info("clip %s: %d frames @ %.1f fps", clip.name, len(clip.rgb), clip.fs)
    t_ref, hr_ref = ppg_reference_hr(clip.ppg, clip.fs, cfg["rppg"]["hr_window_s"], cfg["rppg"]["hr_stride_s"],
                                     tuple(cfg["rppg"]["hr_band_hz"]))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[1].plot(t_ref, hr_ref, "k", lw=2, label="reference (contact PPG)")
    for m in args.methods:
        pulse, t, bpm = estimate(clip.rgb, clip.fs, m, cfg)
        n = min(len(bpm), len(hr_ref))
        mae = float(np.mean(np.abs(bpm[:n] - hr_ref[:n])))
        log.info("%s: window MAE = %.2f BPM over %d windows", m, mae, n)
        ax[0].plot(np.arange(len(pulse)) / clip.fs, pulse / (np.std(pulse) + 1e-12), lw=0.8, label=m.upper())
        ax[1].plot(t, bpm, label=f"{m.upper()} (MAE {mae:.2f})")
    ppg_n = bandpass(detrend(clip.ppg - clip.ppg.mean()), clip.fs, 0.7, 4.0)
    ax[0].plot(np.arange(len(ppg_n)) / clip.fs, ppg_n / (np.std(ppg_n) + 1e-12) - 4, "k", lw=0.8, label="PPG (offset)")
    ax[0].set_xlim(0, 20)
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("pulse (a.u.)")
    ax[0].legend(ncol=3, fontsize=8)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("HR (BPM)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = out_path(cfg, "rppg_classical", "png", tag=clip.name.replace(":", "-"))
    fig.savefig(path, dpi=150)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
