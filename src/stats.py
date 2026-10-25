"""Metrics and statistics: MAE/RMSE/MAPE/Pearson/Lin's CCC, Bland-Altman, SNR/MACC, bootstrap CIs, Wilcoxon."""
from __future__ import annotations

import argparse

import numpy as np
from scipy import signal, stats


def hr_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Agreement metrics between predicted and reference HR (BPM) arrays."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    err = pred - ref
    r = float(np.corrcoef(pred, ref)[0, 1]) if len(pred) > 2 and pred.std() > 0 and ref.std() > 0 else float("nan")
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(100 * np.mean(np.abs(err) / np.maximum(ref, 1e-9))), "pearson_r": r, "ccc": lins_ccc(pred, ref),
            **bland_altman(pred, ref)}


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my, vx, vy = x.mean(), y.mean(), x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


def bland_altman(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Bias and 95% limits of agreement of (pred - ref)."""
    d = np.asarray(pred, float) - np.asarray(ref, float)
    bias, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return {"ba_bias": bias, "ba_loa_low": bias - 1.96 * sd, "ba_loa_high": bias + 1.96 * sd}


def snr_db(pulse: np.ndarray, fs: float, hr_bpm: float, band: tuple[float, float] = (0.75, 3.0), tol_bpm: float = 6.0) -> float:
    """de Haan-style SNR: power around the reference HR and its 2nd harmonic vs the rest of the band (dB)."""
    f, p = signal.periodogram(pulse, fs=fs, nfft=1 << (4 * len(pulse) - 1).bit_length(), detrend=False)
    f0, tol = hr_bpm / 60.0, tol_bpm / 60.0
    in_band = (f >= band[0]) & (f <= band[1])
    sig = ((np.abs(f - f0) <= tol) | (np.abs(f - 2 * f0) <= tol)) & in_band
    noise = in_band & ~sig
    return float(10 * np.log10((p[sig].sum() + 1e-18) / (p[noise].sum() + 1e-18)))


def macc(pred: np.ndarray, ref: np.ndarray, fs: float, max_lag_s: float = 0.5) -> float:
    """Maximum absolute cross-correlation within +/-``max_lag_s`` (signals z-scored)."""
    n = min(len(pred), len(ref))
    a, b = (pred[:n] - pred[:n].mean()) / (pred[:n].std() + 1e-12), (ref[:n] - ref[:n].mean()) / (ref[:n].std() + 1e-12)
    best = 0.0
    for lag in range(-int(max_lag_s * fs), int(max_lag_s * fs) + 1):
        x, y = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        best = max(best, abs(float(np.mean(x * y))))
    return best


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0, stat=np.mean) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``stat`` over *subjects* (``values`` has one entry per subject)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(stat(v)), float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired two-sided Wilcoxon signed-rank test on per-subject errors (a vs b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return {"statistic": 0.0, "p_value": 1.0, "median_diff": 0.0, "n": int(len(d))}
    res = stats.wilcoxon(a, b)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "median_diff": float(np.median(d)), "n": int(len(d))}


def main() -> None:
    """CLI: self-test of the statistics on simulated data."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    rng = np.random.default_rng(0)
    ref = rng.uniform(60, 110, 200)
    pred = ref + rng.normal(1.0, 3.0, 200)
    print("hr_metrics:", {k: round(v, 3) for k, v in hr_metrics(pred, ref).items()})
    print("bootstrap MAE CI:", bootstrap_ci(np.abs(pred - ref)))
    print("wilcoxon vs worse:", paired_wilcoxon(np.abs(pred - ref), np.abs(pred + 5 - ref)))


if __name__ == "__main__":
    main()
