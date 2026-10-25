"""Accuracy-validation maths for CARDION (pure functions: no camera, no files, no web server - unit-tested in ``tests/test_validation.py``).

The question answered is: *how close are CARDION's reliable heart-rate readings to an independent reference monitor, and how often does
CARDION give a reliable reading at all?*  Both halves are always reported together, because a system that only answers when it is easy can
look very accurate while being rarely available.

Definitions
-----------
* A **CARDION row** is the one-per-second result of the app: ``{"t": seconds since the first camera frame, "status": ..., "bpm": ...,
  <method>: bpm of each individual method}``.  Only rows with ``status == "ok"`` are readings CARDION is willing to show.
* A **reference entry** is one value read from the reference monitor: ``{"id", "t" (session seconds the value refers to), "value", ...}``.
* A reference entry is **paired** with the nearest-in-time reliable CARDION row if that row lies within ``tol_s`` seconds of it.  Entries
  with no reliable CARDION row in range are *unpaired*: they are excluded from the error statistics but counted, with the reason, in the
  availability statistics.  Nothing is interpolated: neither the reference nor CARDION is ever estimated between samples.
* The tolerance exists because a manually read value is not instantaneous (the monitor averages over a few seconds, the person types it a
  few seconds later, CARDION's window is 10 s long).  Results are therefore also shown for a range of tolerances.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

MIN_BPM, MAX_BPM = 30.0, 220.0            # values outside this range are rejected as typing mistakes
DEFAULT_TOL_S = 5.0
SENSITIVITY_TOLS = (3.0, 5.0, 10.0)
METHOD_KEYS = ("rgb_pos", "rgb_green", "rgb_chrom", "event_green", "event_pos", "snn", "ann")
MIN_PAIRS_LOA = 20                         # Bland-Altman limits of agreement need a reasonable number of pairs
MIN_PARTICIPANTS_CI = 8                    # a participant-level confidence interval needs a reasonable number of people


def valid_bpm(v) -> bool:
    """True for a finite number in the physiologically plausible range."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and MIN_BPM <= v <= MAX_BPM


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def active_entries(entries: Iterable[dict]) -> list[dict]:
    """Reference entries that were not deleted by the user."""
    return [e for e in entries if not e.get("deleted")]


# ------------------------------------------------------------------ pairing
def pair_readings(entries: list[dict], rows: list[dict], tol_s: float = DEFAULT_TOL_S) -> tuple[list[dict], list[dict]]:
    """Pair each reference entry with the nearest reliable CARDION row within ``tol_s`` seconds. Returns ``(pairs, unpaired)``."""
    ok_rows = [r for r in rows if r.get("status") == "ok" and _finite(r.get("bpm"))]
    pairs, unpaired = [], []
    for e in active_entries(entries):
        near = [r for r in ok_rows if abs(r["t"] - e["t"]) <= tol_s]
        if near:
            r = min(near, key=lambda x: (abs(x["t"] - e["t"]), x["t"]))
            err = float(r["bpm"]) - float(e["value"])
            pairs.append({"entry_id": e["id"], "ref_t": float(e["t"]), "ref_bpm": float(e["value"]), "cardion_t": float(r["t"]), "cardion_bpm": float(r["bpm"]),
                          "dt_s": float(r["t"] - e["t"]), "error": err, "abs_error": abs(err)})
            continue
        around = [r for r in rows if abs(r["t"] - e["t"]) <= tol_s]
        if around:
            nearest = min(around, key=lambda x: abs(x["t"] - e["t"]))
            reason = nearest.get("status", "unknown")
        else:
            reason = "no_estimate_in_window"
        unpaired.append({"entry_id": e["id"], "ref_t": float(e["t"]), "ref_bpm": float(e["value"]), "reason": reason})
    return pairs, unpaired


# ------------------------------------------------------------------ accuracy
def accuracy_metrics(pairs: list[dict]) -> dict:
    """MAE, RMSE, signed bias (CARDION minus reference), median/max absolute error, share within 1/3/5 BPM. All over *paired* readings only."""
    n = len(pairs)
    if n == 0:
        return {"n_pairs": 0}
    err = np.array([p["error"] for p in pairs], float)
    ae = np.abs(err)
    ref = np.array([p["ref_bpm"] for p in pairs], float)
    out = {"n_pairs": n, "mae_bpm": float(ae.mean()), "rmse_bpm": float(np.sqrt((err ** 2).mean())), "bias_bpm": float(err.mean()),
           "median_abs_error_bpm": float(np.median(ae)), "max_abs_error_bpm": float(ae.max()),
           "mape_percent": float(np.mean(ae / ref) * 100.0),
           "within_1_bpm": float(np.mean(ae <= 1.0)), "within_3_bpm": float(np.mean(ae <= 3.0)), "within_5_bpm": float(np.mean(ae <= 5.0)),
           "error_sd_bpm": float(err.std(ddof=1)) if n > 1 else None, "ref_range_bpm": [float(ref.min()), float(ref.max())],
           "ref_sd_bpm": float(ref.std(ddof=1)) if n > 1 else None,
           "mean_abs_pair_offset_s": float(np.mean([abs(p["dt_s"]) for p in pairs])), "max_abs_pair_offset_s": float(max(abs(p["dt_s"]) for p in pairs))}
    cd = np.array([p["cardion_bpm"] for p in pairs], float)
    if n >= 10 and ref.std() > 0 and cd.std() > 0:
        out["pearson_r"] = float(np.corrcoef(ref, cd)[0, 1])
        out["pearson_note"] = ("a narrow reference range (resting people) makes r small even when errors are small; judge accuracy by the errors above"
                               if ref.std(ddof=1) < 5 else "")
    else:
        out["pearson_r"] = None
    return out


def bland_altman(pairs: list[dict]) -> dict:
    """Bias and 95 % limits of agreement - only with at least ``MIN_PAIRS_LOA`` pairs; otherwise explains why they are not given."""
    n = len(pairs)
    if n < MIN_PAIRS_LOA:
        return {"computed": False, "reason": f"needs at least {MIN_PAIRS_LOA} paired readings (have {n}); limits from fewer pairs would be misleading"}
    d = np.array([p["error"] for p in pairs], float)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {"computed": True, "bias_bpm": bias, "sd_bpm": sd, "loa_low_bpm": bias - 1.96 * sd, "loa_high_bpm": bias + 1.96 * sd,
            "caveat": "pairs from the same person are not independent, so these limits are optimistic; see the participant-level summary"}


def sensitivity_to_tolerance(entries: list[dict], rows: list[dict], tols: Iterable[float] = SENSITIVITY_TOLS) -> list[dict]:
    """Does the accuracy depend on the pairing window? One row per tolerance (seconds)."""
    n_entries = len(active_entries(entries))
    out = []
    for tol in tols:
        pairs, _ = pair_readings(entries, rows, tol)
        m = accuracy_metrics(pairs)
        out.append({"tolerance_s": tol, "n_entries": n_entries, "n_pairs": len(pairs), "paired_share": (len(pairs) / n_entries) if n_entries else None,
                    "mae_bpm": m.get("mae_bpm"), "within_5_bpm": m.get("within_5_bpm")})
    return out


def ungated_context(entries: list[dict], rows: list[dict], tol_s: float = DEFAULT_TOL_S) -> dict:
    """What would each individual method have said at every reference entry, *ignoring* the reliability gate?  Shows what the gate hides."""
    out = {}
    for m in METHOD_KEYS:
        errs = []
        for e in active_entries(entries):
            near = [r for r in rows if abs(r["t"] - e["t"]) <= tol_s and _finite(r.get(m))]
            if near:
                r = min(near, key=lambda x: abs(x["t"] - e["t"]))
                errs.append(float(r[m]) - float(e["value"]))
        if errs:
            a = np.abs(errs)
            out[m] = {"n": len(errs), "mae_bpm": float(a.mean()), "bias_bpm": float(np.mean(errs)), "within_5_bpm": float(np.mean(a <= 5.0))}
    return out


# ------------------------------------------------------------------ availability and stability of CARDION itself
def availability(rows: list[dict], min_seconds: float = 8.0) -> dict:
    """How often CARDION produced a reliable reading, how long it took, why it withheld, and how steady the shown number was."""
    n = len(rows)
    if n == 0:
        return {"n_estimates": 0}
    ok = [r for r in rows if r.get("status") == "ok" and _finite(r.get("bpm"))]
    first = min((r["t"] for r in ok), default=None)
    post = [r for r in rows if r["t"] >= min_seconds]
    post_ok = [r for r in post if r.get("status") == "ok" and _finite(r.get("bpm"))]
    dts = np.diff([r["t"] for r in rows]) if n > 1 else np.array([1.0])
    step = float(np.median(dts)) if len(dts) else 1.0
    # episodes of withheld readings after the first reliable one
    episodes, run = [], 0
    started = False
    for r in rows:
        good = r.get("status") == "ok" and _finite(r.get("bpm"))
        if good:
            started = True
            if run:
                episodes.append(run * step)
                run = 0
        elif started:
            run += 1
    if run:
        episodes.append(run * step)
    reasons: dict[str, int] = {}
    for r in post:
        if not (r.get("status") == "ok" and _finite(r.get("bpm"))):
            reasons[r.get("status", "unknown")] = reasons.get(r.get("status", "unknown"), 0) + 1
    jumps = [abs(b["bpm"] - a["bpm"]) for a, b in zip(rows, rows[1:])
             if a.get("status") == "ok" and b.get("status") == "ok" and _finite(a.get("bpm")) and _finite(b.get("bpm")) and (b["t"] - a["t"]) <= 1.5 * step]
    vals = [r["bpm"] for r in ok]
    return {"n_estimates": n, "session_seconds": float(rows[-1]["t"]), "n_reliable": len(ok), "time_to_first_reliable_s": first,
            "reliable_share_whole_session": len(ok) / n,
            "reliable_share_after_warmup": (len(post_ok) / len(post)) if post else None, "warmup_s": min_seconds,
            "withheld_episodes": len(episodes), "withheld_episode_mean_s": float(np.mean(episodes)) if episodes else 0.0,
            "withheld_episode_max_s": float(max(episodes)) if episodes else 0.0, "withheld_reasons_after_warmup": reasons,
            "reliable_bpm_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "mean_abs_change_per_second_bpm": float(np.mean(jumps)) if jumps else None, "max_change_per_second_bpm": float(max(jumps)) if jumps else None}


def reference_availability(entries: list[dict], pairs: list[dict], unpaired: list[dict]) -> dict:
    """How many reference entries could be compared, and why the others could not."""
    n = len(active_entries(entries))
    reasons: dict[str, int] = {}
    for u in unpaired:
        reasons[u["reason"]] = reasons.get(u["reason"], 0) + 1
    return {"n_reference_entries": n, "n_paired": len(pairs), "paired_share": (len(pairs) / n) if n else None, "unpaired_reasons": reasons}


# ------------------------------------------------------------------ a whole session
def session_report(entries: list[dict], rows: list[dict], tol_s: float = DEFAULT_TOL_S, min_seconds: float = 8.0) -> dict:
    """Everything the validation page shows for one session, computed from the reference entries and the saved CARDION rows."""
    pairs, unpaired = pair_readings(entries, rows, tol_s)
    return {"tolerance_s": tol_s, "accuracy": accuracy_metrics(pairs), "bland_altman": bland_altman(pairs),
            "reference_availability": reference_availability(entries, pairs, unpaired), "cardion_availability": availability(rows, min_seconds),
            "tolerance_sensitivity": sensitivity_to_tolerance(entries, rows), "ungated_context": ungated_context(entries, rows, tol_s),
            "pairs": pairs, "unpaired": unpaired}


def headline(report: dict) -> str:
    """One honest sentence: accuracy and availability together."""
    a, ra, ca = report["accuracy"], report["reference_availability"], report["cardion_availability"]
    if not ra["n_reference_entries"]:
        return "No reference readings were entered, so accuracy cannot be assessed. CARDION availability is shown below."
    avail = ca.get("reliable_share_after_warmup")
    avail_txt = f"CARDION showed a reliable reading for {100 * avail:.0f}% of the session after warm-up" if avail is not None else "CARDION availability could not be computed"
    if a["n_pairs"] == 0:
        return (f"None of the {ra['n_reference_entries']} reference readings could be compared (CARDION had no reliable reading within +/-{report['tolerance_s']:.0f} s "
                f"of any of them). {avail_txt}.")
    noun = "paired reading" if a["n_pairs"] == 1 else "paired readings"
    return (f"Mean absolute error {a['mae_bpm']:.1f} BPM over {a['n_pairs']} {noun} (bias {a['bias_bpm']:+.1f} BPM); "
            f"{ra['n_paired']} of {ra['n_reference_entries']} reference readings ({100 * ra['paired_share']:.0f}%) had a reliable CARDION reading within +/-{report['tolerance_s']:.0f} s. "
            f"{avail_txt}.")


# ------------------------------------------------------------------ across sessions / participants
def participant_level(sessions: list[dict]) -> list[dict]:
    """One row per participant from several session reports: that person's pairs pooled, so each person counts once in the summary."""
    by: dict[str, list[dict]] = {}
    for s in sessions:
        by.setdefault(s["participant"], []).extend(s["pairs"])
    rows = []
    for pid, pairs in sorted(by.items()):
        m = accuracy_metrics(pairs)
        if m["n_pairs"]:
            rows.append({"participant": pid, "n_pairs": m["n_pairs"], "mae_bpm": m["mae_bpm"], "bias_bpm": m["bias_bpm"], "within_5_bpm": m["within_5_bpm"]})
    return rows


def bootstrap_ci(values: list[float], n_boot: int = 4000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap CI of the mean over *participants* (each person is one observation). Refuses to run with too few people."""
    n = len(values)
    if n < MIN_PARTICIPANTS_CI:
        return {"computed": False, "reason": f"needs at least {MIN_PARTICIPANTS_CI} participants (have {n}); an interval from fewer people would be misleading"}
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = rng.choice(v, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"computed": True, "mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "level": level, "n_participants": n, "n_boot": n_boot, "seed": seed}


def study_summary(sessions: list[dict], seed: int = 0) -> dict:
    """Pool several session reports (each ``{"participant", "pairs", "availability", ...}``) into participant-level results."""
    prows = participant_level(sessions)
    all_pairs = [p for s in sessions for p in s["pairs"]]
    n_ref = sum(s.get("n_reference_entries", 0) for s in sessions)
    shares = [s["reliable_share_after_warmup"] for s in sessions if s.get("reliable_share_after_warmup") is not None]
    return {"n_sessions": len(sessions), "n_participants": len({s["participant"] for s in sessions}), "n_participants_with_pairs": len(prows),
            "n_reference_entries": n_ref, "n_pairs": len(all_pairs), "paired_share": (len(all_pairs) / n_ref) if n_ref else None,
            "pooled_pair_metrics": accuracy_metrics(all_pairs), "participant_rows": prows,
            "participant_mean_of_mae": float(np.mean([r["mae_bpm"] for r in prows])) if prows else None,
            "participant_median_of_mae": float(np.median([r["mae_bpm"] for r in prows])) if prows else None,
            "participant_mae_ci": bootstrap_ci([r["mae_bpm"] for r in prows], seed=seed),
            "participant_bias_ci": bootstrap_ci([r["bias_bpm"] for r in prows], seed=seed),
            "mean_reliable_share_after_warmup": float(np.mean(shares)) if shares else None,
            "pooled_bland_altman": bland_altman(all_pairs)}
