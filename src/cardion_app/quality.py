"""Signal-quality and reliability logic for the CARDION app.

A heart rate is shown only when (1) enough data has been collected, (2) a face is tracked, (3) the face is not moving much,
(4) lighting is usable and stable, (5) the pulse spectrum has a clear peak and (6) independent estimators agree.
Otherwise the app says *why* it is not showing a number. Thresholds live in ``configs/default.yaml`` (``app.quality``) and are
calibrated on validation data only (see ``scripts/calibrate_quality.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATUS_TEXT = {
    "ok": "Reliable estimate",
    "collecting": "Collecting data",
    "no_face": "No face found",
    "motion": "Too much movement",
    "dark": "Lighting too dark",
    "bright": "Lighting too bright",
    "unstable_light": "Lighting is changing",
    "weak_signal": "Pulse signal too weak",
    "disagree": "Methods disagree",
    "multi_face": "More than one face in view",
}


@dataclass
class QualityParams:
    """Thresholds (defaults are overridden from the config)."""
    min_seconds: float = 8.0          # data needed before the first estimate
    min_face_fraction: float = 0.8    # fraction of recent frames with a tracked face
    max_motion: float = 0.04          # face-centre std (3 s) relative to face width
    min_luma: float = 35.0            # ROI mean luminance (0-255)
    max_luma: float = 235.0
    max_luma_var: float = 0.06        # std/mean of ROI luminance over the window (auto-exposure, flicker)
    min_peak_ratio: float = 3.0       # spectral peak ratio of the contributing estimators
    agree_bpm: float = 6.0            # estimators within this many BPM of the median agree
    min_agree: int = 2                # how many independent estimators must agree
    voters: tuple = ("rgb_green", "rgb_pos", "rgb_chrom")   # RGB methods vote (best on real video); event methods are shown for comparison


@dataclass
class Assessment:
    """Result of one reliability assessment."""
    status: str
    text: str
    bpm: float | None
    spread: float | None
    confidence: float
    detail: dict = field(default_factory=dict)


def assess(estimates: dict[str, dict], env: dict, params: QualityParams) -> Assessment:
    """Combine environment checks and estimator agreement into a status and (maybe) a heart rate.

    ``estimates[name]`` = ``{"bpm": float, "peak_ratio": float}``; ``env`` has ``seconds``, ``face_fraction``, ``motion``,
    ``luma``, ``luma_var``.
    """
    d = {"seconds": env["seconds"], "face_fraction": env["face_fraction"], "motion": env["motion"], "luma": env["luma"],
         "luma_var": env["luma_var"], "multi_face": env.get("multi_face", 0.0)}
    if env.get("multi_face", 0.0) > 0.3:                              # two people in view: never mix their pulses, never show a number
        return Assessment("multi_face", STATUS_TEXT["multi_face"], None, None, 0.1, d)
    if env["seconds"] >= 3.0 and env["face_fraction"] < 0.2:          # nobody in view: say so at once instead of "collecting" for 8 s
        return Assessment("no_face", STATUS_TEXT["no_face"], None, None, 0.1, d)
    if env["seconds"] < params.min_seconds:
        return Assessment("collecting", STATUS_TEXT["collecting"], None, None, min(env["seconds"] / params.min_seconds, 1.0) * 0.3, d)
    for status, bad in (("no_face", env["face_fraction"] < params.min_face_fraction), ("motion", env["motion"] > params.max_motion),
                        ("dark", env["luma"] < params.min_luma), ("bright", env["luma"] > params.max_luma),
                        ("unstable_light", env["luma_var"] > params.max_luma_var)):
        if bad:
            return Assessment(status, STATUS_TEXT[status], None, None, 0.1, d)
    voters = {k: v for k, v in estimates.items() if k in params.voters and np.isfinite(v.get("bpm", np.nan))}
    strong = {k: v for k, v in voters.items() if v["peak_ratio"] >= params.min_peak_ratio}
    d["voters"] = len(voters)
    d["strong"] = len(strong)
    if not strong:
        return Assessment("weak_signal", STATUS_TEXT["weak_signal"], None, None, 0.15, d)
    med = float(np.median([v["bpm"] for v in strong.values()]))
    agree = {k: v for k, v in strong.items() if abs(v["bpm"] - med) <= params.agree_bpm}
    d["agree"] = len(agree)
    if len(agree) < min(params.min_agree, len(strong)) or len(agree) < params.min_agree:
        return Assessment("disagree", STATUS_TEXT["disagree"], None, None, 0.25, d)
    bpm = float(np.mean([v["bpm"] for v in agree.values()]))
    spread = float(np.std([v["bpm"] for v in agree.values()]))
    conf = float(np.clip(0.4 + 0.15 * len(agree) + 0.03 * np.mean([v["peak_ratio"] for v in agree.values()]) - 0.05 * spread, 0, 1))
    return Assessment("ok", STATUS_TEXT["ok"], bpm, spread, conf, d)
