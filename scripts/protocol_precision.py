"""How precisely can N participants pin down CARDION's average error? (planning table for docs/VALIDATION_PROTOCOL.md)

    python scripts/protocol_precision.py

Takes the between-participant spread of the app's per-subject error from the existing validation analysis (``out/discrepancy_*.json``, produced by
``scripts/analyze_discrepancy.py``) as the planning value, and prints the expected half-width of a 95 % t-interval for the mean of per-participant
MAE: ``t(N-1, 0.975) * SD / sqrt(N)``. The SD is an assumption taken from a different dataset (office webcam clips before/after activity), not a
prediction for the user's own setup. Output: ``out/protocol_precision_<time>.json``.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import timestamp  # noqa: E402


def main() -> None:
    f = sorted(glob.glob(str(ROOT / "out" / "discrepancy_*.json")))[-1]
    d = json.loads(Path(f).read_text())
    maes = [v["mae_pooled"] for v in d["per_subject_gated"].values() if v["mae_pooled"] is not None]
    sd = float(np.std(maes, ddof=1))
    rows = [{"n_participants": n, "ci_half_width_bpm": float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))} for n in (5, 8, 10, 15, 20, 30)]
    out = {"source": Path(f).name, "n_subjects_in_source": len(maes), "planning_sd_of_participant_mae_bpm": sd, "rows": rows,
           "note": "assumption: per-participant MAE varies between people about as it did in the validation recordings; real SD is unknown until the pilot is run"}
    print(f"planning SD of per-participant MAE = {sd:.2f} BPM (from {len(maes)} validation subjects, {Path(f).name})")
    for r in rows:
        print(f"  N={r['n_participants']:>2}: 95% CI half-width about +/-{r['ci_half_width_bpm']:.2f} BPM")
    p = ROOT / "out" / f"protocol_precision_{timestamp()}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
