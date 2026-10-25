"""Calibrate / evaluate the app's reliability rule ("show a number only when estimators agree and the spectrum is peaked").

Uses the per-window estimates and spectral peak ratios stored in a ``results_*.json`` written by ``src.evaluate``.
``--mode calibrate`` (validation JSON only) grid-searches rule parameters and writes the chosen values to ``configs/quality_calibrated.json``;
``--mode apply`` reports coverage/accuracy of the frozen rule on another JSON (e.g. the final test run). Never calibrate on test.

A window counts as *shown* when at least ``min_agree`` voter methods with peak ratio >= ``min_ratio`` lie within ``agree_bpm`` of their median;
the shown value is the mean of those methods. Metrics: coverage (fraction of valid windows shown), MAE of shown values, fraction within 5 BPM,
and the MAE of the same method set when always shown (no gating).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import out_path, register_claim, load_config, save_json  # noqa: E402


def windows(res: dict, voters: list[str]):
    """Yield per-window dicts {method: (bpm, q)} with the reference, for valid windows."""
    out = []
    for c in res["per_clip"]:
        wd = c["windows_data"]
        names = [v for v in voters if v in wd]
        if len(names) < 2:
            continue
        n = min(len(wd[v]["bpm"]) for v in names)
        for i in range(n):
            if not wd[names[0]]["valid"][i]:
                continue
            out.append({"ref": wd[names[0]]["ref"][i], "est": {v: (wd[v]["bpm"][i], wd[v]["q"][i]) for v in names}, "clip": c["spec"]})
    return out


def apply_rule(wins: list[dict], min_ratio: float, agree: float, min_agree: int) -> dict:
    """Coverage and accuracy of the agreement rule over ``wins``."""
    shown, errs = 0, []
    for w in wins:
        strong = {k: b for k, (b, q) in w["est"].items() if q >= min_ratio}
        if len(strong) < min_agree:
            continue
        med = np.median(list(strong.values()))
        ok = [b for b in strong.values() if abs(b - med) <= agree]
        if len(ok) >= min_agree:
            shown += 1
            errs.append(abs(np.mean(ok) - w["ref"]))
    errs = np.array(errs)
    return {"coverage": shown / max(len(wins), 1), "mae": float(errs.mean()) if len(errs) else float("nan"),
            "within5": float(np.mean(errs <= 5)) if len(errs) else float("nan"), "n_shown": shown, "n": len(wins)}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["calibrate", "apply"], required=True)
    ap.add_argument("--results", required=True, help="glob of a results_*.json")
    ap.add_argument("--voters", nargs="+", default=["rgb_green", "rgb_pos", "rgb_chrom"])
    ap.add_argument("--target-within5", type=float, default=0.9)
    args = ap.parse_args()
    res = json.loads(Path(sorted(glob.glob(args.results))[-1]).read_text())
    wins = windows(res, args.voters)
    cal = ROOT / "configs" / "quality_calibrated.json"
    cfg = load_config()
    if args.mode == "calibrate":
        assert res["split"] == "val", "calibration must use the validation split"
        best = None
        for ratio, agree, k in itertools.product([2, 3, 4, 5, 6, 8], [2, 3, 4, 6, 8], [2, 3]):
            if k > len(args.voters):
                continue
            m = apply_rule(wins, ratio, agree, k)
            if m["within5"] >= args.target_within5 and (best is None or m["coverage"] > best[1]["coverage"]):
                best = ((ratio, agree, k), m)
        if best is None:
            print("no rule reaches the target; falling back to the strictest setting")
            best = ((8, 2, min(3, len(args.voters))), apply_rule(wins, 8, 2, min(3, len(args.voters))))
        (ratio, agree, k), m = best
        cal.write_text(json.dumps({"min_peak_ratio": ratio, "agree_bpm": agree, "min_agree": k, "voters": args.voters, "val": m,
                                   "target_within5": args.target_within5}, indent=2))
        print("chosen on validation:", {"min_peak_ratio": ratio, "agree_bpm": agree, "min_agree": k}, m)
        cmdc = f'python scripts/calibrate_quality.py --mode calibrate --results "{args.results}"'
        for name, v in (("min_peak_ratio", ratio), ("agree_bpm", agree), ("min_agree", k)):
            register_claim(f"gate.{name}", v, cal, cmdc)
        for name, v in m.items():
            register_claim(f"gate.val.{name}", v, cal, cmdc)
        return
    rule = json.loads(cal.read_text())
    m = apply_rule(wins, rule["min_peak_ratio"], rule["agree_bpm"], rule["min_agree"])
    ungated = {v: float(np.mean([abs(w["est"][v][0] - w["ref"]) for w in wins if v in w["est"]])) for v in args.voters}
    path = save_json({"rule": rule, "applied": m, "ungated_mae": ungated, "results": res["split"]}, out_path(cfg, f"quality_rule_{res['split']}", "json"))
    print("frozen rule on", res["split"], m, "ungated MAE per voter", {k: round(v, 2) for k, v in ungated.items()})
    for k, v in m.items():
        register_claim(f"quality.{res['split']}.{k}", v, path, f"python scripts/calibrate_quality.py --mode apply --results \"{args.results}\"")
    for k, v in ungated.items():
        register_claim(f"quality.{res['split']}.ungated_mae.{k}", v, path, f"python scripts/calibrate_quality.py --mode apply --results \"{args.results}\"")


if __name__ == "__main__":
    main()
