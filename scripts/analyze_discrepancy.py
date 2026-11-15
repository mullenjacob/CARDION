"""Why do the live app's readings agree with a watch within about 1 BPM while the research tables show much larger errors?

    python scripts/analyze_discrepancy.py

Uses only artifacts that already exist: the per-window estimates stored in the *validation* results (``out/results_mcd-val_*.json``, 12 subjects,
recorded before/after activity, reference = pulse-oximeter-derived rate). Nothing is retrained, no threshold is tuned, and the frozen test split
is only *quoted* from its existing results file, never re-analysed. It separates the explanations that can be separated with these data:

1. which estimator produces the number the app shows (``app.quality.voters``) - the SNN/ANN/event methods do not;
2. the reliability gate: accuracy of the shown numbers versus how often a number is shown (coverage);
3. selection vs averaging: on the windows the gate lets through, how does each individual method do, and how does the consensus do;
4. heart-rate range, rest vs after-activity, per-subject coverage, and the best-quality windows;
5. what the data cannot tell (lighting, camera, compression, synchronisation with a watch) - listed, not guessed.

Output: ``out/discrepancy_<time>.json`` and ``.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import timestamp  # noqa: E402

VOTERS = ["rgb_green", "rgb_pos", "rgb_chrom"]
OTHERS = {"event_green": "event_green@0.05", "event_pos": "event_pos@0.05", "snn": "snn_th0.05", "ann": "ann_th0.05"}


def load_windows(res: dict) -> list[dict]:
    """One dict per valid window: subject, condition (before/after), reference, {method: (bpm, peak_ratio)} for every method incl. seed-averaged nets."""
    out = []
    for c in res["per_clip"]:
        wd = c["windows_data"]
        subj, cond = c["spec"].split(":")[1].split("_")
        n = min(len(wd[v]["bpm"]) for v in VOTERS)
        for i in range(n):
            if not wd["rgb_pos"]["valid"][i]:
                continue
            est = {v: (wd[v]["bpm"][i], wd[v]["q"][i]) for v in VOTERS}
            for name, key in OTHERS.items():
                if key in wd:
                    est[name] = (wd[key]["bpm"][i], wd[key]["q"][i])
                else:
                    seeds = [k for k in wd if k.startswith(key + "_s")]
                    if seeds:
                        est[name] = (float(np.mean([wd[k]["bpm"][i] for k in seeds])), float(np.mean([wd[k]["q"][i] for k in seeds])))
            out.append({"subject": subj, "cond": cond, "ref": wd["rgb_pos"]["ref"][i], "est": est})
    return out


def gate(w: dict, rule: dict) -> float | None:
    """The app's rule (``Quality.assess``): at least ``min_agree`` strong voters within ``agree_bpm`` of their median -> their mean; otherwise None."""
    strong = {k: b for k, (b, q) in w["est"].items() if k in rule["voters"] and q >= rule["min_peak_ratio"]}
    if len(strong) < rule["min_agree"]:
        return None
    med = float(np.median(list(strong.values())))
    ok = [b for b in strong.values() if abs(b - med) <= rule["agree_bpm"]]
    return float(np.mean(ok)) if len(ok) >= rule["min_agree"] else None


def subject_mean(wins: list[dict], errs: list[float]) -> float | None:
    """Mean over subjects of each subject's MAE (how the research tables are computed: every person counts once)."""
    by: dict[str, list[float]] = {}
    for w, e in zip(wins, errs):
        by.setdefault(w["subject"], []).append(e)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def stats(wins: list[dict], pred: list[float | None]) -> dict:
    """Coverage and error of a per-window prediction list (None = no number shown)."""
    idx = [i for i, p in enumerate(pred) if p is not None]
    errs = [abs(pred[i] - wins[i]["ref"]) for i in idx]
    sub = [wins[i] for i in idx]
    return {"n_windows": len(wins), "n_shown": len(idx), "coverage": len(idx) / max(len(wins), 1), "mae_pooled": float(np.mean(errs)) if errs else None,
            "mae_subject_mean": subject_mean(sub, errs), "within_5": float(np.mean(np.array(errs) <= 5)) if errs else None,
            "within_3": float(np.mean(np.array(errs) <= 3)) if errs else None, "within_1": float(np.mean(np.array(errs) <= 1)) if errs else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", default="out/results_mcd-val_val_all_*.json", help="validation results (never the test split)")
    a = ap.parse_args()
    val_path = sorted(glob.glob(str(ROOT / a.val)))[-1]
    res = json.loads(Path(val_path).read_text())
    assert res["split"] == "val" and not res.get("final"), "this analysis must not use the frozen test split"
    cal = json.loads((ROOT / "configs" / "quality_calibrated.json").read_text(encoding="utf-8-sig"))
    rule = {k: cal[k] for k in ("min_peak_ratio", "agree_bpm", "min_agree", "voters")}
    wins = load_windows(res)
    n_subj = len({w["subject"] for w in wins})
    out: dict = {"command": "python scripts/analyze_discrepancy.py", "source_validation_file": Path(val_path).name, "rule": rule, "n_windows": len(wins), "n_subjects": n_subj}

    # 1+2: individual methods without any gate
    methods = {}
    for m in VOTERS + list(OTHERS):
        pred = [w["est"][m][0] if m in w["est"] else None for w in wins]
        methods[m] = stats(wins, pred)
    out["ungated_individual"] = methods
    forced = [float(np.mean([w["est"][v][0] for v in VOTERS])) for w in wins]
    out["ungated_rgb_average_every_window"] = stats(wins, forced)

    # the app's gated number
    shown = [gate(w, rule) for w in wins]
    out["app_gated"] = stats(wins, shown)

    # 3: selection vs averaging - individual methods on exactly the windows the gate lets through
    keep = [i for i, s in enumerate(shown) if s is not None]
    sel = [wins[i] for i in keep]
    on_shown = {}
    for m in VOTERS + list(OTHERS):
        errs = [abs(w["est"][m][0] - w["ref"]) for w in sel if m in w["est"]]
        on_shown[m] = {"mae_pooled": float(np.mean(errs)) if errs else None, "within_5": float(np.mean(np.array(errs) <= 5)) if errs else None}
    on_shown["app_consensus"] = {"mae_pooled": out["app_gated"]["mae_pooled"], "within_5": out["app_gated"]["within_5"]}
    out["individual_methods_on_gate_passing_windows"] = on_shown

    # 4: strata
    refs = np.array([w["ref"] for w in wins])
    cuts = [float(np.percentile(refs, 33)), float(np.percentile(refs, 67))]
    def stratum(sel_fn) -> dict:
        idx = [i for i, w in enumerate(wins) if sel_fn(w)]
        sw = [wins[i] for i in idx]
        return {"gated": stats(sw, [shown[i] for i in idx]), "rgb_pos_ungated": stats(sw, [w["est"]["rgb_pos"][0] for w in sw])}
    out["by_reference_hr"] = {f"<{cuts[0]:.0f}": stratum(lambda w: w["ref"] < cuts[0]), f"{cuts[0]:.0f}-{cuts[1]:.0f}": stratum(lambda w: cuts[0] <= w["ref"] < cuts[1]),
                              f">={cuts[1]:.0f}": stratum(lambda w: w["ref"] >= cuts[1])}
    out["by_condition"] = {c: stratum(lambda w, c=c: w["cond"] == c) for c in ("before", "after")}
    per_subj = {}
    for s in sorted({w["subject"] for w in wins}):
        idx = [i for i, w in enumerate(wins) if w["subject"] == s]
        per_subj[s] = stats([wins[i] for i in idx], [shown[i] for i in idx])
    out["per_subject_gated"] = {s: {"coverage": v["coverage"], "mae_pooled": v["mae_pooled"]} for s, v in per_subj.items()}
    out["subjects_with_no_shown_window"] = [s for s, v in per_subj.items() if v["n_shown"] == 0]

    # best-quality windows: weakest voter's peak ratio in the top quartile (a rank filter, nothing is tuned)
    minq = np.array([min(w["est"][v][1] for v in VOTERS) for w in wins])
    top = [i for i in range(len(wins)) if minq[i] >= np.percentile(minq, 75)]
    out["top_quartile_signal_quality_windows"] = {"rgb_pos_ungated": stats([wins[i] for i in top], [wins[i]["est"]["rgb_pos"][0] for i in top]),
                                                  "app_gated": stats([wins[i] for i in top], [shown[i] for i in top]), "n_windows": len(top)}

    # quoted frozen test numbers (read from the existing results file, not recomputed)
    tf = sorted(glob.glob(str(ROOT / "out" / "results_mcd-test_final_*.json")))
    quoted = {}
    if tf:
        tres = json.loads(Path(tf[-1]).read_text())
        quoted["file"] = Path(tf[-1]).name
        quoted["rows_mae_subject_mean"] = {r["method"]: round(r["mae"], 2) for r in tres["rows"]}
    qr = sorted(glob.glob(str(ROOT / "out" / "quality_rule_test_*.json")))
    if qr:
        q = json.loads(Path(qr[-1]).read_text())
        quoted["quality_rule_test_file"] = Path(qr[-1]).name
        quoted["app_gated_on_frozen_test"] = q["applied"]
        quoted["ungated_voters_on_frozen_test"] = q["ungated_mae"]
    out["quoted_frozen_test_numbers"] = quoted
    out["cannot_be_separated_with_these_data"] = [
        "lighting and camera: every MCD clip is a Full-HD webcam recording in office/room light; the live tests used a different camera and room (no way to compare from these files)",
        "recording compression: MCD videos are compressed files; the live path sees uncompressed webcam frames (the earlier compression study in out/ measures this effect separately)",
        "reference synchronisation: MCD reference rates come from a pulse oximeter on the same recording clock; a hand-read watch value is delayed and averaged (this is what validation mode's pairing window and sensitivity table address)",
        "consumer-watch reference error: not measurable here",
        "the live estimates were informal and not blinded, and the sample was a few people at rest (no activity condition)"]

    ts = timestamp()
    (ROOT / "out" / f"discrepancy_{ts}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "out" / f"discrepancy_{ts}.md").write_text(to_markdown(out), encoding="utf-8")
    print(to_markdown(out))
    print("saved", ROOT / "out" / f"discrepancy_{ts}.json")


def _f(v, d=2) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def to_markdown(o: dict) -> str:
    """Readable tables of the analysis."""
    L = [f"# Why live readings can look better than the research tables\n", f"Source: `{o['source_validation_file']}` ({o['n_subjects']} validation subjects, {o['n_windows']} valid 10 s windows). Rule: {o['rule']}\n",
         "## 1. Individual methods, no gate (every window, MAE in BPM)\n", "| method | MAE pooled | MAE (subject mean) | within 5 BPM |", "|---|---|---|---|"]
    for m, v in o["ungated_individual"].items():
        L.append(f"| {m} | {_f(v['mae_pooled'])} | {_f(v['mae_subject_mean'])} | {100 * v['within_5']:.0f}% |")
    f = o["ungated_rgb_average_every_window"]
    L.append(f"| average of the 3 RGB methods, every window | {_f(f['mae_pooled'])} | {_f(f['mae_subject_mean'])} | {100 * f['within_5']:.0f}% |")
    g = o["app_gated"]
    L += ["\n## 2. What the app displays (gated consensus)\n", f"Shown for **{100 * g['coverage']:.0f}%** of windows; MAE **{_f(g['mae_pooled'])}** BPM (subject mean {_f(g['mae_subject_mean'])}); within 1/3/5 BPM: "
          f"{100 * g['within_1']:.0f}% / {100 * g['within_3']:.0f}% / {100 * g['within_5']:.0f}%. Subjects with no shown window at all: {o['subjects_with_no_shown_window'] or 'none'}.\n",
          "## 3. Selection vs averaging: every method on exactly the windows the gate lets through\n", "| method | MAE | within 5 BPM |", "|---|---|---|"]
    for m, v in o["individual_methods_on_gate_passing_windows"].items():
        L.append(f"| {m} | {_f(v['mae_pooled'])} | {_f(None if v['within_5'] is None else 100 * v['within_5'], 0)}% |")
    L += ["\n## 4. Strata (gated number vs RGB POS alone)\n", "| stratum | windows | gated coverage | gated MAE | POS MAE (no gate) |", "|---|---|---|---|---|"]
    for k, grp in (("reference HR", o["by_reference_hr"]), ("condition", o["by_condition"])):
        for name, v in grp.items():
            L.append(f"| {k} {name} | {v['gated']['n_windows']} | {100 * v['gated']['coverage']:.0f}% | {_f(v['gated']['mae_pooled'])} | {_f(v['rgb_pos_ungated']['mae_pooled'])} |")
    t = o["top_quartile_signal_quality_windows"]
    L += [f"\nBest-signal quarter of windows ({t['n_windows']}): POS alone MAE {_f(t['rgb_pos_ungated']['mae_pooled'])}, app gated MAE {_f(t['app_gated']['mae_pooled'])} at {100 * t['app_gated']['coverage']:.0f}% coverage.\n"]
    q = o["quoted_frozen_test_numbers"]
    if q:
        L += ["## 5. Frozen test split (quoted from existing files, not re-analysed)\n", f"`{q.get('file')}` per-subject MAE: {q.get('rows_mae_subject_mean')}\n",
              f"`{q.get('quality_rule_test_file')}`: app gated {q.get('app_gated_on_frozen_test')}; voters without gate {q.get('ungated_voters_on_frozen_test')}\n"]
    L += ["## 6. Cannot be separated with these data\n"] + [f"- {x}" for x in o["cannot_be_separated_with_these_data"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
