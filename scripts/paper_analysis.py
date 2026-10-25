"""Independent audit and extra analyses for the paper, computed only from saved per-window outputs (no model is run, nothing is tuned).

    python scripts/paper_analysis.py

1. **Audit**: recomputes the subject-level MAE, bootstrap CIs, Wilcoxon p-values and Holm adjustment of the frozen test results (and the validation
   results) from ``windows_data`` and compares them with what ``src.evaluate`` reported (``rows`` / ``tests``).
2. **Effect sizes**: paired per-subject differences between pre-specified contrasts with bootstrap CIs over subjects (descriptive; the 12 test
   subjects cannot support equivalence claims, and no new selection is made from them).
3. **Accuracy-availability (risk-coverage)** on the *validation* split: how MAE of the shown number changes with how often a number is shown,
   for the app's rule family and for POS alone; the frozen rule's test result is marked from its existing results file.
4. **Figures**: per-subject dot plot, forest plot, risk-coverage curve, evaluation-protocol diagram, network/pipeline diagram.
5. Registers every number used by the paper in ``paper/claims.json`` (artifact = ``out/paper_analysis_<time>.json``).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_discrepancy import VOTERS, gate, load_windows  # noqa: E402
from src.common import register_claim, timestamp  # noqa: E402
from src.stats import bootstrap_ci, paired_wilcoxon  # noqa: E402

OUT = ROOT / "out"
PRE = "python scripts/paper_analysis.py"
LABEL = {"rgb_pos": "RGB POS", "rgb_chrom": "RGB CHROM", "rgb_green": "RGB green", "event_pos@0.02": "event POS (θ=0.02)", "event_pos@0.05": "event POS (θ=0.05)",
         "ann_th0.05": "ANN (θ=0.05)", "snn_th0.1": "SNN (θ=0.1)", "snn_th0.05": "SNN (θ=0.05)", "ann_th0.1": "ANN (θ=0.1)", "ann_th0.02": "ANN (θ=0.02)",
         "snn_th0.02": "SNN (θ=0.02)", "event_green@0.05": "event green (θ=0.05)"}


def latest(pattern: str) -> Path:
    """Newest file in out/ matching ``pattern``."""
    f = sorted(glob.glob(str(OUT / pattern)))
    if not f:
        raise SystemExit(f"no artifact out/{pattern}")
    return Path(f[-1])


def members(res: dict, fam: str) -> list[str]:
    """Method names of a family: ``snn_th0.05`` -> ``snn_th0.05_s0, _s1, _s2``; single-run methods are their own family."""
    names = list(res["per_clip"][0]["windows_data"])
    seeds = [n for n in names if re.fullmatch(re.escape(fam) + r"_s\d+", n)]
    return seeds or [fam]


def subject_mae(res: dict, method: str) -> dict[str, float]:
    """Subject -> mean over the subject's clips of the clip MAE on valid windows (as ``src.evaluate.subject_table``)."""
    acc: dict[str, list[float]] = {}
    for c in res["per_clip"]:
        wd = c["windows_data"].get(method)
        if wd is None:
            continue
        v = np.array(wd["valid"], bool)
        if not v.any():
            continue
        e = np.abs(np.array(wd["bpm"], float) - np.array(wd["ref"], float))[v]
        acc.setdefault(c["spec"].split(":")[1].split("_")[0], []).append(float(e.mean()))
    return {s: float(np.mean(x)) for s, x in acc.items()}


def family_subject_mae(res: dict, fam: str) -> dict[str, float]:
    """Seed-averaged subject MAE of a family."""
    tabs = [subject_mae(res, m) for m in members(res, fam)]
    subj = sorted(set.intersection(*[set(t) for t in tabs]))
    return {s: float(np.mean([t[s] for t in tabs])) for s in subj}


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (same as ``src.evaluate.holm``)."""
    out, prev = {}, 0.0
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        prev = max(prev, min(1.0, (len(p) - i) * v))
        out[k] = prev
    return out


def audit(res: dict, tag: str, seed: int, n_boot: int) -> dict:
    """Recompute rows and tests from windows_data and compare with the reported ones."""
    fams = {r["method"] for r in res["rows"]}
    mine = {f: family_subject_mae(res, f) for f in fams}
    dm, dlo, dhi = [], [], []
    rows = {}
    for r in res["rows"]:
        v = np.array(list(mine[r["method"]].values()))
        m, lo, hi = bootstrap_ci(v, n_boot, 0.95, seed)
        rows[r["method"]] = {"mae": m, "ci_lo": lo, "ci_hi": hi, "n": len(v)}
        dm.append(abs(m - r["mae"])); dlo.append(abs(lo - r["mae_ci_lo"])); dhi.append(abs(hi - r["mae_ci_hi"]))
    raw, dp = {}, []
    for k, t in res["tests"].items():
        a, b = k.split(" vs ")
        common = sorted(set(mine[a]) & set(mine[b]))
        w = paired_wilcoxon(np.array([mine[a][s] for s in common]), np.array([mine[b][s] for s in common]))
        raw[k] = w["p_value"]
        dp.append(abs(w["p_value"] - t["p_value"]))
    hp = holm(raw)
    dh = [abs(hp[k] - res["tests"][k]["p_holm"]) for k in hp]
    de = [abs(((r["adds_per_10s"] + r["sops_per_10s"]) * 0.9 + r["macs_per_10s"] * 4.6) * 1e-6 - r["energy_uj_per_10s"]) for r in res["rows"]]      # 0.9 pJ / accumulate, 4.6 pJ / MAC
    return {"split": tag, "n_methods": len(rows), "n_tests": len(raw), "max_abs_mae_diff": max(dm), "max_abs_ci_lo_diff": max(dlo), "max_abs_ci_hi_diff": max(dhi),
            "max_abs_p_diff": max(dp), "max_abs_p_holm_diff": max(dh), "max_abs_energy_proxy_diff_uj": max(de), "n_subjects": sorted({len(v) for v in mine.values()})}


def paired(res: dict, a: str, b: str, seed: int, n_boot: int) -> dict:
    """Paired subject-level difference A-B (BPM; negative = A better) with a percentile bootstrap CI over subjects."""
    ta, tb = family_subject_mae(res, a), family_subject_mae(res, b)
    common = sorted(set(ta) & set(tb))
    d = np.array([ta[s] - tb[s] for s in common])
    mean, lo, hi = bootstrap_ci(d, n_boot, 0.95, seed)
    return {"a": a, "b": b, "n": len(d), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "a_better_subjects": int((d < 0).sum()), "median_diff": float(np.median(d))}


CONTRASTS = [("rgb_pos", "rgb_chrom"), ("rgb_pos", "rgb_green"), ("event_pos@0.02", "rgb_pos"), ("ann_th0.05", "rgb_pos"), ("snn_th0.1", "rgb_pos"),
             ("ann_th0.05", "event_pos@0.05"), ("snn_th0.02", "ann_th0.02"), ("snn_th0.05", "ann_th0.05"), ("snn_th0.1", "ann_th0.1")]



def gate_audit(res: dict, rule: dict) -> dict:
    """Apply the frozen rule to one split's windows with our own implementation: coverage, MAE, per-subject view, and the selection effect for single methods."""
    wins = load_windows(res)
    shown = [gate(w, rule) for w in wins]
    idx = [i for i, s in enumerate(shown) if s is not None]
    ref = np.array([w["ref"] for w in wins])
    err = np.abs(np.array([shown[i] for i in idx]) - ref[idx])
    by_subj: dict[str, list[int]] = {}
    for i, w in enumerate(wins):
        by_subj.setdefault(w["subject"], []).append(i)
    cov = {s: float(np.mean([shown[i] is not None for i in ii])) for s, ii in by_subj.items()}
    sub_mae = {s: float(np.mean([abs(shown[i] - ref[i]) for i in ii if shown[i] is not None])) for s, ii in by_subj.items() if any(shown[i] is not None for i in ii)}
    on_shown = {}
    for m in ("rgb_pos", "rgb_green", "rgb_chrom", "event_pos", "ann", "snn"):
        e = [abs(wins[i]["est"][m][0] - ref[i]) for i in idx if m in wins[i]["est"]]
        on_shown[m] = float(np.mean(e)) if e else None
    all_pos = [abs(w["est"]["rgb_pos"][0] - w["ref"]) for w in wins]
    return {"n_windows": len(wins), "n_shown": len(idx), "coverage": len(idx) / len(wins), "mae": float(err.mean()), "within_1": float(np.mean(err <= 1)),
            "within_3": float(np.mean(err <= 3)), "within_5": float(np.mean(err <= 5)), "n_subjects": len(by_subj), "n_subjects_with_shown": len(sub_mae),
            "subject_coverage_min": min(cov.values()), "subject_coverage_median": float(np.median(list(cov.values()))), "subject_coverage_max": max(cov.values()),
            "subject_mean_of_gated_mae": float(np.mean(list(sub_mae.values()))), "subject_max_gated_mae": max(sub_mae.values()),
            "pos_all_windows_mae": float(np.mean(all_pos)), "on_shown_mae": on_shown}

# ---------------------------------------------------------------------------------------------- risk-coverage on validation
def risk_coverage(wins: list[dict]) -> dict:
    """MAE of the shown value versus coverage for the app's rule family (min_agree=2) and for POS alone gated on its own peak ratio."""
    ref = np.array([w["ref"] for w in wins])
    out = {"consensus": {}, "pos_only": []}
    for agree in (4.0, 8.0, 12.0):
        pts = []
        for ratio in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30):
            rule = {"min_peak_ratio": ratio, "agree_bpm": agree, "min_agree": 2, "voters": VOTERS}
            shown = [gate(w, rule) for w in wins]
            idx = [i for i, s in enumerate(shown) if s is not None]
            if len(idx) >= 20:
                err = np.abs(np.array([shown[i] for i in idx]) - ref[idx])
                pts.append({"min_peak_ratio": ratio, "coverage": len(idx) / len(wins), "mae": float(err.mean()), "within5": float(np.mean(err <= 5)), "n_shown": len(idx)})
        out["consensus"][f"agree{agree:.0f}"] = pts
    for ratio in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30):
        idx = [i for i, w in enumerate(wins) if w["est"]["rgb_pos"][1] >= ratio]
        if len(idx) >= 20:
            err = np.abs(np.array([wins[i]["est"]["rgb_pos"][0] for i in idx]) - ref[idx])
            out["pos_only"].append({"min_peak_ratio": ratio, "coverage": len(idx) / len(wins), "mae": float(err.mean()), "within5": float(np.mean(err <= 5)), "n_shown": len(idx)})
    return out


# ---------------------------------------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 220, "font.family": "DejaVu Sans"})
    return plt


def fig_subject_dots(res: dict, path: Path, plt) -> list[str]:
    """Per-subject test MAE for the main methods; subjects ordered by RGB POS error."""
    methods = ["rgb_pos", "rgb_chrom", "event_pos@0.02", "ann_th0.05", "snn_th0.1"]
    tabs = {m: family_subject_mae(res, m) for m in methods}
    subj = sorted(tabs["rgb_pos"], key=lambda s: tabs["rgb_pos"][s])
    cols = {"rgb_pos": "#1b6ca8", "rgb_chrom": "#7aa6c9", "event_pos@0.02": "#e08a1e", "ann_th0.05": "#7b4fa0", "snn_th0.1": "#c0392b"}
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    for m in methods:
        ax.plot(range(len(subj)), [tabs[m][s] for s in subj], "o-" if m == "rgb_pos" else "o", ms=3.8, lw=1.0, color=cols[m], label=LABEL[m], alpha=0.95)
    ax.set_xticks(range(len(subj)))
    ax.set_xticklabels([f"S{i + 1}" for i in range(len(subj))])
    ax.set_xlabel("held-out test subject (ordered by RGB POS error)")
    ax.set_ylabel("subject MAE (BPM)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["0", "2", "5", "10", "20", "40"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=6.2, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return subj


def fig_forest(rows: list[dict], path: Path, plt) -> None:
    """Paired mean differences (A - B) with 95 % bootstrap CIs over the 12 test subjects."""
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.errorbar(r["mean_diff"], yi, xerr=[[r["mean_diff"] - r["ci_lo"]], [r["ci_hi"] - r["mean_diff"]]], fmt="o", color="#1b6ca8", capsize=2.5, ms=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[r['a']]} − {LABEL[r['b']]}" for r in rows])
    ax.set_xlabel("first − second method (BPM)\n< 0: first method better", fontsize=6.8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_risk(rc: dict, test_pt: dict, ungated_pos: float, path: Path, plt) -> None:
    """MAE of the shown number against how often a number is shown (validation windows) and the frozen rule's test result."""
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    cols = {"agree4": "#9ecae1", "agree8": "#1b6ca8", "agree12": "#08306b"}
    for k, pts in rc["consensus"].items():
        ax.plot([100 * p["coverage"] for p in pts], [p["mae"] for p in pts], "o-", ms=2.8, lw=1.0, color=cols[k], label=f"RGB consensus, agree ≤ {k[5:]} BPM")
    ax.plot([100 * p["coverage"] for p in rc["pos_only"]], [p["mae"] for p in rc["pos_only"]], "s--", ms=2.8, lw=1.0, color="#e08a1e", label="RGB POS alone, peak-ratio gate")
    chosen = next(p for p in rc["consensus"]["agree8"] if p["min_peak_ratio"] == 8)
    ax.plot(100 * chosen["coverage"], chosen["mae"], "*", ms=10, color="#c0392b", label="frozen rule (validation)")
    ax.plot(100 * test_pt["coverage"], test_pt["mae"], "D", ms=5, mfc="none", mec="#c0392b", label="frozen rule on test split")
    ax.axhline(ungated_pos, color="gray", lw=0.8, ls=":")
    ax.text(97, ungated_pos + 0.15, "POS, every window", fontsize=6.3, color="gray", ha="right")
    ax.set_xlabel("windows with a displayed number (%)")
    ax.set_ylabel("MAE of displayed numbers (BPM)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10.2)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=5.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#e8eef5", ec="#44546a", fs=7, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", fc=fc, ec=ec, lw=0.9, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#44546a"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))


def fig_protocol(path: Path, plt, n: dict) -> None:
    """Evaluation protocol: subject-disjoint splits and what each split is allowed to decide."""
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.005, 0.36, 0.15, 0.30, f"MCD-rPPG\n{n['total']} subjects\n(one rest and one\npost-exercise\nclip per subject)", "#f3f3f3", fs=6.0)
    ys = [(0.74, "#dce9f5", f"train: {n['train']} subjects\nfit the networks (3 seeds)"), (0.40, "#fdebd0", f"validation: {n['val']} subjects\nchoose epoch, threshold θ, SHS picker,\nreliability-gate rule"),
          (0.06, "#f5d5d3", f"test: {n['test']} subjects (protected)\nscored after the rules were frozen;\nnever used for a decision")]
    for y, c, t in ys:
        box(ax, 0.215, y, 0.285, 0.20, t, c, fs=6.0)
        arrow(ax, 0.157, 0.51, 0.218, y + 0.10)
        arrow(ax, 0.492, y + 0.10, 0.545, 0.51)
    box(ax, 0.55, 0.22, 0.20, 0.58, "estimators\nsame 10 s windows,\n1 s stride\n\nRGB green/POS/CHROM\nevent green/POS\nANN and SNN (3 seeds)", "#e8f1e4", fs=6.0)
    box(ax, 0.81, 0.54, 0.185, 0.34, "accuracy per subject\n(each subject once;\nbootstrap CI;\nWilcoxon + Holm)", "#e8eef5", fs=6.0)
    box(ax, 0.81, 0.10, 0.185, 0.34, "availability\n(share of windows\nwith a number shown,\nMAE of those)", "#e8eef5", fs=6.0)
    arrow(ax, 0.752, 0.60, 0.808, 0.70)
    arrow(ax, 0.752, 0.42, 0.808, 0.28)
    ax.text(0.5, -0.04, "reference: synchronised finger PPG, spectral-peak heart rate; windows with peak ratio < 4 are excluded for every method; subjects are never shared between splits", ha="center", fontsize=6.3, color="#555")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_architecture(path: Path, plt, params: int) -> None:
    """The implemented event path and the matched SNN/ANN (layer sizes from ``src/snn_model.py``)."""
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.985, "(a) Event emulation from video (analytic DVS model, each colour channel separately)", fontsize=7.4, fontweight="bold", va="top")
    top = ["webcam video\n≈30 fps, MPEG-4", "face crop 64×64\nskin-ROI mask", "L = ln(I+1);\nevent when |ΔL| ≥ θ;\nmismatch, refractory,\nleak, shot noise", "ON/OFF events binned\n33.3 ms → 3×32×32\nsigned frames", "classical estimators:\nevent green, event POS\n(ROI event counts)"]
    w, g = 0.178, 0.027
    for i, t in enumerate(top):
        x = i * (w + g)
        box(ax, x, 0.60, w, 0.29, t, "#e8eef5", fs=6.3)
        if i < len(top) - 1:
            arrow(ax, x + w + 0.002, 0.745, x + w + g - 0.002, 0.745)
    ax.text(0.0, 0.50, f"(b) Matched pulse regressors, identical layers ({params:,} parameters)".replace(",", "\u2009"), fontsize=7.4, fontweight="bold", va="top")
    layers = [("input 3×32×32\ntanh(x/4) clip", "#f3f3f3"), ("Conv 5×5, stride 2\n8 channels\n→ N", "#dce9f5"), ("Conv 3×3, stride 2\n16 channels\n→ N", "#dce9f5"), ("AvgPool → 4×4\nFC 256→64\n→ N", "#dce9f5"),
              ("FC 64→1\nleaky integrator\nβ = 0.8", "#dce9f5"), ("pulse waveform\n→ 10 s window\n→ SHS peak\n→ BPM", "#e8f1e4")]
    w2, g2 = 0.15, 0.0200
    for i, (t, c) in enumerate(layers):
        x = i * (w2 + g2 + 0.0)
        box(ax, x, 0.17, w2, 0.26, t, c, fs=6.2)
        if i < len(layers) - 1:
            arrow(ax, x + w2 + 0.002, 0.30, x + w2 + g2 - 0.002, 0.30)
    ax.text(0.0, 0.07, "N = LIF neuron (β = 0.9, fast-sigmoid surrogate gradient, snnTorch) in the SNN and ReLU in the ANN; the read-out is the same non-spiking leaky integrator in both.\n"
                       "Operation counts: ANN = dense MACs; SNN = measured spikes × fan-out (an upper bound); classical estimators = one accumulate per ROI event.", fontsize=6.0, color="#444", va="top")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def make_tables(val: dict, test: dict, res: dict, ts: str) -> None:
    """LaTeX tables: validation-versus-test MAE with the validation-selected threshold marked, and the gating/selection table."""
    tm = {r["method"]: r for r in test["rows"]}
    order = [("RGB green", "rgb_green"), ("RGB CHROM", "rgb_chrom"), ("RGB POS", "rgb_pos"), ("event green", "event_green@{t}"), ("event POS", "event_pos@{t}"), ("ANN", "ann_th{t}"), ("SNN", "snn_th{t}")]
    vm = {r["method"]: r for r in val["rows"]}
    lines = [r"\begin{tabular}{lrrrrrr}\toprule", r" & \multicolumn{3}{c}{validation MAE (BPM)} & \multicolumn{3}{c}{test MAE (BPM)} \\", r"method & $\theta$=0.02 & 0.05 & 0.1 & 0.02 & 0.05 & 0.1 \\\midrule"]
    for label, pat in order:
        if "{t}" not in pat:
            lines.append(f"{label} & \\multicolumn{{3}}{{c}}{{{vm[pat]['mae']:.1f}}} & \\multicolumn{{3}}{{c}}{{{tm[pat]['mae']:.1f}}} \\\\")
            continue
        names = [pat.format(t=t) for t in ("0.02", "0.05", "0.1")]
        best = min(range(3), key=lambda i: vm[names[i]]["mae"])
        cells_v = [f"\\textbf{{{vm[n]['mae']:.1f}}}" if i == best else f"{vm[n]['mae']:.1f}" for i, n in enumerate(names)]
        cells_t = [f"\\textbf{{{tm[n]['mae']:.1f}}}" if i == best else f"{tm[n]['mae']:.1f}" for i, n in enumerate(names)]
        lines.append(f"{label} & " + " & ".join(cells_v) + " & " + " & ".join(cells_t) + r" \\")
    lines += [r"\bottomrule\end{tabular}"]
    (OUT / f"table_valsel_{ts}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ga = {"val": res["gate_audit_val"], "test": res["gate_audit_test"]}
    g = [r"\begin{tabular}{lrr}\toprule", r"& validation & test \\\midrule"]
    row = lambda name, f: g.append(name + " & " + " & ".join(f(ga[s]) for s in ("val", "test")) + r" \\")  # noqa: E731
    row("subjects / valid windows", lambda a: f"{a['n_subjects']} / {a['n_windows']}")
    row("windows with a displayed number", lambda a: f"{a['n_shown']} ({100 * a['coverage']:.0f}\\%)")
    row("subject coverage: min / median / max", lambda a: f"{100 * a['subject_coverage_min']:.0f} / {100 * a['subject_coverage_median']:.0f} / {100 * a['subject_coverage_max']:.0f}\\%")
    row("MAE of displayed numbers (BPM)", lambda a: f"{a['mae']:.2f}")
    row("within 1 / 3 / 5 BPM", lambda a: f"{100 * a['within_1']:.0f} / {100 * a['within_3']:.0f} / {100 * a['within_5']:.0f}\\%")
    g.append(r"\midrule \multicolumn{3}{l}{\emph{MAE (BPM) of single methods on exactly the windows the gate displays}} \\")
    for m, lab in (("rgb_pos", "RGB POS"), ("rgb_chrom", "RGB CHROM"), ("rgb_green", "RGB green"), ("ann", "ANN ($\\theta$=0.05)"), ("event_pos", "event POS ($\\theta$=0.05)"), ("snn", "SNN ($\\theta$=0.05)")):
        row(lab, lambda a, m=m: f"{a['on_shown_mae'][m]:.2f}")
    g.append(r"\midrule \multicolumn{3}{l}{\emph{for reference: MAE (BPM) over all valid windows}} \\")
    row("RGB POS, every window (pooled windows)", lambda a: f"{a['pos_all_windows_mae']:.2f}")
    g.append(r"\bottomrule\end{tabular}")
    (OUT / f"table_gate_{ts}.tex").write_text("\n".join(g) + "\n", encoding="utf-8")
    import shutil
    for n in ("valsel", "gate"):
        (ROOT / "paper" / "tables").mkdir(exist_ok=True)
        shutil.copy(OUT / f"table_{n}_{ts}.tex", ROOT / "paper" / "tables" / f"table_{n}.tex")


def register_env(artifact: Path) -> None:
    """Software versions, seed and one clip's effective frame rate (traceable settings used in the paper's text)."""
    import platform
    import snntorch
    import torch
    import mediapipe
    from src.common import load_config, repo_path
    from src.datasets import mcd_effective_fs
    cfg = load_config()
    reg = lambda k, v: register_claim(k, v, artifact, PRE)  # noqa: E731
    root = repo_path(cfg["datasets"]["mcd_root"])
    stem = "8772_FullHDwebcam_before"
    n = len(np.loadtxt(root / "ppg_sync" / f"{stem}.txt")[:, 0])
    fs = float(mcd_effective_fs(root / "meta" / f"{stem}.txt", n))
    env = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0], "snntorch": snntorch.__version__, "mediapipe": mediapipe.__version__,
           "seed": int(cfg["seed"]), "clip8772_before_fs": fs}
    res = json.loads(artifact.read_text())
    res["env"] = env
    artifact.write_text(json.dumps(res, indent=1), encoding="utf-8")
    reg("env.python_version", env["python"])
    reg("env.torch", env["torch"])
    reg("env.snntorch", env["snntorch"])
    reg("env.mediapipe", env["mediapipe"])
    reg("env.seed", env["seed"])
    reg("data.val.clip8772_before.fs", fs)


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    val_p, test_p = latest("results_mcd-val_val_all_*.json"), latest("results_mcd-test_final_*.json")
    val, test = json.loads(val_p.read_text()), json.loads(test_p.read_text())
    assert val["split"] == "val" and test["split"] == "test"
    res: dict = {"command": PRE, "validation_file": val_p.name, "test_file": test_p.name, "seed": a.seed, "n_boot": a.n_boot}

    res["audit"] = {"val": audit(val, "val", a.seed, a.n_boot), "test": audit(test, "test", a.seed, a.n_boot)}
    for k, v in res["audit"].items():
        print(f"audit {k}: " + ", ".join(f"{x}={y:.2e}" if isinstance(y, float) else f"{x}={y}" for x, y in v.items()))
    res["paired_test"] = [paired(test, x, y, a.seed, a.n_boot) for x, y in CONTRASTS]
    for r in res["paired_test"]:
        print(f"  {r['a']:>16} - {r['b']:<16} {r['mean_diff']:+6.2f} [{r['ci_lo']:+6.2f}, {r['ci_hi']:+6.2f}]  first better in {r['a_better_subjects']}/{r['n']}")

    wins = load_windows(val)
    res["risk_coverage_val"] = risk_coverage(wins)
    qr = json.loads(latest("quality_rule_test_*.json").read_text(encoding="utf-8-sig"))
    res["frozen_rule_test"] = qr["applied"] | {"ungated_mae": qr["ungated_mae"], "file": latest("quality_rule_test_*.json").name}
    pos_val = next(r for r in val["rows"] if r["method"] == "rgb_pos")["mae"]
    plt = setup_mpl()
    fd = ROOT / "paper" / "figures"
    ts = timestamp()
    subj = fig_subject_dots(test, OUT / f"fig_subject_dots_{ts}.png", plt)
    fig_forest(res["paired_test"], OUT / f"fig_forest_{ts}.png", plt)
    pooled_pos_val = float(np.mean([abs(w["est"]["rgb_pos"][0] - w["ref"]) for w in wins]))
    fig_risk(res["risk_coverage_val"], {"coverage": qr["applied"]["coverage"], "mae": qr["applied"]["mae"]}, pooled_pos_val, OUT / f"fig_risk_coverage_{ts}.png", plt)
    n = {"train": 36, "val": 12, "test": 12}
    n["total"] = sum(n.values())
    fig_protocol(OUT / f"fig_protocol_{ts}.png", plt, n)
    fig_architecture(OUT / f"fig_architecture_{ts}.png", plt, 18289)
    rule = json.loads((ROOT / "configs" / "quality_calibrated.json").read_text(encoding="utf-8-sig"))
    res["gate_audit_test"] = gate_audit(test, rule)
    res["gate_audit_val"] = gate_audit(val, rule)
    print("gate audit test:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["gate_audit_test"].items() if k != "on_shown_mae"}, res["gate_audit_test"]["on_shown_mae"])
    res["pooled_pos_val_mae_all_windows"] = pooled_pos_val
    res["n_val_windows"] = len(wins)
    res["subjects_ordered_by_pos"] = subj
    res["figures"] = [f"fig_{k}_{ts}.png" for k in ("subject_dots", "forest", "risk_coverage", "protocol", "architecture")]
    jp = OUT / f"paper_analysis_{ts}.json"
    jp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("saved", jp)
    reg = lambda k, v, art=jp, cmd=PRE: register_claim(k, v, art, cmd)  # noqa: E731
    for split, d in res["audit"].items():
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                reg(f"audit.{split}.{k}", v)
    for r in res["paired_test"]:
        key = f"paired.test.{r['a']}-{r['b']}"
        for f in ("mean_diff", "ci_lo", "ci_hi", "a_better_subjects", "n"):
            reg(f"{key}.{f}", r[f])
    for k, v in res["frozen_rule_test"].items():
        if isinstance(v, float):
            reg(f"gate.test.{k}", v)
    reg("gate.test.n_shown", res["frozen_rule_test"]["n_shown"])
    reg("gate.test.n", res["frozen_rule_test"]["n"])
    for k in ("rgb_green", "rgb_pos", "rgb_chrom"):
        reg(f"gate.test.ungated_{k}", res["frozen_rule_test"]["ungated_mae"][k])
    reg("risk.val.pos_all_windows_mae", pooled_pos_val)
    reg("risk.val.n_windows", len(wins))
    ch = next(p for p in res["risk_coverage_val"]["consensus"]["agree8"] if p["min_peak_ratio"] == 8)
    for k, v in ch.items():
        reg(f"risk.val.chosen.{k}", v)
    for sp in ("test", "val"):
        for k, v in res[f"gate_audit_{sp}"].items():
            if k == "on_shown_mae":
                for m, x in v.items():
                    reg(f"gatea.{sp}.on_shown.{m}", x)
            else:
                reg(f"gatea.{sp}.{k}", v)
    disc = latest("discrepancy_*.json")
    d = json.loads(disc.read_text())
    dreg = lambda k, v: register_claim(k, v, disc, "python scripts/analyze_discrepancy.py")  # noqa: E731
    for m, v in d["ungated_individual"].items():
        dreg(f"select.val.ungated.{m}.mae", v["mae_pooled"])
        dreg(f"select.val.ungated.{m}.within5", v["within_5"])
    dreg("select.val.rgb_avg_all.mae", d["ungated_rgb_average_every_window"]["mae_pooled"])
    for k in ("coverage", "mae_pooled", "mae_subject_mean", "within_1", "within_3", "within_5", "n_shown", "n_windows"):
        dreg(f"select.val.gated.{k}", d["app_gated"][k])
    for m, v in d["individual_methods_on_gate_passing_windows"].items():
        dreg(f"select.val.on_gated.{m}.mae", v["mae_pooled"])
    for name, v in d["by_reference_hr"].items():
        dreg(f"select.val.hr[{name}].gated_mae", v["gated"]["mae_pooled"])
        dreg(f"select.val.hr[{name}].coverage", v["gated"]["coverage"])
        dreg(f"select.val.hr[{name}].pos_mae", v["rgb_pos_ungated"]["mae_pooled"])
    for c, v in d["by_condition"].items():
        dreg(f"select.val.cond.{c}.gated_mae", v["gated"]["mae_pooled"])
        dreg(f"select.val.cond.{c}.coverage", v["gated"]["coverage"])
        dreg(f"select.val.cond.{c}.pos_mae", v["rgb_pos_ungated"]["mae_pooled"])
    t = d["top_quartile_signal_quality_windows"]
    dreg("select.val.topq.coverage", t["app_gated"]["coverage"])
    dreg("select.val.topq.gated_mae", t["app_gated"]["mae_pooled"])
    dreg("select.val.topq.pos_mae", t["rgb_pos_ungated"]["mae_pooled"])
    for p in res["risk_coverage_val"]["pos_only"]:
        for k in ("coverage", "mae", "within5", "n_shown"):
            reg(f"risk.val.pos_only.r{p['min_peak_ratio']}.{k}", p[k])
    for kk, pts in res["risk_coverage_val"]["consensus"].items():
        for p in pts:
            for k in ("coverage", "mae"):
                reg(f"risk.val.{kk}.r{p['min_peak_ratio']}.{k}", p[k])
    for r in val["rows"]:
        reg(f"val.rows.{r['method']}.mae", r["mae"], val_p, "python -m src.evaluate --dataset mcd --split val ...")
    make_tables(val, test, res, ts)
    reg("audit.n_boot", a.n_boot)
    from src.common import load_config as _lc, repo_path as _rp
    if (_rp(_lc()["datasets"]["mcd_root"]) / "ppg_sync").is_dir():          # environment/frame-rate claims are recorded once, from the original run and data
        register_env(jp)
    else:
        print("MCD-rPPG files not found: environment and frame-rate claims left as originally registered")
    print("claims registered")
    import shutil
    for k in ("subject_dots", "forest", "risk_coverage", "protocol", "architecture"):
        shutil.copy(OUT / f"fig_{k}_{ts}.png", fd / f"fig_{k}.png")
    print("figures copied to paper/figures")


if __name__ == "__main__":
    main()
