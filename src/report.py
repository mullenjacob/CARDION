"""Tables and figures for the paper, built only from saved ``results_*.json`` files (no recomputation).

    python -m src.evaluate --report "out/results_mcd-test_final_*.json"
"""
from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import numpy as np

from .common import out_path, register_claim

log = logging.getLogger("cardion.report")


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("@", r"@")


def _load(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"nothing matches {pattern}")
    res = json.loads(Path(files[-1]).read_text())
    res["_file"] = files[-1]
    return res


def _row(res: dict, method: str) -> dict | None:
    return next((r for r in res["rows"] if r["method"] == method), None)


def main_table(res: dict, methods: list[str]) -> list[str]:
    """LaTeX tabular: MAE (CI), RMSE, Pearson r, Lin's CCC, energy proxy."""
    lines = ["\\begin{tabular}{lrrrrrr}\\toprule", "Method & MAE & 95\\% CI & RMSE & $r$ & CCC & energy ($\\mu$J) \\\\\\midrule"]
    for m in methods:
        r = _row(res, m)
        if r is None:
            continue
        sd = f" $\\pm${r['mae_seed_sd']:.2f}" if r.get("mae_seed_sd") is not None else ""
        lines.append(f"{_tex(m)} & {r['mae']:.2f}{sd} & [{r['mae_ci_lo']:.2f}, {r['mae_ci_hi']:.2f}] & {r['rmse']:.2f} & {r['pearson_r']:.2f} & "
                     f"{r['ccc']:.2f} & {r['energy_uj_per_10s']:.3g} \\\\")
    return lines + ["\\bottomrule\\end{tabular}"]


def efficiency_table(res: dict, methods: list[str]) -> list[str]:
    """Operation counts, energy proxy (post-sensor), software-emulation ops and measured CPU time per 10 s window."""
    lines = ["\\begin{tabular}{lrrrrrr}\\toprule", "Method & adds & MACs & SOPs & energy ($\\mu$J) & emul. ops & CPU (ms) \\\\\\midrule"]
    for m in methods:
        r = _row(res, m)
        if r is None:
            continue
        lines.append(f"{_tex(m)} & {r['adds_per_10s']:.3g} & {r['macs_per_10s']:.3g} & {r['sops_per_10s']:.3g} & {r['energy_uj_per_10s']:.3g} & "
                     f"{r['emulation_ops_per_10s']:.3g} & {r['wall_ms_per_10s']:.3g} \\\\")
    return lines + ["\\bottomrule\\end{tabular}"]


def strata_table(res: dict, methods: list[str], key: str) -> list[str]:
    """MAE per stratum (state / brightness tier / activity tier / HR tier)."""
    groups = sorted({g for m in methods for g in res["strata"][key].get(m, {})})
    lines = ["\\begin{tabular}{l" + "r" * len(groups) + "}\\toprule", "Method & " + " & ".join(f"{g} (n={res['strata'][key][methods[0]].get(g, {}).get('n', '-')})" for g in groups) + " \\\\\\midrule"]
    for m in methods:
        d = res["strata"][key].get(m, {})
        lines.append(_tex(m) + " & " + " & ".join(f"{d[g]['mae']:.2f}" if g in d else "--" for g in groups) + " \\\\")
    return lines + ["\\bottomrule\\end{tabular}"]


def tests_table(res: dict) -> list[str]:
    """Paired Wilcoxon tests over subjects with Holm correction."""
    lines = ["\\begin{tabular}{lrrr}\\toprule", "Comparison (A vs B) & mean diff A$-$B & $p$ & $p_{\\mathrm{Holm}}$ \\\\\\midrule"]
    for k, t in res["tests"].items():
        lines.append(f"{_tex(k)} & {t['mean_diff']:+.2f} & {t['p_value']:.3g} & {t.get('p_holm', float('nan')):.3g} \\\\")
    return lines + ["\\bottomrule\\end{tabular}"]


def build_report(pattern: str, cfg: dict) -> None:
    """Write LaTeX tables and figures for the result file matching ``pattern``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = _load(pattern)
    tag = f"{res['dataset']}-{res['split']}_{res['tag']}"
    methods = [r["method"] for r in res["rows"]]
    order = ["rgb_green", "rgb_pos", "rgb_chrom"] + sorted(m for m in methods if m.startswith("event")) + sorted(m for m in methods if m.startswith(("snn", "ann")))
    order = [m for m in order if m in methods]
    paths = {}
    for stem, lines in (("table_main", main_table(res, order)), ("table_efficiency", efficiency_table(res, order)), ("table_tests", tests_table(res))):
        paths[stem] = out_path(cfg, f"{stem}_{tag}", "tex", stamp=False)
        paths[stem].write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key in ("state", "brightness", "activity", "heart_rate"):
        sel = [m for m in order if m in res["strata"][key]]
        if sel:
            paths[f"table_strata_{key}"] = out_path(cfg, f"table_strata_{key}_{tag}", "tex", stamp=False)
            paths[f"table_strata_{key}"].write_text("\n".join(strata_table(res, sel, key)) + "\n", encoding="utf-8")

    # --- accuracy vs energy Pareto
    fig, ax = plt.subplots(figsize=(6.4, 4))
    pts = [(r["energy_uj_per_10s"], r["mae"], r["method"]) for r in res["rows"] if r["energy_uj_per_10s"] > 0]
    kinds = (("RGB (no events)", "s", lambda m: m.startswith("rgb")), ("event, classical", "o", lambda m: m.startswith("event")),
             ("SNN", "^", lambda m: m.startswith("snn")), ("ANN", "v", lambda m: m.startswith("ann")))
    for label, mk, pred in kinds:
        sel = [p for p in pts if pred(p[2])]
        if sel:
            ax.scatter([p[0] for p in sel], [p[1] for p in sel], marker=mk, label=label, s=34)
    front, best = [], np.inf
    for p in sorted(pts):
        if p[1] < best:
            best = p[1]; front.append(p)
    ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="k", lw=0.8, label="Pareto front")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("energy proxy per 10 s window ($\\mu$J, post-sensor compute only)"); ax.set_ylabel("HR MAE (BPM, subject mean)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p_par = out_path(cfg, f"fig_pareto_{tag}", "png", stamp=False)
    fig.savefig(p_par, dpi=150); plt.close(fig)

    # --- threshold ablation
    ths = sorted(res["thresholds"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for stem, mk in (("event_green", "o"), ("event_pos", "s"), ("snn_th", "^"), ("ann_th", "v")):
        ys = []
        for th in ths:
            name = f"{stem}@{th}" if stem.startswith("event") else f"{stem}{th}"
            r = _row(res, name)
            ys.append(r["mae"] if r else np.nan)
        if np.isfinite(ys).any():
            ax.plot(ths, ys, marker=mk, label=stem.replace("_th", ""))
    for m, ls in (("rgb_green", ":"), ("rgb_pos", "--"), ("rgb_chrom", "-.")):
        r = _row(res, m)
        if r:
            ax.axhline(r["mae"], color="k", ls=ls, lw=0.9, label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("emulated contrast threshold"); ax.set_ylabel("HR MAE (BPM)"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p_abl = out_path(cfg, f"fig_ablation_threshold_{tag}", "png", stamp=False)
    fig.savefig(p_abl, dpi=150); plt.close(fig)

    # --- seed variability
    fam = [r for r in res["rows"] if r["method"].startswith(("snn", "ann")) and r.get("n_seeds", 1) > 1]
    if fam:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for i, r in enumerate(sorted(fam, key=lambda r: r["method"])):
            ax.errorbar(i, r["mae"], yerr=[[r["mae"] - r["mae_seed_min"]], [r["mae_seed_max"] - r["mae"]]], fmt="o", capsize=4)
        ax.set_xticks(range(len(fam))); ax.set_xticklabels([r["method"] for r in sorted(fam, key=lambda r: r["method"])], rotation=30, fontsize=7)
        ax.set_ylabel("test MAE (BPM); bars = seed min/max")
        fig.tight_layout()
        p_seed = out_path(cfg, f"fig_seeds_{tag}", "png", stamp=False)
        fig.savefig(p_seed, dpi=150); plt.close(fig)

    # --- Bland-Altman (pooled valid windows)
    show = [m for m in ("rgb_pos", "rgb_green") + tuple(m for m in order if m.startswith(("event_green", "snn", "ann")))][:6]
    ncols = min(3, len(show))
    nrows = -(-len(show) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.35 * nrows + 0.2), squeeze=False)    # two-column page width; 3 panels per row stay legible
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, show):
        key = lambda c: m if m in c["windows_data"] else f"{m}_s0"          # model families: first seed  # noqa: E731
        pred = np.concatenate([np.array(c["windows_data"][key(c)]["bpm"])[np.array(c["windows_data"][key(c)]["valid"])] for c in res["per_clip"]])
        ref = np.concatenate([np.array(c["windows_data"][key(c)]["ref"])[np.array(c["windows_data"][key(c)]["valid"])] for c in res["per_clip"]])
        # models are reported per family; use the first seed's windows here
        d, mean = pred - ref, (pred + ref) / 2
        bias, sd = d.mean(), d.std(ddof=1)
        ax.scatter(mean, d, s=3, alpha=0.3)
        for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            ax.axhline(y, color="k", ls=ls, lw=0.7)
        ax.set_title(f"{m}\nbias {bias:.1f}, LoA [{bias - 1.96 * sd:.0f}, {bias + 1.96 * sd:.0f}]", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("mean HR (BPM)", fontsize=7.5)
    for row in axes:
        row[0].set_ylabel("estimate - reference", fontsize=7.5)
    fig.tight_layout()
    p_ba = out_path(cfg, f"fig_bland_altman_{tag}", "png", stamp=False)
    fig.savefig(p_ba, dpi=150); plt.close(fig)

    # --- strata bars (state)
    if "state" in res["strata"]:
        sel = [m for m in order if m in res["strata"]["state"]]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        states = sorted({g for m in sel for g in res["strata"]["state"][m]})
        w = 0.8 / max(len(states), 1)
        for j, g in enumerate(states):
            ax.bar(np.arange(len(sel)) + j * w, [res["strata"]["state"][m].get(g, {}).get("mae", np.nan) for m in sel], w, label=g)
        ax.set_xticks(np.arange(len(sel)) + w / 2); ax.set_xticklabels(sel, rotation=40, ha="right", fontsize=6)
        ax.set_yscale("log"); ax.set_ylabel("MAE (BPM)"); ax.legend(title="state (before = rest, after = exercise)", fontsize=6)
        fig.tight_layout()
        p_st = out_path(cfg, f"fig_strata_state_{tag}", "png", stamp=False)
        fig.savefig(p_st, dpi=150); plt.close(fig)

    # --- per-subject scatter: where do the learned models differ from the classical estimators?
    subj = {}
    for c in res["per_clip"]:
        sid = c["spec"].split(":")[1].split("_")[0]
        for m, v in c["mae"].items():
            if v is not None:
                subj.setdefault(m, {}).setdefault(sid, []).append(v)
    subj = {m: {s: float(np.mean(v)) for s, v in d.items()} for m, d in subj.items()}
    pairs = [(a, b) for a, b in (("rgb_pos", "event_pos@0.05"), ("event_pos@0.05", "ann_th0.05_s0"), ("ann_th0.05_s0", "snn_th0.05_s0")) if a in subj and b in subj]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 3.2), squeeze=False)
        for ax, (a, b) in zip(axes[0], pairs):
            common = sorted(set(subj[a]) & set(subj[b]))
            xa, xb = [subj[a][s] for s in common], [subj[b][s] for s in common]
            ax.scatter(xa, xb, s=14)
            lim = max(max(xa), max(xb)) * 1.05
            ax.plot([0, lim], [0, lim], "k:", lw=0.8)
            ax.set_xlabel(f"{a} MAE"); ax.set_ylabel(f"{b} MAE"); ax.set_title(f"{sum(np.array(xb) < np.array(xa))}/{len(common)} subjects: y better", fontsize=8)
        fig.tight_layout()
        p_sc = out_path(cfg, f"fig_subject_scatter_{tag}", "png", stamp=False)
        fig.savefig(p_sc, dpi=150); plt.close(fig)
    log.info("report written: %s", [str(p) for p in paths.values()] + [str(p_par), str(p_abl), str(p_ba)])
    cmd = f'python -m src.evaluate --report "{pattern}"'
    register_claim(f"report.{tag}.n_clips", len(res["clips"]), res["_file"], cmd)
    prefix = f"results.{res['dataset']}.{res['split']}.{res['tag']}"
    for skey, fams in res["strata"].items():                 # stratified MAE and group sizes
        for fam, groups in fams.items():
            for g, v in groups.items():
                register_claim(f"{prefix}.{fam}.strata.{skey}.{g}", v["mae"], res["_file"], cmd)
                register_claim(f"{prefix}.{fam}.strata_n.{skey}.{g}", v["n"], res["_file"], cmd)
    # per-subject head-to-head counts (how many subjects does model B beat model A on?)
    for a, b in (("rgb_pos", "event_pos@0.05"), ("event_pos@0.05", "ann_th0.05_s0"), ("ann_th0.05_s0", "snn_th0.05_s0"), ("rgb_pos", "ann_th0.05_s0")):
        if a in subj and b in subj:
            common = sorted(set(subj[a]) & set(subj[b]))
            register_claim(f"{prefix}.wins.{b}_beats_{a}", int(sum(subj[b][s] < subj[a][s] for s in common)), res["_file"], cmd)
            register_claim(f"{prefix}.wins.n_subjects", len(common), res["_file"], cmd)
