"""Evaluate every method on identical clips and windows; write result JSON/CSV, tables and figures into ``out/``.

Methods (one table row each): RGB green / POS / CHROM, classical event estimators (green, POS) at every requested contrast threshold,
and trained SNN / ANN checkpoints (several seeds per family). All estimates use the same 10 s sliding-window FFT protocol and are
scored against the contact-PPG HR on the *same valid windows*. Statistics are per **subject** (the clips of a subject are pooled).

Protected splits: ``mcd:test`` can only be evaluated with ``--final`` (it is meant to be touched once, after all choices are frozen).

    python -m src.evaluate --dataset mcd --split val --tag val --thresholds 0.02 0.05 \
        --models snn_th0.02="out/ckpt_snn_mcd_th0.02_s*.pt" ann_th0.02="out/ckpt_ann_mcd_th0.02_s*.pt"
    python -m src.evaluate --dataset mcd --split test --final --tag final ...        # once
    python -m src.evaluate --report "out/results_mcd-test_final_*.json"            # tables + figures from saved JSON
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .common import add_common_args, out_path, register_claim, resolve_cfg, save_json, set_seed, setup_logging
from .datasets import Clip, clip_specs, load_clip, state_of, subject_of
from .efficiency import E_AC_PJ, E_MAC_PJ, macs_per_step, sops_from_counts
from .stats import bootstrap_ci, hr_metrics, paired_wilcoxon

log = logging.getLogger("cardion.evaluate")
PROTECTED = {("mcd", "test")}
POS_MACS_PER_SAMPLE, CHROM_MACS_PER_SAMPLE = 600, 60       # analytic counts (see paper, efficiency accounting)
EMULATION_OPS_PER_PIXEL_SUBSTEP = 8                        # log, subtract, 2 compares, floor, mismatch scale, leak, shot test
DVS_UPSAMPLE = 4


def energy_uj(adds: float, macs: float, sops: float) -> float:
    """Energy proxy in microjoules: accumulates at 0.9 pJ, MACs at 4.6 pJ (45 nm, Horowitz 2014). Not a measurement."""
    return ((adds + sops) * E_AC_PJ + macs * E_MAC_PJ) * 1e-6


def load_models(specs: list[str], cfg: dict) -> dict[str, dict]:
    """``NAME=glob`` -> {name_sK: {model, kind, threshold, path}} for every matching checkpoint (one per seed)."""
    import torch
    from .snn_model import build_model
    out: dict[str, dict] = {}
    for spec in specs or []:
        name, _, pattern = spec.partition("=")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"no checkpoint matches {pattern}")
        for k, f in enumerate(files):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = build_model(ck["kind"], ck.get("cfg", cfg))
            m.load_state_dict(ck["state"])
            m.eval()
            out[f"{name}_s{k}"] = {"model": m, "kind": ck["kind"], "threshold": float(ck["cfg"]["dvs"]["threshold"]), "path": f,
                                   "family": name, "val_mae": ck.get("val_mae"), "epoch": ck.get("epoch"),
                                   "input_kind": ck.get("cfg", {}).get("snn", {}).get("input_kind", "events")}
            log.info("loaded %s <- %s (epoch %s, val MAE %s, threshold %s)", f"{name}_s{k}", Path(f).name, ck.get("epoch"), ck.get("val_mae"),
                     out[f"{name}_s{k}"]["threshold"])
    return out


def clip_covariates(clip: Clip, ref_bpm: np.ndarray) -> dict:
    """Clip-level conditions used for stratified results (rest/exercise, brightness, frame-difference activity, heart-rate range)."""
    luma = float(np.mean(0.299 * clip.rgb[:, 0] + 0.587 * clip.rgb[:, 1] + 0.114 * clip.rgb[:, 2]))
    g = clip.face.mean(axis=3).astype(np.float32)
    motion = float(np.median(np.abs(np.diff(g, axis=0)).mean(axis=(1, 2)))) if len(g) > 2 else float("nan")
    return {"state": state_of(clip.name), "luma": luma, "activity": motion, "ref_hr": float(np.mean(ref_bpm)) if len(ref_bpm) else float("nan")}


def evaluate_clip(spec: str, cfg: dict, thresholds: list[float], models: dict) -> dict:
    """All methods on one clip -> {method: {bpm, ref, valid, adds, macs, sops, wall_ms}}, plus covariates."""
    import torch
    from .rppg_classical import bandpass, detrend, estimate_q, reference_windows, sliding_hr_q
    from .rppg_events import pos_from_traces
    from .train import event_meta, event_tensor, make_input, predict_pulse
    r = cfg["rppg"]
    band, hr_band, hm = tuple(r["band_hz"]), tuple(r["hr_band_hz"]), r.get("hr_method", "peak")
    clip = load_clip(spec, cfg)
    dur = len(clip.rgb) / clip.fs
    scale = 10.0 / dur
    ref, valid = reference_windows(clip.ppg, clip.fs, cfg)
    roi_px = float(cv2.resize(clip.face_mask, (cfg["dvs"]["crop_size"],) * 2, interpolation=cv2.INTER_NEAREST).sum())
    T = len(clip.rgb)
    out: dict[str, dict] = {}

    def add(name: str, bpm: np.ndarray, wall: float, adds: float = 0.0, macs: float = 0.0, sops: float = 0.0, emu: float = 0.0,
            q: np.ndarray | None = None) -> None:
        k = min(len(bpm), len(ref))
        out[name] = {"bpm": bpm[:k], "ref": ref[:k], "valid": valid[:k], "q": (q[:k] if q is not None else np.zeros(k)), "adds": adds * scale, "macs": macs * scale, "sops": sops * scale,
                     "emulation_ops": emu * scale, "wall_ms": 1000 * wall * scale}

    for m, ch, mac in (("green", 1, 0), ("pos", 3, POS_MACS_PER_SAMPLE), ("chrom", 3, CHROM_MACS_PER_SAMPLE)):
        t0 = time.perf_counter()
        _, _, bpm, q = estimate_q(clip.rgb, clip.fs, m, cfg)
        add(f"rgb_{m}", bpm, time.perf_counter() - t0, adds=roi_px * ch * T, macs=mac * T, q=q)
    for th in thresholds:
        x = event_tensor(clip, cfg, th).astype(np.float32)[: len(clip.rgb)]
        meta = event_meta(clip, cfg, th)
        mask = meta["mask32"].astype(bool)
        emu = EMULATION_OPS_PER_PIXEL_SUBSTEP * DVS_UPSAMPLE * 3 * cfg["dvs"]["crop_size"] ** 2 * (T - 1)
        net = x[:, :, mask].sum(axis=2) / max(mask.sum(), 1)                    # (T,3) net events per ROI pixel per bin
        tr = np.cumsum(net, axis=0)
        t0 = time.perf_counter()
        g = bandpass(detrend(tr[:, 1], 100.0), clip.fs, band[0], band[1])
        _, b, q = sliding_hr_q(g, clip.fs, r["hr_window_s"], r["hr_stride_s"], hr_band, hm)
        add(f"event_green@{th}", b, time.perf_counter() - t0, adds=float(meta["n_roi"][1]), emu=emu, q=q)
        t0 = time.perf_counter()
        p = pos_from_traces(tr, clip.fs, r["pos_window_s"], band)
        _, b, q = sliding_hr_q(p, clip.fs, r["hr_window_s"], r["hr_stride_s"], hr_band, hm)
        add(f"event_pos@{th}", b, time.perf_counter() - t0, adds=float(meta["n_roi"].sum()), macs=POS_MACS_PER_SAMPLE * T, emu=emu, q=q)
        for name, md in models.items():
            if abs(md["threshold"] - th) > 1e-9:
                continue
            m = md["model"]
            xin = x if md["input_kind"] == "events" else make_input(clip, cfg, th, md["input_kind"]).astype(np.float32)[: len(clip.rgb)]
            t0 = time.perf_counter()
            pulse = predict_pulse(m, xin, "cpu")
            wall = time.perf_counter() - t0
            pulse = bandpass(pulse - pulse.mean(), clip.fs, band[0], band[1])
            _, bpm, q = sliding_hr_q(pulse, clip.fs, r["hr_window_s"], r["hr_stride_s"], hr_band, hm)
            if md["kind"] == "snn":
                add(name, bpm, wall, sops=sops_from_counts(m), emu=emu, q=q)
            else:
                add(name, bpm, wall, macs=macs_per_step(m, xin.shape[2], xin.shape[3]) * xin.shape[0], emu=emu, q=q)
    return {"methods": out, "cov": clip_covariates(clip, ref[valid[: len(ref)]] if valid.any() else ref), "spec": spec}


def subject_table(per_clip: list[dict], method: str) -> dict[str, float]:
    """Subject -> mean over the subject's clips of the clip's mean absolute error (valid windows only)."""
    acc: dict[str, list[float]] = {}
    for c in per_clip:
        m = c["methods"].get(method)
        if m is None or not m["valid"].any():
            continue
        e = np.abs(m["bpm"] - m["ref"])[m["valid"]]
        acc.setdefault(subject_of(c["spec"]), []).append(float(e.mean()))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n, prev, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        prev = max(prev, min(1.0, (n - i) * p))
        out[k] = prev
    return out


def summarise(per_clip: list[dict], cfg: dict, models: dict, reference: str) -> dict:
    """Rows (per method and per model family), paired tests, strata."""
    methods = sorted({m for c in per_clip for m in c["methods"]})
    n_boot, ci = int(cfg["eval"]["bootstrap_n"]), float(cfg["eval"]["ci"])
    families: dict[str, list[str]] = {}
    for m in methods:
        fam = models[m]["family"] if m in models else m
        families.setdefault(fam, []).append(m)
    rows, subj_by_fam = [], {}
    for fam, members in families.items():
        tabs = [subject_table(per_clip, m) for m in members]
        subjects = sorted(set.intersection(*[set(t) for t in tabs]))
        per_subj = np.array([np.mean([t[s] for t in tabs]) for s in subjects])          # seed-averaged subject MAE
        subj_by_fam[fam] = dict(zip(subjects, per_subj))
        mean, lo, hi = bootstrap_ci(per_subj, n_boot, ci, cfg["seed"])
        seed_means = [float(np.mean(list(t.values()))) for t in tabs]
        def pooled(m: str, key: str) -> np.ndarray:
            return np.concatenate([c["methods"][m][key][c["methods"][m]["valid"]] for c in per_clip if m in c["methods"]])
        per_member = [hr_metrics(pooled(m, "bpm"), pooled(m, "ref")) for m in members]       # window-pooled agreement metrics, averaged over seeds
        win = {k: float(np.mean([d[k] for d in per_member])) for k in per_member[0]}
        agg = lambda k: float(np.mean([np.mean([c["methods"][m][k] for c in per_clip if m in c["methods"]]) for m in members]))  # noqa: E731
        rows.append({"method": fam, "n_seeds": len(members), "n_subjects": len(subjects), "mae": mean, "mae_ci_lo": lo, "mae_ci_hi": hi,
                     "mae_seed_sd": float(np.std(seed_means, ddof=1)) if len(members) > 1 else None, "mae_seed_min": float(min(seed_means)),
                     "mae_seed_max": float(max(seed_means)), **{("win_mae" if k == "mae" else k): v for k, v in win.items()},
                     "adds_per_10s": agg("adds"), "macs_per_10s": agg("macs"), "sops_per_10s": agg("sops"), "emulation_ops_per_10s": agg("emulation_ops"),
                     "energy_uj_per_10s": energy_uj(agg("adds"), agg("macs"), agg("sops")), "wall_ms_per_10s": agg("wall_ms")})
    # paired tests vs the reference method and between selected pairs
    tests: dict[str, dict] = {}
    raw: dict[str, float] = {}
    pairs = [(f, reference) for f in subj_by_fam if f != reference]
    pairs += [(a, b) for a in subj_by_fam for b in subj_by_fam if a.startswith("snn") and b.startswith("ann") and a.split("_", 1)[-1] == b.split("_", 1)[-1]]
    pairs += [(a, b) for a in subj_by_fam for b in subj_by_fam if a.startswith("snn") and b.startswith("event_green@") and a.split("th")[-1] == b.split("@")[-1]]
    for a, b in dict.fromkeys(pairs):
        common = sorted(set(subj_by_fam[a]) & set(subj_by_fam[b]))
        if len(common) < 5:
            continue
        va, vb = np.array([subj_by_fam[a][s] for s in common]), np.array([subj_by_fam[b][s] for s in common])
        w = paired_wilcoxon(va, vb)
        tests[f"{a} vs {b}"] = {**w, "mean_diff": float(np.mean(va - vb))}
        raw[f"{a} vs {b}"] = w["p_value"]
    for k, p in holm(raw).items():
        tests[k]["p_holm"] = p
    # strata (clip level): state, brightness, activity, heart-rate range
    covs = [c["cov"] for c in per_clip]
    cuts = {k: np.nanpercentile([c[k] for c in covs], [33.3, 66.7]) for k in ("luma", "activity", "ref_hr")}
    def tier(c: dict, k: str) -> str:
        return "low" if c[k] <= cuts[k][0] else "mid" if c[k] <= cuts[k][1] else "high"
    strata: dict[str, dict] = {}
    for fam, members in families.items():
        for key, fn in (("state", lambda c: c["state"]), ("brightness", lambda c: tier(c, "luma")), ("activity", lambda c: tier(c, "activity")),
                        ("heart_rate", lambda c: tier(c, "ref_hr"))):
            groups: dict[str, list[float]] = {}
            for c in per_clip:
                vals = []
                for m in members:
                    mm = c["methods"].get(m)
                    if mm is not None and mm["valid"].any():
                        vals.append(float(np.abs(mm["bpm"] - mm["ref"])[mm["valid"]].mean()))
                if vals:
                    groups.setdefault(fn(c["cov"]), []).append(float(np.mean(vals)))
            strata.setdefault(key, {})[fam] = {g: {"mae": float(np.mean(v)), "n": len(v)} for g, v in groups.items()}
    return {"rows": rows, "tests": tests, "strata": strata, "strata_cuts": {k: [float(x) for x in v] for k, v in cuts.items()}}


def run_single(args: argparse.Namespace, cfg: dict) -> None:
    """Evaluate all methods on one dataset/split and write JSON/CSV and diagnostic figures."""
    if (args.dataset, args.split) in PROTECTED and not args.final:
        raise SystemExit(f"{args.dataset}:{args.split} is a protected final-test split. Pass --final only for the one frozen final evaluation.")
    models = load_models(args.models, cfg)
    specs = clip_specs(args.dataset, args.split, cfg, args.max_clips)
    thresholds = args.thresholds or [cfg["dvs"]["threshold"]]
    log.info("evaluating %d clips (%s:%s) at thresholds %s with %d models", len(specs), args.dataset, args.split, thresholds, len(models))
    per_clip = []
    for i, spec in enumerate(specs):
        per_clip.append(evaluate_clip(spec, cfg, thresholds, models))
        if i % 6 == 0:
            log.info("  %d/%d %s", i + 1, len(specs), spec)
    res = summarise(per_clip, cfg, models, args.reference)
    res.update({"dataset": args.dataset, "split": args.split, "tag": args.tag, "thresholds": thresholds, "clips": specs, "final": bool(args.final),
                "hr_method": cfg["rppg"].get("hr_method", "peak"), "ref_min_peak_ratio": cfg["rppg"].get("ref_min_peak_ratio"),
                "models": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in models.items()},
                "protocol": {"hr_window_s": cfg["rppg"]["hr_window_s"], "stride_s": cfg["rppg"]["hr_stride_s"]},
                "per_clip": [{"spec": c["spec"], "cov": c["cov"], "mae": {m: (float(np.abs(v["bpm"] - v["ref"])[v["valid"]].mean()) if v["valid"].any() else None)
                                                                         for m, v in c["methods"].items()},
                              "windows_data": {m: {"bpm": np.round(v["bpm"], 1).tolist(), "ref": np.round(v["ref"], 1).tolist(), "valid": v["valid"].tolist(),
                                                    "q": np.round(v["q"], 2).tolist()}
                                               for m, v in c["methods"].items()},
                              "valid_windows": int(next(iter(c["methods"].values()))["valid"].sum()),
                              "windows": int(len(next(iter(c["methods"].values()))["valid"]))} for c in per_clip]})
    stem = f"{args.dataset}-{args.split}_{args.tag}"
    path = save_json(res, out_path(cfg, "results", "json", tag=stem))
    with open(out_path(cfg, "results", "csv", tag=stem), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(res["rows"][0]))
        wr.writeheader(); wr.writerows(res["rows"])
    log.info("%-26s %5s %7s %17s %7s %7s %10s", "method", "seeds", "MAE", "95% CI (subjects)", "RMSE", "r", "energy uJ")
    for r in sorted(res["rows"], key=lambda r: r["mae"]):
        log.info("%-26s %5d %7.2f [%6.2f, %6.2f] %7.2f %7.3f %10.3g", r["method"], r["n_seeds"], r["mae"], r["mae_ci_lo"], r["mae_ci_hi"], r["rmse"],
                 r["pearson_r"], r["energy_uj_per_10s"])
    for k, t in res["tests"].items():
        log.info("paired Wilcoxon %-45s p=%.3g p_holm=%.3g mean diff %.2f", k, t["p_value"], t.get("p_holm", float("nan")), t["mean_diff"])
    cmd = f"python -m src.evaluate --dataset {args.dataset} --split {args.split} --tag {args.tag}" + (" --final" if args.final else "") + \
        " --thresholds " + " ".join(map(str, thresholds)) + "".join(f' --models {m}' for m in args.models or [])
    key = f"results.{args.dataset}.{args.split}.{args.tag}"
    for r in res["rows"]:
        for field in ("mae", "mae_ci_lo", "mae_ci_hi", "mae_seed_sd", "rmse", "pearson_r", "ccc", "ba_bias", "ba_loa_low", "ba_loa_high", "energy_uj_per_10s",
                      "adds_per_10s", "macs_per_10s", "sops_per_10s", "emulation_ops_per_10s", "wall_ms_per_10s", "n_seeds", "n_subjects"):
            if r.get(field) is not None and np.isfinite(r[field]):
                register_claim(f"{key}.{r['method']}.{field}", r[field], path, cmd)
    for k, t in res["tests"].items():
        for field in ("p_value", "p_holm", "mean_diff"):
            if field in t:
                register_claim(f"{key}.test[{k}].{field}", t[field], path, cmd)
    log.info("wrote %s", path)


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--dataset", default="mcd", choices=["synthetic", "ubfc", "pure", "scamps", "mcd"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--final", action="store_true", help="acknowledge evaluating a protected final-test split")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--thresholds", nargs="*", type=float, default=None)
    ap.add_argument("--models", nargs="*", default=[], metavar="NAME=GLOB", help="trained checkpoints (one per seed) grouped as a family")
    ap.add_argument("--reference", default="rgb_pos", help="method that paired tests compare against")
    ap.add_argument("--report", default=None, help="glob of results JSONs -> tables and figures (see report.py)")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    if args.report:
        from .report import build_report
        build_report(args.report, cfg)
    else:
        run_single(args, cfg)


if __name__ == "__main__":
    main()
