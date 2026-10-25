"""Summarise the learning-dynamics diagnostic: synthetic-trained SNN/ANN runs monitored on SCAMPS every epoch (never used for selection).

Answers "why did a 40-epoch SNN reach 0.20 BPM on SCAMPS while the 12-epoch one reached 9.71?": it compares cross-domain error across
epochs and seeds against the (synthetic) validation error that selected the checkpoint. Reads ``out/diag_<model>_s<seed>.log``.
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
from src.common import out_path, register_claim, load_config, save_json  # noqa: E402


def parse(path: str) -> dict:
    """Epoch table from a training log (handles UTF-16 logs written by PowerShell redirection)."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="ignore")
    rows = [(int(m[1]), float(m[2]), float(m[3])) for m in re.finditer(r"epoch (\d+)/\d+ loss [\d.]+ val MAE ([\d.]+) diag_scamps_all=([\d.]+)", text)]
    return {"epoch": [r[0] for r in rows], "val": [r[1] for r in rows], "scamps": [r[2] for r in rows]}


def main() -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = load_config()
    runs = {Path(f).stem.replace("diag_", ""): parse(f) for f in sorted(glob.glob(str(ROOT / "out" / "diag_*_s*.log")))}
    runs = {k: v for k, v in runs.items() if len(v["epoch"]) >= 20}
    summary = {}
    for name, r in runs.items():
        sc, val = np.array(r["scamps"]), np.array(r["val"])
        b = int(np.argmin(val))
        summary[name] = {"epochs": len(sc), "best_val_epoch": r["epoch"][b], "best_val_mae": float(val[b]), "scamps_at_selected": float(sc[b]),
                         "scamps_median": float(np.median(sc)), "scamps_min": float(sc.min()), "scamps_max": float(sc.max()),
                         "frac_epochs_below_1p5": float(np.mean(sc < 1.5)), "val_min": float(val.min()), "val_max": float(val.max()),
                         "corr_val_vs_scamps": float(np.corrcoef(val, sc)[0, 1])}
    path = save_json({"runs": summary, "curves": runs}, out_path(cfg, "diag_epoch_dynamics", "json"))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for name, r in runs.items():
        ax.plot(r["epoch"], r["scamps"], marker=".", ms=3, lw=0.9, label=name)
    ax.set_yscale("log"); ax.set_xlabel("training epoch"); ax.set_ylabel("SCAMPS MAE (BPM), monitored only")
    ax.axhline(1.5, color="k", ls=":", lw=0.7); ax.legend(fontsize=7)
    fig.tight_layout()
    fig_path = out_path(cfg, "fig_diag_epoch_dynamics", "png", stamp=False)
    fig.savefig(fig_path, dpi=150)
    cmd = "python scripts/summarize_diag.py"
    for name, s in summary.items():
        for k, v in s.items():
            register_claim(f"diag.{name}.{k}", v, path, cmd)
        print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})


if __name__ == "__main__":
    main()
