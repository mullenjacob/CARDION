"""Dataset statistics table (LaTeX) and claims: subjects/clips per split, effective fps, reference-HR range, valid-window fractions.

Reads only metadata/PPG (no labels are used for any model choice here); the protected test split's *reference HR distribution* is reported
descriptively, as is standard for dataset tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, out_path, register_claim, repo_path  # noqa: E402
from src.datasets import dataset_subjects, load_clip, clip_specs, mcd_effective_fs  # noqa: E402
from src.rppg_classical import reference_windows  # noqa: E402


def main() -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    root = repo_path(cfg["datasets"]["mcd_root"])
    rows = []
    for split in ("train", "val", "test"):
        specs = clip_specs("mcd", split, cfg)
        fs_l, refs, valid_frac = [], [], []
        for s in specs:
            cid = s.split(":")[1]
            pid, step = cid.split("_")
            stem = f"{pid}_FullHDwebcam_{step}"
            ppg = np.loadtxt(root / "ppg_sync" / f"{stem}.txt")[:, 0]
            fs = mcd_effective_fs(root / "meta" / f"{stem}.txt", len(ppg))
            start = int(cfg["datasets"]["mcd"]["start_s"] * 29.9)          # same excerpt as src.datasets.load_mcd_clip: a fixed number of FRAMES (60 s at the nominal 29.9 fps),
            n = min(int(cfg["datasets"]["mcd"]["seconds"] * 29.9), len(ppg) - start)   # so recordings captured at a lower effective rate cover more wall-clock time (see scripts/reconcile_windows.py)
            ref, valid = reference_windows(ppg[start:start + n], fs, cfg)
            fs_l.append(fs); refs.append(ref[valid]); valid_frac.append(valid.mean())
        refs = np.concatenate(refs)
        rows.append({"split": split, "subjects": len(specs) // 2, "clips": len(specs), "fs": float(np.median(fs_l)), "windows": int(len(refs)),
                     "hr_min": float(refs.min()), "hr_median": float(np.median(refs)), "hr_max": float(refs.max()), "valid_frac": float(np.mean(valid_frac))})
    lines = ["\\begin{tabular}{lrrrrr}\\toprule", "Split & subjects & clips & windows & ref. HR (BPM) median [min, max] & valid \\\\\\midrule"]
    for r in rows:
        lines.append(f"MCD-rPPG {r['split']} & {r['subjects']} & {r['clips']} & {r['windows']} & {r['hr_median']:.0f} [{r['hr_min']:.0f}, {r['hr_max']:.0f}] & {100 * r['valid_frac']:.0f}\\% \\\\")
    sc = dataset_subjects("scamps", cfg)
    lines.append(f"SCAMPS example (diagnostic) & {len(sc)} & {len(sc)} & -- & -- & -- \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    p_tex = out_path(cfg, "table_dataset", "tex", stamp=False)
    p_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p_json = out_path(cfg, "dataset_stats", "json", stamp=False)
    p_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        for k, v in r.items():
            if k != "split":
                register_claim(f"dataset.mcd.{r['split']}.{k}", v, p_json, "python scripts/dataset_stats.py")
    register_claim("dataset.scamps.clips", len(sc), p_json, "python scripts/dataset_stats.py")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
