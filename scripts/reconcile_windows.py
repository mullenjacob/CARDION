"""Reconcile the window counts of the dataset table (Table II) with those of the evaluation (Tables III-VII).

    python scripts/reconcile_windows.py

Two code paths count 10 s windows (1 s stride) of the same recordings:

* ``scripts/dataset_stats.py`` (dataset table): excerpt = ``int(60 s * effective_fs)`` samples of the contact PPG, where the effective frame rate is
  decoded frames / wall-clock span from the capture time stamps.
* ``src/datasets.load_mcd_clip`` (evaluation): excerpt = ``int(60 * 29.9) = 1794`` *frames*, whatever the effective frame rate - so a recording captured at
  18 fps covers 97 s of wall-clock time and yields more windows.

This script recomputes both from the raw PPG/meta files and compares them with the counts stored in the saved evaluation results (``windows`` and
``valid_windows`` per clip). Output: ``out/window_reconciliation_<time>.json`` / ``.md``; totals are registered as claims ``winrec.*``.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, register_claim, repo_path, timestamp  # noqa: E402
from src.datasets import clip_specs, mcd_effective_fs  # noqa: E402
from src.rppg_classical import reference_windows  # noqa: E402

CMD = "python scripts/reconcile_windows.py"


def main() -> None:
    cfg = load_config()
    root = repo_path(cfg["datasets"]["mcd_root"])
    d = cfg["datasets"]["mcd"]
    if not (root / "ppg_sync").is_dir():
        print(f"MCD-rPPG PPG/meta files not found under {root}. Download them first (python scripts/download_mcd.py); the saved result of this check is "
              "out/window_reconciliation_*.json. Nothing was changed.")
        raise SystemExit(0)
    out: dict = {"command": CMD, "splits": {}}
    md = ["# Window-count reconciliation", "",
          "| split | dataset-table windows (60 s at effective fps) | evaluation windows (1794 frames) | evaluation valid windows (stored) | evaluation valid windows (recomputed) | clips whose counts differ |", "|---|---|---|---|---|---|"]
    for split, tag in (("val", "val_all"), ("test", "final")):
        res = json.loads(Path(sorted(glob.glob(str(ROOT / "out" / f"results_mcd-{split}_{tag}_*.json")))[-1]).read_text())
        stored = {c["spec"]: (c["windows"], c["valid_windows"]) for c in res["per_clip"]}
        rows, tot = [], {"table": 0, "eval_all": 0, "eval_valid": 0, "stored_all": 0, "stored_valid": 0}
        for spec in clip_specs("mcd", split, cfg):
            pid, step = spec.split(":")[1].split("_")
            stem = f"{pid}_FullHDwebcam_{step}"
            ppg = np.loadtxt(root / "ppg_sync" / f"{stem}.txt")[:, 0]
            fs = mcd_effective_fs(root / "meta" / f"{stem}.txt", len(ppg))
            start = int(d["start_s"] * 29.9)
            n_tab = int(d["seconds"] * fs)                       # dataset_stats.py
            n_eval = min(int(d["seconds"] * 29.9), len(ppg) - start)      # load_mcd_clip: fixed frame count
            ref_t, val_t = reference_windows(ppg[start:start + n_tab].astype(np.float64), fs, cfg)
            ref_e, val_e = reference_windows(ppg[start:start + n_eval].astype(np.float64), fs, cfg)
            sw, sv = stored[spec]
            rows.append({"clip": spec, "fs": round(fs, 2), "frames_table": n_tab, "frames_eval": n_eval, "table_valid": int(val_t.sum()), "eval_windows": int(len(ref_e)),
                         "eval_valid": int(val_e.sum()), "stored_windows": sw, "stored_valid": sv})
            tot["table"] += int(val_t.sum()); tot["eval_all"] += int(len(ref_e)); tot["eval_valid"] += int(val_e.sum()); tot["stored_all"] += sw; tot["stored_valid"] += sv
        diff = [r for r in rows if r["table_valid"] != r["stored_valid"]]
        out["splits"][split] = {"totals": tot, "clips": rows, "clips_with_difference": [r["clip"] for r in diff]}
        md.append(f"| {split} | {tot['table']} | {tot['eval_all']} | {tot['stored_valid']} | {tot['eval_valid']} | {len(diff)} |")
        for k, v in tot.items():
            register_claim(f"winrec.{split}.{k}", v, ROOT / "out" / f"window_reconciliation_{TS}.json", CMD)
        register_claim(f"winrec.{split}.n_clips_differ", len(diff), ROOT / "out" / f"window_reconciliation_{TS}.json", CMD)
        for r in diff:
            cid = r['clip'].split(':')[1]
            for k in ('fs', 'frames_table', 'frames_eval', 'table_valid', 'stored_valid'):
                register_claim(f"winrec.{split}.clip{cid}.{k}", r[k], ROOT / "out" / f"window_reconciliation_{TS}.json", CMD)
        md += ["", f"Clips whose dataset-table count differs from the evaluation ({split}):", "", "| clip | effective fps | frames (table) | frames (evaluation) | valid (table) | valid (evaluation, stored) |", "|---|---|---|---|---|---|"]
        md += [f"| {r['clip']} | {r['fs']} | {r['frames_table']} | {r['frames_eval']} | {r['table_valid']} | {r['stored_valid']} |" for r in diff]
        md.append("")
    (ROOT / "out" / f"window_reconciliation_{TS}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "out" / f"window_reconciliation_{TS}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


TS = timestamp()

if __name__ == "__main__":
    main()
