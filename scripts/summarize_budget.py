"""Does a 4x larger training budget help the ANN (threshold 0.05)? Compares best validation MAE of the 1x runs (3 seeds) with the 4x run.

Reads ``out/learning_curve_ann_mcd_th0.05_s*.json`` and ``out/learning_curve_ann_mcdbig_th0.05_s0.json`` (validation subjects only).
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
from src.common import load_config, out_path, register_claim, save_json  # noqa: E402


def best_val(path: str) -> float:
    """Best validation MAE of a run."""
    return float(json.loads(Path(path).read_text())["best_val_mae"])


def main() -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    base = [best_val(f) for f in sorted(glob.glob(str(ROOT / "out" / "learning_curve_ann_mcd_th0.05_s*.json")))]
    big = [best_val(f) for f in sorted(glob.glob(str(ROOT / "out" / "learning_curve_ann_mcdbig_th0.05_s*.json")))]
    res = {"base_mean": float(np.mean(base)), "base_min": float(np.min(base)), "base_max": float(np.max(base)), "n_base": len(base),
           "big_best_val": float(np.min(big)) if big else None, "n_big": len(big), "windows_per_epoch_base": 128, "windows_per_epoch_big": 512}
    path = save_json(res, out_path(cfg, "budget_experiment", "json"))
    print(res)
    for k, v in res.items():
        if v is not None:
            register_claim(f"budget.{k}", v, path, "python scripts/summarize_budget.py")


if __name__ == "__main__":
    main()
