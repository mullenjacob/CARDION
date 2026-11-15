"""Print the main SCAMPS results table (markdown) from the newest saved result JSONs (used for the README)."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def newest(pattern: str) -> dict:
    """Load the newest JSON matching a glob in out/."""
    path = sorted(glob.glob(str(ROOT / "out" / pattern)))[-1]
    data = json.loads(Path(path).read_text())
    data["_name"] = Path(path).name
    return data


def main() -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    base, long_run = newest("results_scamps-all_th0.02_*.json"), newest("results_scamps-all_long0.02_*.json")
    print(f"<!-- {base['_name']} ; {long_run['_name']} -->")
    print("| method | MAE (BPM) | 95% CI over clips | energy proxy (uJ / 10 s) |")
    print("|---|---|---|---|")
    rows = base["rows"] + [dict(r, method="snn (40 epochs, post hoc)") for r in long_run["rows"] if r["method"] == "snn"]
    for r in rows:
        print(f"| {r['method']} | {r['mae_subject_mean']:.2f} | {r['mae_ci_lo']:.2f}-{r['mae_ci_hi']:.2f} | {r['energy_uj_per_10s']:.3g} |")


if __name__ == "__main__":
    main()
