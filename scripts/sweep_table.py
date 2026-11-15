"""Print the threshold-sweep summary (mean over clips) as a markdown table from the newest out/dvs_sweep_*.json."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=None, help="sweep JSON (default: newest in out/)")
    args = ap.parse_args()
    path = args.file or sorted(glob.glob(str(ROOT / "out" / "dvs_sweep_*.json")))[-1]
    res = json.loads(Path(path).read_text())
    print(f"<!-- {Path(path).name} -->")
    print(f"| threshold | events/px/s | mean abs r (net events vs PPG) | chance abs r (mismatched PPG) |")
    print("|---|---|---|---|")
    for i, th in enumerate(res["thresholds"]):
        rows = [res["per_clip"][c][i] for c in res["per_clip"]]
        rate = np.mean([x["events_per_pixel_per_s"] for x in rows])
        r = np.mean([abs(x["r_net_sum"]) for x in rows])
        null = np.mean([abs(x["r_null_green"]) for x in rows])
        print(f"| {th} | {rate:.3g} | {r:.2f} | {null:.2f} |")


if __name__ == "__main__":
    main()
