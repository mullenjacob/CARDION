"""Pick the app's SNN and ANN from the trained checkpoints using VALIDATION error only and copy them to ``models/``.

Every ``out/ckpt_{snn,ann}_mcd_th*_s*.pt`` stores its best validation MAE (``val_mae``, mcd:val subjects). The checkpoint with the lowest
value per model type is exported together with a ``models.json`` describing training data, threshold and validation error. The common
event threshold for the live app (``app.threshold`` in the config) is the one of the better of the two models.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="out/ckpt_{kind}_mcd_th*_s*.pt")
    args = ap.parse_args()
    meta, best = {}, {}
    for kind in ("snn", "ann"):
        cands = []
        for f in glob.glob(str(ROOT / args.pattern.format(kind=kind))):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            cands.append((float(ck["val_mae"]), f, ck))
        if not cands:
            print(f"no {kind} checkpoints found")
            continue
        cands.sort(key=lambda c: c[0])
        v, f, ck = cands[0]
        shutil.copy(f, ROOT / "models" / f"{kind}.pt")
        th = float(ck["cfg"]["dvs"]["threshold"])
        best[kind] = th
        meta[kind] = {"source_checkpoint": Path(f).name, "val_mae_bpm": v, "epoch": ck["epoch"], "threshold": th,
                      "note": (f"Trained on 36 real subjects of MCD-rPPG (72 clips, simulated events at threshold {th}); selected on 12 validation "
                               f"subjects (MAE {v:.1f} BPM). Not better than the RGB methods - shown for comparison.")}
        print(kind, "->", Path(f).name, f"val MAE {v:.2f}, threshold {th}")
    (ROOT / "models" / "models.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if best:
        th = best.get("ann", next(iter(best.values())))
        cfgp = ROOT / "configs" / "default.yaml"
        txt = cfgp.read_text(encoding="utf-8-sig")
        import re
        txt = re.sub(r"(app:\s*\n\s*threshold:\s*)[\d.]+", rf"\g<1>{th}", txt)
        cfgp.write_text(txt, encoding="utf-8")
        print("app.threshold set to", th, "(threshold of the ANN, the better validation model)" if "ann" in best else "")


if __name__ == "__main__":
    main()
