"""Regression check that the positioning assistant did not change any heart-rate estimate.

Run it once per source tree (PYTHONPATH selects the tree) and diff the JSON lines, e.g. after `git archive backup-before-guidance`:

    PYTHONPATH=<old tree> python scripts/compare_estimates.py > old.txt ;  PYTHONPATH=<this tree> python scripts/compare_estimates.py > new.txt

Feeds the same 600 real frames of two recordings through the full pipeline (RGB, event, SNN, ANN) and prints the status and BPM of every method every second.
"""
import json, sys, cv2, glob, numpy as np
from pathlib import Path
from src.cardion_app.pipeline import Pipeline
from src.cardion_app.engine import load_models
from src.common import load_config
import src.cardion_app.pipeline as P
cfg = load_config()
models, _ = load_models(cfg)
out = {"root": P.__file__}
for name in ("8384", "5156"):
    path = glob.glob(str(Path(__file__).resolve().parent.parent / "data" / "raw" / "mcd_rppg" / "video" / f"{name}*.avi"))[0]
    cap = cv2.VideoCapture(path); frames = []
    while len(frames) < 600:
        ok, b = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(cv2.resize(b, (640, 480)), cv2.COLOR_BGR2RGB))
    pl = Pipeline(cfg, models=models, use_landmarks=True)
    rows = []
    for i, f in enumerate(frames):
        pl.push(f, i / 30.0)
        if i % 30 == 29 and pl.seconds() > 2:
            a = pl.estimate()
            rows.append([a.status, None if a.bpm is None else round(a.bpm, 6)] + [round(float(pl.last_est[k]["bpm"]), 6) for k in sorted(pl.last_est)])
    out[name] = rows
    if hasattr(pl, "close"): pl.close()
print("JSON" + json.dumps(out))

