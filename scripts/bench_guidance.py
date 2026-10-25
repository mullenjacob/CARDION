"""Does the positioning assistant slow the live pipeline down?  Pushes the same real-face frames through the pipeline with and without
guidance (RGB path, then the full event + SNN/ANN path) and times the face detector with one and with two faces allowed.

    python scripts/bench_guidance.py --frames 240

Results (ms per frame, frames per second) are printed and saved to ``out/guidance_benchmark_<time>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_guidance import clips, read_frames  # noqa: E402
from src.cardion_app.engine import load_models  # noqa: E402
from src.cardion_app.pipeline import Pipeline  # noqa: E402
from src.common import load_config, timestamp  # noqa: E402


def time_pipeline(cfg: dict, frames: list[np.ndarray], guidance: bool, events: bool, models: dict) -> dict:
    """Seconds per frame for push() plus the 1 Hz estimate() calls."""
    pl = Pipeline(cfg, models=models if events else None, use_landmarks=True, guidance=guidance)
    pl.set_events(events)
    for i, fr in enumerate(frames[:20]):                      # warm-up (model load, first detections)
        pl.push(fr, i / 30.0)
    pl.reset()
    t0 = time.perf_counter()
    for i, fr in enumerate(frames):
        pl.push(fr, i / 30.0)
        if i % 30 == 29:
            pl.estimate()
    dt = time.perf_counter() - t0
    return {"guidance": guidance, "events": events, "ms_per_frame": 1000 * dt / len(frames), "fps": len(frames) / dt}


def detector_cost(cfg: dict, frames: list[np.ndarray], num_faces: int) -> float:
    """Milliseconds per detection with the landmarker configured for ``num_faces``."""
    from mediapipe.tasks.python import BaseOptions, vision
    from src.capture import LandmarkSkinROI, ensure_model
    import mediapipe as mp
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=ensure_model(cfg).read_bytes()),
                                        running_mode=vision.RunningMode.IMAGE, num_faces=num_faces)
    lm = vision.FaceLandmarker.create_from_options(opts)
    imgs = [mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)) for f in frames[:60]]
    lm.detect(imgs[0])
    t0 = time.perf_counter()
    for im in imgs:
        lm.detect(im)
    return 1000 * (time.perf_counter() - t0) / len(imgs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config()
    frames, _ = read_frames(clips(1)[0], a.frames / 30.0 + 1)
    frames = frames[: a.frames]
    models, _ = load_models(cfg)
    rows = []
    for events in (False, True):
        for _ in range(a.repeat):
            for guidance in (False, True):
                r = time_pipeline(cfg, frames, guidance, events, models)
                rows.append(r)
                print(f"events={events!s:5} guidance={guidance!s:5} {r['ms_per_frame']:.1f} ms/frame = {r['fps']:.1f} fps", flush=True)
    det = {f"num_faces={n}": detector_cost(cfg, frames, n) for n in (1, 2)}
    print("face detector:", {k: f"{v:.1f} ms" for k, v in det.items()})
    out = ROOT / "out" / f"guidance_benchmark_{timestamp()}.json"
    out.write_text(json.dumps({"command": f"python scripts/bench_guidance.py --frames {a.frames}", "pipeline": rows, "detector_ms": det}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
