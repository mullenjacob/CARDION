"""Does limiting the CPU thread pools (PyTorch / OpenCV) make the live pipeline cheaper without changing its output?

    python scripts/bench_threads.py --frames 300

Runs the same real-face frames through the full pipeline (face tracking, event emulation, SNN + ANN) with different PyTorch thread counts and
reports wall time, processor seconds (all threads) and whether the SNN/ANN outputs and all seven BPM estimates are bit-identical to the default.
Results: ``out/bench_threads_<time>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_guidance import clips, read_frames  # noqa: E402
from src.cardion_app.engine import load_models  # noqa: E402
from src.cardion_app.pipeline import METHODS, Pipeline  # noqa: E402
from src.common import load_config, timestamp  # noqa: E402


def run(cfg: dict, frames: list[np.ndarray], models: dict, threads: int | None) -> dict:
    """One full pass; returns timings and the numbers produced."""
    if threads is not None:
        torch.set_num_threads(threads)
    pl = Pipeline(cfg, models=models, use_landmarks=True, guidance=True)
    for i, fr in enumerate(frames[:20]):
        pl.push(fr, i / 30.0)
    pl.reset()
    cpu0, t0 = time.process_time(), time.perf_counter()
    ests = []
    for i, fr in enumerate(frames):
        pl.push(fr, i / 30.0)
        if i % 30 == 29:
            a = pl.estimate()
            ests.append({"status": a.status, "bpm": a.bpm, **{m: pl.last_est.get(m, {}).get("bpm") for m in METHODS}})
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    out = {"threads": threads if threads is not None else torch.get_num_threads(), "ms_per_frame": 1000 * wall / len(frames),
           "cpu_cores_used": cpu / wall, "snn_out": list(pl.bin_out["snn"]), "ann_out": list(pl.bin_out["ann"]), "estimates": ests,
           "timing_s": dict(pl.timing)}
    pl.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--threads", type=int, nargs="+", default=[0, 1, 2, 4])
    a = ap.parse_args()
    cfg = load_config()
    frames, _ = read_frames(clips(1)[0], a.frames / 30.0 + 1)
    frames = frames[: a.frames]
    models, _ = load_models(cfg)
    default_threads = torch.get_num_threads()
    base = None
    rows = []
    for th in a.threads:
        r = run(cfg, frames, models, None if th == 0 else th)
        if base is None:
            base = r
        same_net = r["snn_out"] == base["snn_out"] and r["ann_out"] == base["ann_out"]
        max_net_diff = float(max(np.max(np.abs(np.array(r[k]) - np.array(base[k]))) for k in ("snn_out", "ann_out")))
        same_est = r["estimates"] == base["estimates"]
        row = {k: r[k] for k in ("threads", "ms_per_frame", "cpu_cores_used", "timing_s")}
        row.update({"network_outputs_bit_identical": same_net, "max_abs_network_difference": max_net_diff, "all_estimates_identical": same_est})
        rows.append(row)
        print(f"torch threads={r['threads']:>2}  {r['ms_per_frame']:.1f} ms/frame  cpu={r['cpu_cores_used']:.2f} cores  "
              f"networks identical={same_net} (max diff {max_net_diff:.2e})  estimates identical={same_est}", flush=True)
    out = ROOT / "out" / f"bench_threads_{timestamp()}.json"
    out.write_text(json.dumps({"command": f"python scripts/bench_threads.py --frames {a.frames}", "default_torch_threads": default_threads, "rows": rows}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
