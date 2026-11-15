"""Train the spiking pulse regressor (or the matched ANN) on emulated-event clips with proper model selection.

* Training windows are sampled at random *inside the clips of training subjects only*.
* The best epoch is chosen on the **validation** sets only. ``--diag-sets`` are evaluated every epoch purely for diagnosis
  (learning dynamics) and are never used for selection or early stopping.
* Event tensors are cached under ``data/events`` (one emulation per clip, threshold and seed); ``precompute`` fills the cache
  in parallel.
* Loss = MSE + (1 - Pearson) between the predicted waveform and the bandpassed, z-scored contact PPG.

    python -m src.train --model snn --train-sets mcd:train --val-sets mcd:val --diag-sets scamps:all --seed 0 --tag mcd_th0.02
    python -m src.train precompute --sets mcd:train mcd:val mcd:test --threshold 0.02 --workers 8
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import add_common_args, out_path, repo_path, resolve_cfg, save_json, set_seed, setup_logging
from .datasets import Clip, clip_specs, load_clip, subject_of
from .dvs_emulate import emulate_rgb, prepare_face
from .events import frame_sequence
from .rppg_classical import bandpass, detrend, reference_windows, sliding_hr
from .snn_model import build_model

log = logging.getLogger("cardion.train")


@dataclass
class Item:
    """One clip prepared for the networks: events (T,3,H,W), target, and what is needed to score HR."""
    spec: str
    subject: str
    fs: float
    x: np.ndarray
    y: np.ndarray
    ppg: np.ndarray


def label_pulse(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Bandpassed (0.7-4 Hz), z-scored contact PPG used as the regression target."""
    p = bandpass(detrend(ppg - ppg.mean()), fs, 0.7, 4.0)
    return ((p - p.mean()) / (p.std() + 1e-8)).astype(np.float32)


def event_cache_path(clip: Clip, cfg: dict, threshold: float, backend: str) -> Path:
    """Cache file for the event tensor of ``clip`` at ``threshold``."""
    pf = float(cfg["dvs"].get("prefilter_hz", 0) or 0)
    tag = f"_pf{pf:g}" if pf > 0 else ""
    return repo_path("data/events") / f"{clip.name.replace(':', '-')}_n{len(clip.rgb)}_{backend}_th{threshold}_s{cfg['seed']}{tag}.npz"


def prefilter_log_intensity(frames: np.ndarray, fs: float, cutoff_hz: float, eps: float = 1.0, order: int = 4) -> np.ndarray:
    """Causal Butterworth low-pass of every pixel's log intensity (flicker suppression assumption), returned as intensity.

    Real indoor video aliases 50/60 Hz mains flicker to a few Hz (e.g. 10 Hz at 30 fps) with amplitude far above the pulse; a pixel
    contrast detector then turns that into drift. This optional stage models a sensor/front-end with limited temporal bandwidth.
    """
    from scipy import signal
    sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    L = np.log(frames.astype(np.float32) + eps)
    zi = signal.sosfilt_zi(sos)[:, :, None, None] * L[0][None, None]
    out, _ = signal.sosfilt(sos, L, axis=0, zi=zi)
    return (np.exp(out) - eps).astype(np.float32)


def emulate_event_frames(face: np.ndarray, mask: np.ndarray, fs: float, cfg: dict, threshold: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emulate the three colour channels and accumulate events straight into 33 ms frame bins (no event list is ever stored).

    Returns (frames (T,3,H/2,W/2) float16 signed counts, per-channel events inside the skin ROI, per-channel total events). The
    channels are emulated as one (3H x W) image (no spatial coupling in the pixel model; independent mismatch and noise per pixel).
    """
    from .dvs_emulate import AnalyticDVS, _interp_frames
    T, H, W = face.shape[:3]
    h2, w2 = H // 2, W // 2
    stride = 3 * h2 * w2
    stacked = np.concatenate([face[..., c] for c in range(3)], axis=1)
    pf = float(cfg["dvs"].get("prefilter_hz", 0) or 0)
    if pf > 0:
        stacked = prefilter_log_intensity(stacked, fs, pf, float(cfg["dvs"]["log_eps"]))
    dvs = AnalyticDVS.from_cfg(cfg, threshold, seed=seed)
    dvs.start(stacked[0], fs)
    acc = np.zeros(T * stride, np.float32)
    n_roi, n_all = np.zeros(3, np.int64), np.zeros(3, np.int64)
    m = mask.astype(bool)
    for t_prev, t_now, img in _interp_frames(stacked, fs, dvs.up):
        for tt, rx, ry, rp in dvs.step(img, t_prev, t_now):
            b = np.minimum((tt * fs).astype(np.int64), T - 1)
            lo = int(b.min())
            c, y = ry // H, ry % H
            idx = (((b - lo) * 3 + c) * h2 + y // 2) * w2 + rx // 2
            span = int(b.max()) - lo + 1
            acc[lo * stride:(lo + span) * stride] += np.bincount(idx, weights=rp, minlength=span * stride).astype(np.float32)
            n_all += np.bincount(c, minlength=3)
            n_roi += np.bincount(c[m[y, rx]], minlength=3)
    return acc.reshape(T, 3, h2, w2).astype(np.float16), n_roi, n_all


def event_tensor(clip: Clip, cfg: dict, threshold: float | None = None, backend: str | None = None) -> np.ndarray:
    """(T,3,H/2,W/2) float16 event-frame tensor for a clip, cached under ``data/events``."""
    th = threshold if threshold is not None else cfg["dvs"]["threshold"]
    be = backend or cfg["dvs"]["backend"]
    cache = event_cache_path(clip, cfg, th, be)
    if cache.exists():
        return np.load(cache)["x"]
    face, mask = prepare_face(clip, int(cfg["dvs"]["crop_size"]))
    x, n_roi, n_all = emulate_event_frames(face, mask, clip.fs, cfg, th, cfg["seed"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, n_roi=n_roi, n_all=n_all, mask32=cv2.resize(mask, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_NEAREST))
    return x


def event_meta(clip: Clip, cfg: dict, threshold: float, backend: str | None = None) -> dict:
    """ROI event counts and the 32x32 ROI mask stored next to the cached event tensor (computing it if absent)."""
    be = backend or cfg["dvs"]["backend"]
    cache = event_cache_path(clip, cfg, threshold, be)
    if not cache.exists() or "n_roi" not in np.load(cache).files:
        cache.unlink(missing_ok=True)
        event_tensor(clip, cfg, threshold, be)
    z = np.load(cache)
    return {"n_roi": z["n_roi"], "n_all": z["n_all"], "mask32": z["mask32"]}


def parse_sets(items: list[str] | None, cfg: dict) -> list[str]:
    """Expand ``dataset:split`` strings into clip specs."""
    specs: list[str] = []
    for it in items or []:
        ds, _, split = it.partition(":")
        specs += clip_specs(ds, split or "all", cfg)
    return specs


def logdiff_tensor(clip: Clip, cfg: dict, threshold: float) -> np.ndarray:
    """Control input: *unquantised* frame-to-frame log-intensity change on the same 32x32 grid, in units of ``1/threshold``.

    This is what an ideal, noise-free-quantiser contrast detector would integrate; it removes the thresholding/refractory/shot-noise
    stages of the emulator but keeps the camera noise, compression artefacts and flicker of the video.
    """
    face, _ = prepare_face(clip, int(cfg["dvs"]["crop_size"]))
    L = np.log(face.astype(np.float32) + float(cfg["dvs"]["log_eps"]))
    d = np.diff(L, axis=0, append=L[-1:]) / threshold                              # (T,H,W,3): change k->k+1 sits in bin k (like events)
    T, H, W, _ = d.shape
    return d.reshape(T, H // 2, 2, W // 2, 2, 3).sum(axis=(2, 4)).transpose(0, 3, 1, 2).astype(np.float16)


def make_input(clip: Clip, cfg: dict, threshold: float, kind: str = "events") -> np.ndarray:
    """Network input tensor (T,3,H/2,W/2) for ``kind`` in {events, logdiff}."""
    return logdiff_tensor(clip, cfg, threshold) if kind == "logdiff" else event_tensor(clip, cfg, threshold)


def build_items(specs: list[str], cfg: dict, threshold: float, kind: str = "events") -> list[Item]:
    """Load clips and input tensors for ``specs`` (emulating events on demand)."""
    items = []
    for s in specs:
        clip = load_clip(s, cfg)
        x = make_input(clip, cfg, threshold, kind)
        k = min(len(x), len(clip.ppg))
        items.append(Item(s, subject_of(s), clip.fs, x[:k], label_pulse(clip.ppg, clip.fs)[:k], clip.ppg[:k]))
    return items


def _precompute_one(args: tuple) -> str:
    """Worker: make sure the event cache for one (spec, threshold) exists."""
    spec, cfg, th = args
    torch.set_num_threads(1)
    clip = load_clip(spec, cfg)
    event_tensor(clip, cfg, th)
    return spec


def precompute(specs: list[str], cfg: dict, thresholds: list[float], workers: int) -> None:
    """Fill the event cache for all specs/thresholds using ``workers`` processes."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, cfg, th) for th in thresholds for s in specs]
    log.info("precomputing %d event tensors with %d workers", len(jobs), workers)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(_precompute_one, jobs)):
            if i % 10 == 0:
                log.info("  %d/%d %s (%.0fs)", i + 1, len(jobs), s, time.time() - t0)


def corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation per window (mean over batch)."""
    p, t = pred - pred.mean(1, keepdim=True), target - target.mean(1, keepdim=True)
    r = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-8)
    return (1 - r).mean()


def sample_batch(items: list[Item], rng: np.random.Generator, batch: int, seq: int) -> tuple[np.ndarray, np.ndarray]:
    """Random windows (uniform over clips, uniform start) -> (B,seq,3,H,W), (B,seq)."""
    xs, ys = [], []
    for _ in range(batch):
        it = items[rng.integers(len(items))]
        s = int(rng.integers(0, len(it.x) - seq + 1))
        xs.append(it.x[s:s + seq]); ys.append(it.y[s:s + seq])
    return np.stack(xs).astype(np.float32), np.stack(ys)


@torch.no_grad()
def predict_pulse(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Run the model over a whole clip (recurrent state carried across the full sequence)."""
    model.eval()
    return model(torch.from_numpy(x.astype(np.float32))[None].to(device))[0].cpu().numpy()


@torch.no_grad()
def predict_many(model: torch.nn.Module, items: list[Item], device: str, batch: int = 8) -> list[np.ndarray]:
    """Batched whole-clip inference (clips are truncated to the shortest in each batch, then restored in order)."""
    model.eval()
    out: list[np.ndarray | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i].x))
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        T = min(len(items[i].x) for i in idx)
        x = torch.from_numpy(np.stack([items[i].x[:T] for i in idx]).astype(np.float32)).to(device)
        y = model(x).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = y[j]
    return out  # type: ignore[return-value]


def window_errors(pulse: np.ndarray, it: Item, cfg: dict) -> np.ndarray:
    """Absolute HR errors (BPM) on the valid reference windows of one clip."""
    r = cfg["rppg"]
    pulse = bandpass(pulse - pulse.mean(), it.fs, r["band_hz"][0], r["band_hz"][1])
    _, bpm = sliding_hr(pulse, it.fs, r["hr_window_s"], r["hr_stride_s"], tuple(r["hr_band_hz"]), r.get("hr_method", "peak"))
    ref, valid = reference_windows(it.ppg, it.fs, cfg)
    k = min(len(bpm), len(ref))
    err = np.abs(bpm[:k] - ref[:k])[valid[:k]]
    return err


def subject_mae(items: list[Item], pulses: list[np.ndarray], cfg: dict) -> tuple[float, dict[str, float]]:
    """Mean over subjects of the subject's mean window error (clips of one subject pooled)."""
    per: dict[str, list[float]] = {}
    for it, p in zip(items, pulses):
        e = window_errors(p, it, cfg)
        if len(e):
            per.setdefault(it.subject, []).append(float(e.mean()))
    subj = {s: float(np.mean(v)) for s, v in per.items()}
    return (float(np.mean(list(subj.values()))) if subj else float("nan")), subj


def train(model_kind: str, train_items: list[Item], val_items: list[Item], diag: dict[str, list[Item]], cfg: dict, epochs: int,
          windows_per_epoch: int, tag: str, train_seed: int, val_every: int = 1, diag_every: int = 5) -> dict:
    """Train one model; select the epoch on validation MAE; return history and checkpoint path."""
    s, device = cfg["snn"], cfg["device"]
    torch.manual_seed(train_seed)
    model = build_model(model_kind, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])
    rng = np.random.default_rng(train_seed)
    seq, bs = int(s["seq_len"]), int(s["batch_size"])
    ckpt = out_path(cfg, f"ckpt_{tag}", "pt")
    hist, best = [], float("inf")
    for ep in range(epochs):
        t0, losses = time.time(), []
        model.train()
        for _ in range(max(1, windows_per_epoch // bs)):
            xb, yb = sample_batch(train_items, rng, bs, seq)
            xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb) + corr_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        evaluate_now = (ep + 1) % val_every == 0 or ep + 1 == epochs
        if not evaluate_now:
            hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_mae": None})
            continue
        val_mae, _ = subject_mae(val_items, predict_many(model, val_items, device), cfg)
        row = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_mae": val_mae}
        if (ep + 1) % diag_every == 0 or ep + 1 == epochs:
            for name, its in diag.items():                  # diagnostic only: never used for selection
                row[f"diag_{name}"], _ = subject_mae(its, predict_many(model, its, device), cfg)
        hist.append(row)
        log.info("epoch %d/%d loss %.3f val MAE %.2f %s (%.0fs)", ep + 1, epochs, row["train_loss"], val_mae,
                 " ".join(f"{k}={v:.2f}" for k, v in row.items() if k.startswith("diag_")), time.time() - t0)
        if val_mae < best:
            best = val_mae
            torch.save({"state": model.state_dict(), "kind": model_kind, "cfg": cfg, "epoch": ep + 1, "val_mae": val_mae}, ckpt)
    return {"history": hist, "best_val_mae": best, "best_epoch": hist[int(np.argmin([h["val_mae"] if h["val_mae"] is not None else np.inf for h in hist]))]["epoch"], "checkpoint": str(ckpt)}


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("command", nargs="?", default="train", choices=["train", "precompute"])
    ap.add_argument("--model", default="snn", choices=["snn", "ann"])
    ap.add_argument("--train-sets", nargs="+", default=["synthetic:train"])
    ap.add_argument("--val-sets", nargs="+", default=["synthetic:val"])
    ap.add_argument("--diag-sets", nargs="*", default=[], help="monitored every epoch, NEVER used for selection")
    ap.add_argument("--sets", nargs="*", default=[], help="for precompute: dataset:split specs")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--thresholds", nargs="*", type=float, default=None, help="precompute: several thresholds")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--windows-per-epoch", type=int, default=256)
    ap.add_argument("--input", default="events", choices=["events", "logdiff"], help="network input (logdiff = unquantised control)")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--diag-every", type=int, default=6)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--train-seed", type=int, default=None, help="model init / window sampling seed (data seed stays --seed)")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = default)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    train_seed = args.train_seed if args.train_seed is not None else cfg["seed"]
    set_seed(train_seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    th = args.threshold if args.threshold is not None else cfg["dvs"]["threshold"]
    cfg["dvs"]["threshold"] = th                                  # stored in the checkpoint: models are tied to their training threshold
    cfg["snn"]["input_kind"] = args.input
    if args.command == "precompute":
        precompute(parse_sets(args.sets, cfg), cfg, args.thresholds or [th], args.workers)
        return
    tr_specs, va_specs = parse_sets(args.train_sets, cfg), parse_sets(args.val_sets, cfg)
    assert not ({subject_of(s) for s in tr_specs} & {subject_of(s) for s in va_specs}), "train/val subjects overlap"
    log.info("train clips %d, val clips %d, diag %s", len(tr_specs), len(va_specs), args.diag_sets)
    train_items, val_items = build_items(tr_specs, cfg, th, args.input), build_items(va_specs, cfg, th, args.input)
    diag = {d.replace(":", "_"): build_items(parse_sets([d], cfg), cfg, th, args.input) for d in args.diag_sets}
    res = train(args.model, train_items, val_items, diag, cfg, args.epochs or int(cfg["snn"]["epochs"]), args.windows_per_epoch,
                f"{args.model}_{args.tag}", train_seed, args.val_every, args.diag_every)
    res.update({"model": args.model, "seed": cfg["seed"], "train_seed": train_seed, "threshold": th, "train_sets": args.train_sets, "val_sets": args.val_sets,
                "diag_sets": args.diag_sets, "n_train_clips": len(tr_specs), "n_val_clips": len(va_specs)})
    save_json(res, out_path(cfg, f"learning_curve_{args.model}_{args.tag}", "json"))
    log.info("best val MAE %.2f at epoch %d; checkpoint %s", res["best_val_mae"], res["best_epoch"], res["checkpoint"])


if __name__ == "__main__":
    main()
