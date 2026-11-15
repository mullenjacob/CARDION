"""Dataset loaders (UBFC-rPPG, PURE, SCAMPS, author clips), a synthetic fallback generator with a known
pulse, subject-disjoint splits, checksums and a split manifest CLI.

Every loader returns a :class:`Clip`. Splits are always by *subject*, never by frame or clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .common import add_common_args, repo_path, resolve_cfg, save_json, set_seed, setup_logging, timestamp

log = logging.getLogger("cardion.datasets")

SKIN_TONES = {  # approximate sRGB skin albedo (light -> dark); pulsatile contrast shrinks with melanin
    "I-II": ((224, 172, 148), 1.00), "III": ((198, 134, 108), 0.85),
    "IV": ((160, 105, 78), 0.65), "V": ((120, 78, 56), 0.50), "VI": ((84, 56, 40), 0.40)}
PULSE_SIGNATURE = np.array([0.25, 1.00, 0.60])  # relative R:G:B pulsatile absorption (green strongest)


@dataclass
class Clip:
    """One video clip with its contact-PPG reference."""
    name: str
    subject: str
    fs: float
    ppg: np.ndarray                       # (T,) contact PPG resampled to frame times
    rgb: np.ndarray                       # (T,3) mean skin RGB trace
    face: np.ndarray | None = None        # (T,S,S,3) uint8 face crop for event emulation
    face_mask: np.ndarray | None = None   # (S,S) uint8 skin mask in crop coordinates
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- synthetic generator
def _ppg_waveform(n: int, fs: float, hr0: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pulse waveform with slow HR variability and a dicrotic notch; returns (waveform, instantaneous bpm)."""
    t = np.arange(n) / fs
    drift = 1 + 0.04 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28)) + \
        0.02 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.28))
    f = hr0 / 60.0 * drift
    phase = 2 * np.pi * np.cumsum(f) / fs
    w = np.sin(phase) + 0.35 * np.sin(2 * phase + 1.0) + 0.1 * np.sin(3 * phase + 2.0)
    return (w - w.mean()) / w.std(), f * 60.0


def make_synthetic_clip(clip_id: int, cfg: dict, seed: int | None = None, hr0: float | None = None) -> Clip:
    """Render a small face-like clip with a *known* pulse, shading, texture, motion, flicker and noise."""
    sc = cfg["datasets"]["synthetic"]
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed * 100003 + clip_id)
    fs, size, n = float(sc["fps"]), int(sc["size"]), int(sc["clip_seconds"] * sc["fps"])
    tone = list(SKIN_TONES)[clip_id % len(SKIN_TONES)]
    albedo, contrast = SKIN_TONES[tone]
    hr0_drawn = rng.uniform(55, 120)
    hr0 = hr0_drawn if hr0 is None else float(hr0)
    wave, _ = _ppg_waveform(n, fs, hr0, rng)
    amp = rng.uniform(0.006, 0.012) * contrast * float(sc.get("amp_scale", 1.0))   # relative intensity modulation of green (amp_scale<1: weaker, real-video-like pulse)
    sigma = rng.uniform(*sc["noise_sigma_range"])                # 8-bit sensor noise (counts)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy, ax_, ay = size / 2, size / 2, size * 0.36, size * 0.46
    ellipse = (((xx - cx) / ax_) ** 2 + ((yy - cy) / ay) ** 2) <= 1.0
    inner = (((xx - cx) / (ax_ * 0.8)) ** 2 + ((yy - cy) / (ay * 0.8)) ** 2) <= 1.0
    shade = 1.0 - 0.25 * (((xx - cx * 0.8) / size) ** 2 + ((yy - cy * 0.8) / size) ** 2) ** 0.5 * 2
    tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.5)
    tex = 1.0 + 0.04 * tex / (tex.std() + 1e-9)
    base = np.array(albedo, np.float32)[None, None, :] * (shade * tex)[:, :, None]
    bg = np.full((size, size, 3), 60.0, np.float32) * (1 + 0.1 * cv2.GaussianBlur(
        rng.normal(0, 1, (size, size, 3)).astype(np.float32), (0, 0), 4))
    scene = np.where(ellipse[:, :, None], base, bg)
    drift = 1 + 0.015 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * np.arange(n) / fs + rng.uniform(0, 6.28))
    fl_amp, fl_hz = float(sc.get("flicker_amp", 0.0)), float(sc.get("flicker_hz", 10.0))
    if fl_amp > 0:                                                  # aliased mains flicker (e.g. 100 Hz lighting sampled at 30 fps -> 10 Hz)
        drift = drift * (1 + fl_amp * np.sin(2 * np.pi * fl_hz * np.arange(n) / fs + rng.uniform(0, 6.28)))
    walk = np.cumsum(rng.normal(0, 0.03, (n, 2)), axis=0)       # slow sub-pixel head motion (px)
    walk = np.clip(walk - walk.mean(0), -0.8, 0.8)
    frames = np.empty((n, size, size, 3), np.uint8)
    for i in range(n):
        mod = 1.0 - amp * wave[i] * PULSE_SIGNATURE            # more blood -> less reflected light
        img = scene.copy()
        img[ellipse] = img[ellipse] * mod[None, :]
        m = np.float32([[1, 0, walk[i, 0]], [0, 1, walk[i, 1]]])
        img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = img * drift[i] + rng.normal(0, sigma, img.shape)
        frames[i] = np.clip(np.round(img), 0, 255).astype(np.uint8)
    rgb = frames[:, inner].astype(np.float64).mean(axis=1)
    delay = int(round(0.1 * fs))                                 # finger PPG lags the face by ~100 ms
    ppg = np.roll(wave, delay) + rng.normal(0, 0.05, n)
    return Clip(name=f"synthetic:{clip_id}", subject=f"syn{clip_id:03d}", fs=fs, ppg=ppg, rgb=rgb,
                face=frames, face_mask=inner.astype(np.uint8),
                meta={"hr0_bpm": float(hr0), "tone": tone, "amp_green": float(amp), "noise_sigma": float(sigma),
                      "generator_seed": int(seed), "source": "synthetic"})


# ----------------------------------------------------------------------------- real-video face extraction
def extract_face_video(vid: Path, cfg: dict, crop: int, start_frame: int = 0, n_frames: int | None = None,
                       detect_every: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode ``vid`` once and return (skin-ROI mean RGB (T,3), square face crops (T,crop,crop,3) uint8, skin mask (crop,crop), fps).

    Face landmarks are detected every ``detect_every`` frames; the square crop box is linearly interpolated between detections
    (no per-detection jumps, which would create artificial events), and frames without a face reuse the nearest detection.
    The RGB trace uses the most recent skin mask in full-resolution coordinates.
    """
    from .capture import LandmarkSkinROI, masked_mean_rgb, video_frames
    detect_every = detect_every or int(cfg["face"]["crop_detect_every"])
    frames, fs, _ = video_frames(vid)
    roi = LandmarkSkinROI(cfg)
    roi.every = 1                                                         # we control the detection schedule here
    kept: list[np.ndarray] = []
    boxes: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    for i, fr in enumerate(frames):
        if i < start_frame:
            continue
        if n_frames is not None and i >= start_frame + n_frames:
            break
        j = i - start_frame
        kept.append(fr)
        if j % detect_every == 0:
            pts = roi.landmarks(fr)
            if pts is not None:
                m = roi.mask_from_landmarks(pts, fr.shape[:2])
                ys, xs = np.nonzero(m)
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                side = 1.1 * max(xs.max() - xs.min(), ys.max() - ys.min())
                boxes[j] = np.array([cx, cy, side])
                masks[j] = m
    roi.close()
    if not boxes:
        raise RuntimeError(f"no face found in {vid}")
    det = np.array(sorted(boxes))
    n = len(kept)
    interp = np.stack([np.interp(np.arange(n), det, [boxes[d][k] for d in det]) for k in range(3)], axis=1)
    h, w = kept[0].shape[:2]
    last_det = det[np.clip(np.searchsorted(det, np.arange(n), side="right") - 1, 0, None)]   # latest detection <= frame
    rgb, faces = [], []
    for j, fr in enumerate(kept):
        cx, cy, side = interp[j]
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        s = int(round(side))
        x0, y0 = min(max(x0, 0), max(w - s, 0)), min(max(y0, 0), max(h - s, 0))
        faces.append(cv2.resize(fr[y0:y0 + s, x0:x0 + s], (crop, crop), interpolation=cv2.INTER_AREA))
        rgb.append(masked_mean_rgb(fr, masks[last_det[j]]))
    mid = det[len(det) // 2]
    cx, cy, side = boxes[mid]
    s = int(round(side))
    x0, y0 = min(max(int(round(cx - side / 2)), 0), max(w - s, 0)), min(max(int(round(cy - side / 2)), 0), max(h - s, 0))
    mask = cv2.resize(masks[mid][y0:y0 + s, x0:x0 + s], (crop, crop), interpolation=cv2.INTER_NEAREST)
    return np.asarray(rgb), np.asarray(faces), mask, float(fs)


# ----------------------------------------------------------------------------- MCD-rPPG (real humans, CC BY 4.0)
def mcd_subjects(cfg: dict) -> list[str]:
    """Patient IDs for which both states' frontal-webcam video and synchronised PPG are on disk."""
    root = repo_path(cfg["datasets"]["mcd_root"])
    ids = []
    for p in sorted((root / "video").glob("*_FullHDwebcam_before.avi")):
        pid = p.name.split("_")[0]
        if all((root / "video" / f"{pid}_FullHDwebcam_{s}.avi").exists() and (root / "ppg_sync" / f"{pid}_FullHDwebcam_{s}.txt").exists()
               for s in ("before", "after")):
            ids.append(pid)
    return sorted(ids, key=int)


def mcd_effective_fs(meta_path: Path, n_frames: int) -> float:
    """Effective frame rate = decoded frames / wall-clock duration from the per-frame capture timestamps (dropped frames included)."""
    import datetime as dt
    lines = [ln.split(None, 1)[1].strip() for ln in meta_path.read_text().splitlines() if ln.strip()]
    dur = (dt.datetime.fromisoformat(lines[-1]) - dt.datetime.fromisoformat(lines[0])).total_seconds()
    return n_frames / dur


def load_mcd_clip(clip_id: str, cfg: dict, crop: int | None = None) -> Clip:
    """``clip_id`` = ``<patient>_<before|after>``; a ``seconds``-long excerpt starting ``start_s`` into the video.

    ``before`` = rest, ``after`` = post-exercise (higher HR, more motion). The contact-PPG (``ppg_sync``) has one sample per decoded frame.
    """
    d = cfg["datasets"]["mcd"]
    crop = crop or int(cfg["dvs"]["crop_size"])
    root = repo_path(cfg["datasets"]["mcd_root"])
    pid, step = clip_id.split("_")
    stem = f"{pid}_FullHDwebcam_{step}"
    start, n = int(d["start_s"] * 29.9), int(d["seconds"] * 29.9)
    cache = repo_path(cfg["datasets"]["cache_dir"]) / f"mcd_{clip_id}_c{crop}_s{start}_n{n}.npz"
    if cache.exists():
        z = np.load(cache)
        rgb, face, mask, fs = z["rgb"], z["face"], z["mask"], float(z["fs"])
    else:
        rgb, face, mask, fs = extract_face_video(root / "video" / f"{stem}.avi", cfg, crop, start, n)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, rgb=rgb, face=face, mask=mask, fs=fs)
    ppg_all = np.loadtxt(root / "ppg_sync" / f"{stem}.txt")[:, 0]
    fs = mcd_effective_fs(root / "meta" / f"{stem}.txt", len(ppg_all))     # container fps headers are unreliable (24 vs ~30)
    ppg = ppg_all[start:start + len(rgb)].astype(np.float64)
    k = min(len(ppg), len(rgb))
    return Clip(name=f"mcd:{clip_id}", subject=pid, fs=fs, ppg=ppg[:k], rgb=rgb[:k], face=face[:k], face_mask=mask,
                meta={"source": "mcd-rppg", "state": step, "camera": "FullHDwebcam(frontal)"})


# ----------------------------------------------------------------------------- UBFC-rPPG
def parse_ubfc_gt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UBFC DATASET_2 ground_truth.txt: line 1 PPG waveform, line 2 HR (BPM), line 3 timestamps (s)."""
    rows = [np.array(line.split(), dtype=np.float64) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError(f"{path}: expected 3 rows, got {len(rows)}")
    return rows[0], rows[1], rows[2]


def ubfc_subjects(cfg: dict) -> list[str]:
    """Subject folders (``subject<N>``) that have a ground_truth.txt, numerically sorted."""
    root = repo_path(cfg["datasets"]["ubfc_root"])
    subs = [p.name for p in root.glob("subject*") if (p / "ground_truth.txt").exists()]
    return sorted(subs, key=lambda s: int(s.replace("subject", "")))


def load_ubfc_clip(subject: str, cfg: dict, crop: int | None = None) -> Clip:
    """Load one UBFC clip; ROI trace and face crop are cached in ``data/cache``."""
    from .capture import LandmarkSkinROI, masked_mean_rgb, video_frames
    root = repo_path(cfg["datasets"]["ubfc_root"]) / subject
    vid = root / "vid.avi"
    if not vid.exists():
        raise FileNotFoundError(f"{vid} missing (Google Drive quota?). See data/README.md; scripts/download_ubfc.py")
    crop = crop or int(cfg["dvs"]["crop_size"])
    cache = repo_path(cfg["datasets"]["cache_dir"]) / f"ubfc_{subject}_c{crop}.npz"
    ppg_raw, _, t_gt = parse_ubfc_gt(root / "ground_truth.txt")
    if cache.exists():
        z = np.load(cache)
        rgb, face, mask, fs = z["rgb"], z["face"], z["mask"], float(z["fs"])
    else:
        rgb, face, mask, fs = extract_face_video(vid, cfg, crop)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, rgb=rgb, face=face, mask=mask, fs=fs)
    t_frames = np.arange(len(rgb)) / fs
    ppg = np.interp(t_frames, t_gt - t_gt[0], ppg_raw)
    return Clip(name=f"ubfc:{subject}", subject=subject, fs=fs, ppg=ppg, rgb=rgb, face=face, face_mask=mask,
                meta={"source": "ubfc-rppg", "gt_len": len(ppg_raw)})


# ----------------------------------------------------------------------------- PURE / SCAMPS (not exercised: no data yet)
def load_pure_clip(run: str, cfg: dict) -> Clip:
    """PURE run ``<id>-<setup>``: PNG frames + JSON with /FullPackage (PPG @60Hz) and /Image timestamps."""
    from .capture import LandmarkSkinROI, masked_mean_rgb
    root = repo_path(cfg["datasets"]["pure_root"]) / run
    meta = json.loads((root / f"{run}.json").read_text())
    pngs = sorted((root / run).glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no PNG frames under {root / run}")
    t_img = np.array([e["Timestamp"] for e in meta["/Image"]], dtype=np.float64) * 1e-9
    t_ppg = np.array([e["Timestamp"] for e in meta["/FullPackage"]], dtype=np.float64) * 1e-9
    wave = np.array([e["Value"]["waveform"] for e in meta["/FullPackage"]], dtype=np.float64)
    ppg = np.interp(t_img[:len(pngs)], t_ppg, wave)
    roi = LandmarkSkinROI(cfg)
    rgb = []
    for p in pngs:
        fr = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = roi.mask(fr)
        rgb.append(masked_mean_rgb(fr, m) if m is not None else np.full(3, np.nan))
    roi.close()
    fs = 1.0 / np.median(np.diff(t_img))
    return Clip(name=f"pure:{run}", subject=run.split("-")[0], fs=float(fs), ppg=ppg, rgb=np.asarray(rgb),
                meta={"source": "pure"})


def load_scamps_clip(path: Path, cfg: dict, crop: int | None = None) -> Clip:
    """SCAMPS ``.mat`` (HDF5) clip: ``Xsub`` stored as (c,w,h,t) floats in [0,1], ``d_ppg`` (t,1), ``skin_mask`` (w,h,t-1)."""
    import h5py
    crop = crop or int(cfg["dvs"]["crop_size"])
    with h5py.File(path, "r") as h:
        x = np.asarray(h["Xsub"]).transpose(3, 2, 1, 0)          # -> (t,h,w,c)
        ppg = np.asarray(h["d_ppg"]).reshape(-1).astype(np.float64)
        skin = np.asarray(h["skin_mask"]).transpose(2, 1, 0).mean(axis=0) > 0.5   # (h,w)
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    rgb = x[:, skin].astype(np.float64).mean(axis=1)
    face = np.stack([cv2.resize(f, (crop, crop), interpolation=cv2.INTER_AREA) for f in x])
    mask = cv2.resize(skin.astype(np.uint8), (crop, crop), interpolation=cv2.INTER_NEAREST)
    return Clip(name=f"scamps:{path.stem}", subject=path.stem, fs=30.0, ppg=ppg[:len(x)], rgb=rgb, face=face,
                face_mask=mask, meta={"source": "scamps"})

def load_clip(spec: str, cfg: dict) -> Clip:
    """Load ``synthetic:<id>``, ``ubfc:subject<N>``, ``pure:<run>`` or ``scamps:<path>``."""
    kind, _, ident = spec.partition(":")
    if kind == "synthetic":
        return make_synthetic_clip(int(ident), cfg)
    if kind == "ubfc":
        return load_ubfc_clip(ident, cfg)
    if kind == "mcd":
        return load_mcd_clip(ident, cfg)
    if kind == "pure":
        return load_pure_clip(ident, cfg)
    if kind == "scamps":
        return load_scamps_clip(repo_path(cfg["datasets"]["scamps_root"]) / "scamps_videos_example" / f"{ident}.mat", cfg)
    raise ValueError(f"unknown clip spec {spec!r}")


# ----------------------------------------------------------------------------- splits, checksums, manifest
def subject_split(subjects: list[str], fracs: list[float], seed: int) -> dict[str, list[str]]:
    """Seeded subject-disjoint train/val/test split."""
    subs = sorted(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
    return {"train": sorted(subs[:n_tr]), "val": sorted(subs[n_tr:n_tr + n_va]), "test": sorted(subs[n_tr + n_va:])}


STATES = ("before", "after")


def clip_specs(dataset: str, split: str, cfg: dict, limit: int | None = None) -> list[str]:
    """Clip specs (``dataset:id``) of a subject-disjoint split; ``split='all'`` uses every subject.

    MCD-rPPG has two clips (rest/post-exercise) per subject; both always land in the same split.
    """
    subs = dataset_subjects(dataset, cfg)
    chosen = subs if split == "all" else subject_split(subs, cfg["datasets"]["split_fracs"], cfg["seed"])[split]
    if dataset == "synthetic":
        specs = [f"synthetic:{int(s.replace('syn', ''))}" for s in chosen]
    elif dataset == "mcd":
        specs = [f"mcd:{s}_{st}" for s in chosen for st in STATES]
    else:
        specs = [f"{dataset}:{s}" for s in chosen]
    return specs[:limit] if limit else specs


def subject_of(spec: str) -> str:
    """Subject identifier of a clip spec (clips of one subject are pooled before statistics)."""
    kind, _, ident = spec.partition(":")
    return f"{kind}:{ident.split('_')[0]}" if kind == "mcd" else spec


def state_of(spec: str) -> str:
    """Recording state (``before`` = rest, ``after`` = post-exercise) for MCD clips, else ``n/a``."""
    return spec.rpartition("_")[2] if spec.startswith("mcd:") else "n/a"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dataset_subjects(name: str, cfg: dict) -> list[str]:
    """Subject IDs available for a dataset name."""
    if name == "ubfc":
        return ubfc_subjects(cfg)
    if name == "synthetic":
        return [f"syn{i:03d}" for i in range(int(cfg["datasets"]["synthetic"]["n_subjects"]))]
    if name == "mcd":
        return mcd_subjects(cfg)
    if name == "pure":
        root = repo_path(cfg["datasets"]["pure_root"])
        return sorted({p.name.split("-")[0] for p in root.glob("*-*") if p.is_dir()})
    if name == "scamps":
        return sorted(p.stem for p in repo_path(cfg["datasets"]["scamps_root"]).rglob("P*.mat"))
    raise ValueError(f"no subject listing for {name!r}")


def build_manifest(name: str, cfg: dict, with_checksums: bool = True) -> dict:
    """Split manifest: exact subject IDs per split plus file checksums and availability."""
    subjects = dataset_subjects(name, cfg)
    if not subjects:
        raise SystemExit(f"no subjects found for dataset {name!r}")
    split = subject_split(subjects, cfg["datasets"]["split_fracs"], cfg["seed"])
    files: dict[str, dict] = {}
    if name == "ubfc" and with_checksums:
        root = repo_path(cfg["datasets"]["ubfc_root"])
        for s in subjects:
            for fn in ("ground_truth.txt", "vid.avi"):
                p = root / s / fn
                files[f"{s}/{fn}"] = ({"sha256": sha256_file(p), "bytes": p.stat().st_size} if p.exists()
                                      else {"missing": True})
    if name == "mcd":
        ck = repo_path(cfg["datasets"]["splits_dir"]) / "mcd_checksums.json"
        if ck.exists():
            files = {k: {"sha256": v} for k, v in json.loads(ck.read_text())["sha256"].items()}
    return {"dataset": name, "seed": cfg["seed"], "fracs": cfg["datasets"]["split_fracs"], "n_subjects": len(subjects),
            "split": split, "files": files, "created": timestamp(),
            "subject_disjoint": not (set(split["train"]) & set(split["test"]) or set(split["train"]) & set(split["val"])
                                     or set(split["val"]) & set(split["test"]))}


def _preprocess_one(arg: tuple) -> str:
    """Worker: build (and cache) the face crops / RGB traces of one clip."""
    spec, cfg = arg
    load_clip(spec, cfg)
    return spec


def main() -> None:
    """CLI: ``manifest`` emits a subject-disjoint split manifest; ``synth`` renders a fallback clip summary."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("command", choices=["manifest", "synth", "preprocess"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sets", nargs="*", default=None, help="preprocess: dataset:split specs (default: every clip of --dataset)")
    ap.add_argument("--dataset", default="ubfc", choices=["ubfc", "pure", "synthetic", "scamps", "mcd"])
    ap.add_argument("--no-checksums", action="store_true")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    if args.command == "preprocess":
        from concurrent.futures import ProcessPoolExecutor
        specs = [s for it in (args.sets or [f"{args.dataset}:all"]) for s in clip_specs(*it.split(":"), cfg)]
        log.info("preprocessing %d clips with %d workers", len(specs), args.workers)
        with ProcessPoolExecutor(args.workers) as ex:
            for i, s in enumerate(ex.map(_preprocess_one, [(s, cfg) for s in specs])):
                if i % 10 == 0:
                    log.info("  %d/%d %s", i + 1, len(specs), s)
        return
    if args.command == "manifest":
        man = build_manifest(args.dataset, cfg, with_checksums=not args.no_checksums)
        path = save_json(man, repo_path(cfg["datasets"]["splits_dir"]) / f"{args.dataset}_seed{cfg['seed']}.json")
        log.info("wrote %s", path)
        log.info("subjects=%d train=%d val=%d test=%d disjoint=%s", man["n_subjects"], len(man["split"]["train"]),
                 len(man["split"]["val"]), len(man["split"]["test"]), man["subject_disjoint"])
        log.info("test subjects: %s", man["split"]["test"])
        missing = [k for k, v in man["files"].items() if v.get("missing")]
        if missing:
            log.warning("%d listed files are not on disk (e.g. %s)", len(missing), missing[:3])
    else:
        clip = make_synthetic_clip(0, cfg)
        log.info("synthetic clip: %d frames, hr0=%.1f bpm, tone=%s, noise=%.2f", len(clip.rgb), clip.meta["hr0_bpm"],
                 clip.meta["tone"], clip.meta["noise_sigma"])


if __name__ == "__main__":
    main()
