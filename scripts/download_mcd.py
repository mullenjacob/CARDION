"""Download a seeded, subject-level subset of MCD-rPPG (Hugging Face ``kyegorov/mcd_rppg``, CC BY 4.0).

Real human faces, frontal consumer webcam (640x480, 30 fps, ~3 min), contact-PPG synchronised to every frame, two states per
subject (rest = ``before``, post-exercise = ``after``). Only the ``FullHDwebcam`` (frontal view) videos are fetched.
Files are checked against the sizes reported by the Hub API; SHA-256 of every file is written to
``data/splits/mcd_checksums.json``. Resumable. Dataset terms: cite Egorov et al. 2025 (ACM MM, arXiv:2508.17924); no re-identification.

    python scripts/download_mcd.py --n-subjects 60 --seed 1337
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/kyegorov/mcd_rppg"
CAMERA = "FullHDwebcam"
UA = {"User-Agent": "cardion-research/1.0"}


def api_sizes() -> dict[str, int]:
    """File sizes (bytes) from the Hub API, used as an integrity check."""
    with urllib.request.urlopen(urllib.request.Request("https://huggingface.co/api/datasets/kyegorov/mcd_rppg?blobs=true", headers=UA), timeout=120) as r:
        return {s["rfilename"]: s.get("size", 0) for s in json.load(r)["siblings"]}


def fetch(rel: str, dest: Path, size: int, tries: int = 10) -> str:
    """Download ``rel`` to ``dest`` (skip if present with correct size); returns sha256. Uses a keep-alive session and backoff."""
    import requests
    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        return hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(f"{BASE}/resolve/main/{rel}", headers=UA, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if size and tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {rel}: {tmp.stat().st_size} != {size}")
            tmp.replace(dest)
            return hashlib.sha256(dest.read_bytes()).hexdigest()
        except Exception as e:  # noqa: BLE001 - retry any network error
            print(f"retry {k + 1}/{tries} {rel}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 2 ** (k + 1)))
    raise SystemExit(f"failed to download {rel}")

def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-subjects", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/raw/mcd_rppg")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    sizes = api_sizes()
    fetch("db.csv", out / "db.csv", sizes.get("db.csv", 0))
    rows = list(csv.DictReader(open(out / "db.csv", encoding="utf-8")))
    by_pid: dict[str, set[str]] = {}
    for r in rows:
        if r["camera"] == CAMERA:
            by_pid.setdefault(r["patient_id"], set()).add(r["step"])
    pids = sorted(p for p, s in by_pid.items() if {"before", "after"} <= s)
    random.Random(a.seed).shuffle(pids)
    chosen = sorted(pids[:a.n_subjects], key=int)
    print(f"{len(pids)} eligible subjects; downloading {len(chosen)}: {chosen}", flush=True)
    jobs = []
    for pid in chosen:
        for step in ("before", "after"):
            stem = f"{pid}_{CAMERA}_{step}"
            for rel in (f"video/{stem}.avi", f"ppg_sync/{stem}.txt", f"meta/{stem}.txt"):
                jobs.append(rel)
    total = sum(sizes.get(j, 0) for j in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB", flush=True)
    sums: dict[str, str] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (rel, h) in enumerate(zip(jobs, ex.map(lambda j: fetch(j, out / j, sizes.get(j, 0)), jobs))):
            sums[rel] = h
            if i % 12 == 0:
                print(f"{i + 1}/{len(jobs)} {rel}", flush=True)
    manifest = {"source": BASE, "license": "CC BY 4.0", "camera": CAMERA, "seed": a.seed, "subjects": chosen,
                "sha256": sums, "sizes": {j: sizes.get(j, 0) for j in jobs}}
    dest = ROOT / "data" / "splits" / "mcd_checksums.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=1))
    print("done; wrote", dest)


if __name__ == "__main__":
    sys.exit(main())
