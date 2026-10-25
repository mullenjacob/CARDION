# Data

Which data the study uses, where to get it, and what is (not) in this repository.

| dataset | role in the study | in this repository? |
|---|---|---|
| **MCD-rPPG** (real humans, frontal webcam) | **the only real-human dataset of the main experiments**: training, validation and the protected test split, all reported accuracy | no (download it, see below); only the split manifest and file checksums |
| SCAMPS 10-video example (rendered faces) | diagnostics and the correctness anchor against rPPG-Toolbox's POS; **not** part of the reported real-human evaluation | no |
| Synthetic face generator (`src/datasets.py::make_synthetic_clip`) | plumbing tests, unit tests and the controlled noise/flicker study; **not** a benchmark | code only |
| UBFC-rPPG | considered as the primary dataset at the start; **not used**: the videos could not be downloaded (Google Drive quota) | no |
| PURE | considered; **not used**: access requires a request to the authors, never obtained | no |
| MMPD | considered; not used | no |
| Author's own webcam clips (`src/record.py`) | none in the paper; the code can record clips for qualitative use | no |

The paper's statements about this are in the *Experimental Setup* and *Limitations* sections.

## MCD-rPPG (main dataset)

* Source: Hugging Face `kyegorov/mcd_rppg` - https://huggingface.co/datasets/kyegorov/mcd_rppg. Egorov et al., ACM Multimedia 2025, doi:10.1145/3746027.3758255. The official dataset page lists the licence as **CC BY 4.0** (checked 2026-10-25; re-check before relying on it). Attribution is required; the dataset contains real people's faces, so do not redistribute the videos or attempt re-identification.
* What the study used: a seeded random subset of **60 subjects** (seed 1337), frontal consumer webcam (`FullHDwebcam`, 640x480 MPEG-4 in an `.avi` container, effective 18-30 fps), one **rest** (`before`) and one **post-exercise** (`after`) recording per subject, with finger PPG synchronised to every decoded frame.
* Split: **subject-disjoint 36 training / 12 validation / 12 test subjects**; both recordings of a subject stay in the same split. The exact subject IDs are in `data/splits/mcd_seed1337.json`. The test split is evaluated through a code guard (`--final`); its scoring history is documented in `docs/TEST_SPLIT_HISTORY.md`.
* Download (about 7 GB, resumable; sizes are checked against the Hub API):

```
python scripts/download_mcd.py --n-subjects 60 --seed 1337
```

  Layout produced under `data/raw/mcd_rppg/`: `video/<id>_FullHDwebcam_{before,after}.avi`, `ppg_sync/<id>_FullHDwebcam_{before,after}.txt` (PPG, one row per frame) and `meta/<id>_FullHDwebcam_{before,after}.txt` (frame time stamps, used to compute the effective frame rate).
* Checksums: `data/splits/mcd_checksums.json` holds the SHA-256 of every downloaded file. `python -m src.datasets manifest --dataset mcd` rewrites the split manifest from the files on disk (it reproduces the same split for the same subjects and seed).

### How it is processed

1. `python -m src.datasets preprocess --dataset mcd --workers 6`: MediaPipe face landmarks, a face crop and skin-region mean RGB per frame, cached under `data/cache/` (excerpt of 1794 frames, about 60 s, starting 5 s into each recording).
2. `python -m src.train precompute ...`: **simulated events** from the compressed video with the analytic DVS emulator at thresholds 0.02, 0.05 and 0.1 (cached in `data/events/`). No event camera is involved.
3. Heart rate from 10 s windows (1 s stride) for every method; the reference is the spectral peak of the finger PPG in the same window, with a validity rule. Details: paper, Sections IV-V.

## SCAMPS (diagnostics only)

https://github.com/danmcduff/scampsdataset. Only the **10-video example** (1.2 GB) was used (`scamps_videos_example.tar.gz` from `facesyntheticspubwedata.z6.web.core.windows.net/neurips-2022/`, SHA-256 `2EB3B3C915903F4BC0387A93D37E8711D378AF2E4209573495AB77DD6011A9C2`), extracted to `data/raw/scamps/scamps_videos_example/P0000NN.mat`. Terms: see the upstream repository. It is needed only for `scripts/anchor_toolbox.py`, `scripts/summarize_diag.py` and a few diagnostics; none of the reported real-human accuracy depends on it.

## Synthetic generator

Face-like clips with a known pulse (HR 55-120 BPM, skin-tone dependent amplitude, shading, texture, sub-pixel head motion, illumination drift, sensor noise). Used by the unit tests and by `scripts/noise_study.py` (controlled noise and flicker study). It is much easier than real video and is **not** a benchmark.

## UBFC-rPPG, PURE (not used)

`scripts/list_ubfc_drive.py`, `download_ubfc.py`, `retry_ubfc.py` and the loaders for UBFC-rPPG and PURE remain in the code from the early plan, but no UBFC video was obtained (only the ground-truth text files, which are not included here) and PURE was never obtained, so **no result in the paper uses them** and the PURE loader is untested on real data. UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg.

## What is deliberately not in the repository

Everything under `data/raw/`, `data/cache/` and `data/events/` (see `.gitignore`): the MCD-rPPG videos and PPG files, the SCAMPS example, derived caches, and any recording made with the application. The repository contains the split manifest and checksums for the 60 MCD-rPPG subjects (their public dataset IDs) and saved results indexed by those IDs and by recording (`out/`); it contains no video, face image, or private recording.

## Which commands need the downloaded data

| command | needs |
|---|---|
| `python -m pytest -q tests` | nothing (some tests skip without a face video, the MCD files, or a camera) |
| `python scripts/check_claims.py`, `python scripts/paper_analysis.py` | nothing: they use the saved results in `out/` |
| `python scripts/reconcile_windows.py`, `python scripts/dataset_stats.py` | MCD-rPPG PPG and meta files |
| `python -m src.datasets preprocess`, `src.train`, `scripts/eval_mcd.ps1`, `scripts/reproduce_all.ps1` | the MCD-rPPG videos (about 7 GB); `anchor_toolbox.py` also the SCAMPS example and `scripts/fetch_third_party.py` |
| `scripts/noise_study.py` | no download (uses the synthetic generator) |
