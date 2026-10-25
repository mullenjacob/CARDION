# CARDION

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23647425.svg)](https://doi.org/10.5281/zenodo.23647425)

Repository: https://github.com/mullenjacob/CARDION  
Archived release v1.0.0: https://doi.org/10.5281/zenodo.23647425

**Does event emulation preserve the pulse?** Code, trained models, saved results and paper for a subject-disjoint study of heart-rate estimation from real webcam video with frame-based RGB estimators, **simulated** event-camera processing, a spiking neural network (SNN) and a matched ReLU network.

> Research prototype. **Not a medical device**; no clinical validation. Events are *emulated* from ordinary video - **no event-camera hardware is used**, and nothing here is evidence about physical event sensors. The paper is a preprint (not peer reviewed).

Paper: Jacob Mullen, *Does Event Emulation Preserve the Pulse? Frame-Based, Simulated-Event and Spiking Heart-Rate Estimators on Real Webcam Video* - [`paper/main.pdf`](paper/main.pdf).
How to cite: see [`CITATION.cff`](CITATION.cff). Licences: see [`LICENSING.md`](LICENSING.md).

## What CARDION investigates

1. **RQ1** - On unseen subjects, do estimators that use events emulated from compressed webcam video (classical, ANN, SNN) match frame-based RGB estimators (green, CHROM, POS)?
2. **RQ2** - Does an SNN offer an accuracy or operation-count advantage over a matched ANN and over classical event processing on such streams?
3. **RQ3** - How much of the accuracy of a *displayed* heart rate depends on the reliability gate that decides when to display it, and which estimator produces the displayed number?

## Main findings (MCD-rPPG, 36/12/12 subject-disjoint train/validation/test subjects)

| result (held-out test subjects, subject-level MAE) | value |
|---|---|
| RGB POS | 5.78 BPM (95 % bootstrap CI 2.5-10.1) |
| classical event POS (validation-selected threshold 0.02) | 16.07 BPM |
| ANN on emulated events (threshold 0.05) | 12.82 BPM |
| SNN on emulated events (threshold 0.1) | 18.89 BPM |
| reliability-gated RGB consensus | 2.37 BPM, **but a number is shown for only 41 % of windows** (3-72 % across subjects) |

* RGB POS was clearly more accurate than every emulated-event estimator we built; the SNN showed no accuracy advantage over the matched ANN and needs more synaptic operations than the dense ANN at fine thresholds. These are observed accuracy differences for these estimators, one dataset (12 test subjects), one small architecture and an operation-count energy *proxy* (not measured power); they do not prove that the pulse is absent from emulated events.
* Most of the gated number's accuracy comes from **selection**: on exactly the windows the gate shows, POS alone has 1.46 BPM MAE. Accuracy of displayed readings must be read together with availability.
* The number shown by the application is a **classical RGB consensus** (green, POS, CHROM that agree); the SNN, ANN and event estimators run alongside for comparison and never determine it.
* Test-set history, exploratory versus confirmatory analyses, and all limitations are in the paper (Sections V and IX).

## Conventional RGB versus simulated-event processing

*RGB path:* face landmarks -> skin-region mean colour per frame -> green / CHROM / POS pulse -> band-pass -> spectral peak (sub-harmonic summation) over 10 s windows.
*Event path:* the same face crop -> analytic DVS emulator per colour channel (log-intensity threshold, mismatch, refractory period, leak, shot noise; cross-checked with v2e) -> 33.3 ms event frames -> classical event estimators, or an 18 k-parameter SNN (snnTorch, leaky integrate-and-fire) or identical ReLU network -> pulse waveform -> the same heart-rate picker.

## Install and run (Windows)

1. Download or clone this repository to a short path (for example `C:\CARDION`).
2. Double-click **`Launch CARDION.bat`**. The first run creates a private Python environment (about 1.5 GB, internet needed once; PyTorch, MediaPipe, snnTorch and others are downloaded from their official indexes) and downloads the MediaPipe face-landmarker model (not included in this repository; see `REPRODUCE.md`). Your browser opens the local interface (`http://127.0.0.1:8765/`, reachable only from your computer).
3. Choose **Live camera**, **Video file** or **Built-in demo** (a simulated face with a known pulse; works without a camera), then **Start**. **Validation mode** helps you compare readings with a reference monitor. Video is processed on your computer and is not uploaded.
4. `Run CARDION Self-Test.bat` runs the automated self-test; `Test and Repair CARDION.bat` diagnoses installation problems.

Other platforms: the research code is plain Python (3.10+, tested with 3.11); install `requirements.txt` in a virtual environment. The launcher and the browser interface are Windows-oriented and untested elsewhere.

## Reproduce the paper

*Fast checks (minutes; no dataset needed):*

```
python -m pytest -q tests                      # unit and integration tests (tests needing local data/camera clips skip or are marked)
python scripts/check_claims.py                 # every number used in the paper resolves to a saved output file
python scripts/paper_analysis.py               # independent recomputation of the reported statistics from the saved window-level results (no dataset needed)
python scripts/reconcile_windows.py            # reconciles dataset-table and evaluation window counts (needs the downloaded MCD-rPPG PPG/meta files; otherwise exits with an explanation)
python scripts/test_history_audit.py           # documents the test-split history (maintainer-only: needs the original development git history, which this repository does not contain, and then exits with an explanation; the recorded evidence is in `docs/TEST_SPLIT_HISTORY.md`, see also paper Section V)
```

*Maintainer-only scripts* (they need the author's development history, local tools or accounts and are not part of reproducing the results): `scripts/test_history_audit.py` (exits with an explanation when the history is absent), `scripts/make_release.py`, `scripts/verify_release.py` and `scripts/finalize_availability.py` (publication packaging), `scripts/fetch_bib.py` and `scripts/verify_refs.py` (need internet), `scripts/profile_live.py`, `scripts/bench_threads.py` and `scripts/fps_effect.py` (timing on one particular PC).

*Rebuild the manuscript* (compiling needs a LaTeX distribution with `IEEEtran`; without one the script still regenerates `paper/main.tex` and says so): `python scripts/build_paper.py` regenerates `paper/main.tex` from `paper/main.tex.in` and `paper/claims.json` (it refuses to build if a number is not registered) and compiles `paper/main.pdf`.

*Re-run the experiments* (about 8-12 h on a laptop CPU; downloads about 7 GB): follow [`REPRODUCE.md`](REPRODUCE.md), which also separates the application, the demo, the tests, the saved statistics and the manuscript build (the first install step is `python -m venv .venv` and `pip install -r requirements.txt`). Previously saved results used by the paper are in `out/`; `models/` holds the trained checkpoints; `data/splits/` holds the subject-disjoint split manifest and file checksums.

## Datasets and third-party material (not redistributed here)

The main experiments use **MCD-rPPG** only (real webcam video, 36/12/12 subject-disjoint train/validation/test subjects, with events *simulated* from the compressed video). SCAMPS (10-video example) and a synthetic generator serve only for diagnostics and tests; UBFC-rPPG and PURE were considered but not obtained and are not used. Details, layout, and which commands need the downloaded data: [`data/README.md`](data/README.md).

| material | source | licence / terms |
|---|---|---|
| MCD-rPPG videos and PPG | Hugging Face `kyegorov/mcd_rppg`; Egorov et al., ACM MM 2025, doi:10.1145/3746027.3758255 | CC BY 4.0 (official dataset page, checked 2026-10-25); `python scripts/download_mcd.py` |
| SCAMPS 10-video example (diagnostics only) | https://github.com/danmcduff/scampsdataset | see upstream |
| MediaPipe face-landmarker model | downloaded automatically from Google | see upstream model card |
| rPPG-Toolbox reference subset (anchor check only) | https://github.com/ubicomplab/rPPG-Toolbox via `scripts/fetch_third_party.py` | Responsible AI Source Code License v1.1 - read it before use |
| v2e (optional cross-check) | https://github.com/SensorsINI/v2e | MIT |

Trained checkpoints, split manifests and saved results derived from MCD-rPPG are shared with attribution (see `LICENSING.md`). The repository contains no video, face image or recording of any person, no recordings made with the application, and no credentials. It does contain the 60 MCD-rPPG subjects' public dataset IDs and results indexed by them and by recording (per-subject and per-window estimates), which are derived from that publicly available dataset; do not use them to try to identify anyone.

## Important limitations

Simulated events only; one dataset and 12 test subjects (wide intervals; non-significant differences are not equivalence); compressed MPEG-4 recordings with frame rates from 18 to 30 fps; no skin-tone, glasses, motion or low-light analysis; small networks and short training; energy numbers are operation-count proxies; the reliability-gate comparison and the single-method gate curve are exploratory; the test split was scored twice because of a reporting bug (documented, no model or rule changed) and re-read for later descriptive analyses; no accuracy study with the live webcam has been done (the live application was exercised with recorded faces, a built-in simulated face and a camera with nobody in view); informal comparisons with consumer wearables are anecdotes, not evidence. See the paper for details.

## AI assistance

Much of the software, experiment execution, analysis, literature search and first drafting was done with an AI coding assistant (Claude, Anthropic) under the author's direction; the paper's Declarations section describes this. The author is responsible for the content.

## Repository map

`src/` research code (`dvs_emulate.py`, `events.py`, `rppg_classical.py`, `rppg_events.py`, `snn_model.py`, `ann_model.py`, `train.py`, `evaluate.py`, `stats.py`, `efficiency.py`) and the application (`src/cardion_app/`) - `scripts/` experiments, audits and paper build - `configs/` - `models/` - `data/splits/` - `out/` saved results - `paper/` manuscript sources - `tests/` - `docs/` - `specs/`.
