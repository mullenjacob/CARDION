# REPRODUCE

Six different things can be "reproduced". They need different things; do only what you need.

| goal | needs | time | section |
|---|---|---|---|
| Run the live application | Windows (launcher) or Python + webcam | minutes | 2 |
| Run the sample demonstration | nothing but the install (built-in simulated face) | minutes | 2 |
| Run the automated tests | the Python environment | 2-3 min | 3 |
| Reproduce the saved statistics and tables | the Python environment (`out/` is included) | minutes | 4 |
| Re-run the experiments from the original recordings | MCD-rPPG download (about 7 GB), 8-12 h CPU | hours | 6 |
| Rebuild the manuscript | a LaTeX distribution with `IEEEtran` | minutes | 5 |

All commands run from the repository root. CPU only; no GPU is needed. Seeds are fixed (`--seed`, default 1337; networks use `--train-seed`). Commands are given for Windows PowerShell; on Linux/macOS the research code is plain Python and the same `python -m ...` / `python scripts/...` commands apply with `.venv/bin/python` (the launchers and the `.ps1` scripts are Windows-only and untested elsewhere).

## 1. Install (standard Python and pip)

Python 3.10 or newer (tested with 3.11). Verified from a clean extraction of the released archive:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is the complete pinned research environment (PyTorch, snnTorch, MediaPipe, OpenCV, pytest, ...; a large download). `requirements-app.txt` is the smaller set the Windows launcher installs for the application only. [`uv`](https://docs.astral.sh/uv/) can replace `python -m venv` and `pip` if you already have it; it is optional and not needed.

### MediaPipe face model (downloaded separately)

`models/face_landmarker.task` is **not** in the repository: it is Google's MediaPipe model and its redistribution terms are not clear to us, so it is fetched from Google instead. `src/capture.py::ensure_model` downloads it **automatically on first use** (the application, the self-test, and any script that tracks a face) from the URL in `configs/default.yaml` (`face.landmarker_url`, `storage.googleapis.com/mediapipe-models/...`) and logs its SHA-256. If the download fails (no internet, a proxy), download that URL by hand and save the file as `models/face_landmarker.task`; nothing else is needed. The code does not compare the file with a pinned hash; the copy used for all experiments is 3,758,596 bytes with SHA-256 `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF`, so compare the logged value (or `Get-FileHash models\face_landmarker.task`) with it. Tests that need the model skip until it exists.

## 2. Run the application and the demonstration

* Windows, no setup: double-click `Launch CARDION.bat` (it builds its own private environment inside the folder, about 1.5 GB, internet needed once, and downloads the face model). See `docs/USER_GUIDE.md`.
* From the Python environment of section 1: `.venv\Scripts\python.exe -m src.cardion_app` opens the local interface on `http://127.0.0.1:8765/`. **Built-in demo** (a simulated face with a known pulse, no camera) is the sample demonstration; **Live camera** needs a webcam and **Video file** needs a recording of a face. `-m src.cardion_app --autostart demo --no-browser` starts the demo without opening a browser.
* `.venv\Scripts\python.exe -m src.cardion_app --selftest` runs the application self-test (report: `out\selftest_report.md`). Parts that need a recorded face video are skipped when none is present.

The application shows a number only when its reliability gate allows it; the number comes from the classical RGB estimators, not the SNN (paper Section VII). No accuracy study with the live webcam has been done.

## 3. Automated tests

```
.venv\Scripts\python.exe -m pytest -q tests
```

On the released archive in a fresh environment this gave **82 passed, 17 skipped, 0 failed**. The skips need a recorded face video, the downloaded MCD-rPPG files, recorded clips, or a slow throw-away environment (`CARDION_SLOW_TESTS=1`).

## 4. Reproduce the saved statistics (no dataset needed)

```
python scripts/check_claims.py       # every number used in the paper resolves to a saved output file (191/191)
python scripts/paper_analysis.py     # independent recomputation of the reported statistics from the saved window-level results in out/
```

`scripts/reconcile_windows.py` (window-count reconciliation) additionally needs the downloaded MCD-rPPG PPG and meta files and otherwise exits with an explanation. `scripts/test_history_audit.py` needs the author's development git history, which the public repository does not contain; it exits with an explanation, and the recorded evidence is in `docs/TEST_SPLIT_HISTORY.md`.

## 5. Rebuild the manuscript

```
python scripts/build_paper.py
```

regenerates `paper/main.tex` from `paper/main.tex.in` and `paper/claims.json` (it refuses to build if a number is not registered) and compiles `paper/main.pdf` when a LaTeX distribution with `IEEEtran` is found (MiKTeX or TeX Live with `pdflatex` and `bibtex` on the path; otherwise it says so and leaves `main.tex` ready to compile). On Windows `scripts\build_paper.ps1` does the same with the portable toolchain of the author's machine.

## 6. Re-run the experiments from the original recordings

Requires the MCD-rPPG recordings (`data/README.md`: Hugging Face `kyegorov/mcd_rppg`, CC BY 4.0, about 7 GB), the face model of section 1, and Windows PowerShell for the `.ps1` drivers. Expect roughly 8-12 hours on a laptop CPU; training (3 parallel worker processes) dominates.

Everything in one go:

```
powershell -File scripts\reproduce_all.ps1
```

which downloads MCD-rPPG (60 subjects; checksums in `data\splits\mcd_checksums.json`), fetches the third-party reference code (`scripts\fetch_third_party.py`: rPPG-Toolbox subset and v2e at pinned commits, each under its own licence), preprocesses the clips, emulates events at three thresholds, trains 18 SNN/ANN runs plus controls, evaluates on validation (selection, calibration), runs the **protected test evaluation** (`--final`; the original study scored the test split twice, see `docs/TEST_SPLIT_HISTORY.md`, so re-running it is a reproduction, not a fresh confirmatory test), runs the diagnostics (these also need the SCAMPS 10-video example, `data/README.md`), and builds the report and the paper. `reproduce_all.ps1` expects the environment of section 1 at `.venv\`.

Individual steps:

```
python scripts/download_mcd.py --n-subjects 60 --seed 1337          # data (resumable)
python -m src.datasets manifest --dataset mcd                       # frozen subject-disjoint split -> data/splits/mcd_seed1337.json
python -m src.datasets preprocess --dataset mcd --workers 6         # face tracking, crops, RGB traces (cached in data/cache)
python -m src.train precompute --sets mcd:train mcd:val mcd:test --thresholds 0.02 0.05 0.1 --workers 6
powershell -File scripts\run_mcd_training.ps1 -Thresholds 0.05 -Seeds 0,1,2 -Workers 3     # SNN + ANN per seed (model selection on mcd:val only)
powershell -File scripts\eval_mcd.ps1 -Split val -Tag val_all       # validation table, tests, strata
python scripts/calibrate_quality.py --mode calibrate --results "out/results_mcd-val_val_all_*.json"   # app reliability rule (validation only)
python scripts/export_app_models.py                                  # best-on-validation SNN/ANN -> models/
powershell -File scripts\eval_mcd.ps1 -Split test -Tag final -Final # protected test split (already scored; see docs/TEST_SPLIT_HISTORY.md)
python -m src.evaluate --report "out/results_mcd-test_final_*.json" # tables + figures
python scripts/build_paper.py                                        # paper/main.pdf (fails if any number is not in paper/claims.json)
```

Diagnostics: `scripts/summarize_diag.py`, `scripts/real_video_diagnostics.py`, `scripts/event_vs_intensity.py`, `scripts/noise_study.py`, `scripts/compression_study.py`, `scripts/summarize_budget.py`, `scripts/app_real_video_check.py`, `scripts/anchor_toolbox.py`. Bibliography checks (`scripts/fetch_bib.py`, `scripts/verify_refs.py`) need internet. Publication-packaging scripts (`make_release.py`, `verify_release.py`, `finalize_availability.py`) are maintainer-only.

Re-running overwrites files in `out/` with new time-stamped results; the saved results that the paper uses are kept in `out/`, so back them up first if you want to compare.

## Protocol guarantees

* Splits are by **subject** (both recordings of a subject stay together); the manifest was frozen before the test run (commit "FREEZE before final test", documented in `docs/TEST_SPLIT_HISTORY.md`).
* `mcd:test` can only be evaluated with `--final` (code guard, covered by a unit test).
* Model/epoch selection, HR-picking method, reference-validity threshold and the app's reliability thresholds use training/validation subjects only.
* Every number in the paper is generated from `paper/claims.json` (claim -> artifact file -> command); the build aborts on an unregistered number.
