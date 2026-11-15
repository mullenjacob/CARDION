# Development notes (deviations, substitutions, gotchas)

# NOTES — deviations, substitutions, gotchas

## Phase 0 — environment probe (2026-10-22)

| item | finding |
|---|---|
| OS | Windows 11 Pro 10.0.26300 |
| CPU / RAM | Intel i7-1260P (16 logical cores), 15.7 GB |
| GPU | none found (`nvidia-smi` absent); torch is CPU-only (`torch 2.14.1+cpu`, `cuda False`) |
| Disk | C: 832 GB free |
| Python | none on PATH (Microsoft Store stub only). Installed CPython **3.11.17** via portable `uv 0.13.5` |
| git / ffmpeg / pdflatex | none on PATH at start. **git**: portable MinGit 2.56.0.3 in `tools/mingit`. **ffmpeg**: not needed (OpenCV wheel bundles FFmpeg; FFV1 + MJPG writers verified). **pdflatex: still missing** (see Phase 9 note) |
| imports in `.venv` | `torch 2.14.1+cpu`, `cv2 5.0.0` (opencv-contrib), `mediapipe 1.1.1`, `snntorch 1.0.0`, `numpy 2.4.6`, `scipy 1.17.1` — all import OK |

## Environment setup facts

- `tools/` (portable uv + MinGit, gitignored) and `.venv/` are **not relocatable**. If the folder is moved, recreate:
  `tools\uv\uv.exe venv --python 3.11 .venv` then `tools\uv\uv.exe pip install --python .venv\Scripts\python.exe -r requirements.txt`
  (see REPRODUCE.md). `git init` needed `--template=` because of a long Windows path.
- `torch` in `requirements.txt` is pinned to the PyPI version; on a CUDA machine install the matching CUDA wheel first if you want the GPU.

## Deviations and substitutions

1. **Extra modules** `src/common.py` (config/logging/seeding/out handling/claims registry) and `src/datasets.py`
   (loaders, synthetic generator, splits, manifest, checksums). The charter layout has no home for these but Phase 1/3 require them.
2. **MediaPipe 1.1.1 ships only the Tasks API** (no `mp.solutions`) and no bundled model. `src/capture.py` uses
   `FaceLandmarker` with `face_landmarker.task` (3.58 MB, sha256 `64184e22…e0bc9ff`, downloaded from
   `storage.googleapis.com/mediapipe-models/...` to `data/models/`, gitignored with data). Verified on `skimage.data.astronaut()`
   (face found; eyes/lips excluded) — **not yet verified on a real video frame** (no real video yet, see 4).
3. **v2e is not on PyPI.** Source zip from GitHub (`third_party/v2e_src`, gitignored) is imported via `sys.path`; extra deps installed:
   `screeninfo engineering-notation numba easygui dv-processing`. `EventEmulator(device="cpu")` works. v2e's SuperSloMo temporal
   upsampling (needs a pretrained download) is **replaced by linear frame interpolation** (`dvs.upsample`).
4. **UBFC-rPPG videos are blocked by Google Drive quota** ("Too many users have viewed or downloaded this file recently") for all 42
   `vid.avi` files; the 42 `ground_truth.txt` files downloaded fine. All Phase 2–4 numbers so far therefore come from the **synthetic
   fallback generator** (`datasets.make_synthetic_clip`), *not* real data. Retry: `python scripts/download_ubfc.py` (resumable; quota
   can take up to 24 h to clear). PURE needs a request to the authors (per rPPG-Toolbox README) and is not available.
5. **rPPG-Toolbox reference subset is fetched, not committed** (its licence is custom, `NOASSERTION`): `python scripts/fetch_third_party.py` downloads
   POS_WANG, CHROME_DEHAAN, post_process and utils at pinned commit `b7500b8…` into `third_party/rppg_toolbox_ref/pkg` (gitignored), plus v2e (MIT) at
   `cac6384…`. Changes: package `__init__.py` files are empty (upstream star-imports methods we do not fetch). Needs `scikit-image`, `scikit-learn`. Upstream `POS_WANG` calls `np.mat`, removed in numpy 2 → `scripts/anchor_toolbox.py` sets
   `np.mat = np.asmatrix` (compat shim only; code unmodified).
6. **OpenCV 5.0 wheels bundle no Haar cascades**, so the planned "detector box" fallback ROI is not implemented; ROI = landmark skin mask.
7. **CHROM band**: Toolbox CHROM filters 0.7–2.5 Hz, ours 0.7–4.0 Hz (config `rppg.band_hz`), which is why CHROM waveform correlation with the
   Toolbox (≈0.98) is lower than POS (≈0.997) in the anchor.
8. `fft_hr` zero-pads 4× beyond the next power of two (finer BPM grid than the Toolbox's plain next-power-of-two).

## Interpretation gotchas (important for the paper)

- **Synthetic clips are easy** (window MAE ≈ 0.05 BPM for POS). They validate plumbing, not performance. Do not report them as benchmark results.
- **Events from 30 fps video contain nothing above 15 Hz.** The 10 ms event-bin "latency" study exposes interpolation, not new information.
- **Pulse information in events at realistic thresholds (≥0.1) comes from leak/noise dithering**, not from the pulse crossing the threshold
  (pulse is ≈0.5–1 % intensity, thresholds are 10–30 %). `python -m src.dvs_emulate ablate` shows |r| jumps from ≈0.3 to ≈0.9 at θ=0.1
  when the assumed leak rate goes 0 → 0.5 Hz. Real-sensor leak/noise is unmeasured here → large sim-to-real uncertainty.
- Correlation is reported with a **mismatched-PPG chance baseline** (|r| ≈ 0.01–0.1) because a ±0.5 s lag search inflates raw |r|.
- `paper/claims.json` entries from smoke tests on tiny splits must not be cited; only entries whose command is listed in HANDOFF are real.
- The `chrom` method shows window-MAE exactly 0 on the single smoke-test clip — a ≈1-clip artefact, not a result.

## Session 2 additions (2026-10-22/23)

- **SCAMPS**: only the 10-video example (1.2 GB) was downloaded; the 600 GB full set was not attempted. Used as the cross-dataset test set (all 10 clips, never trained on). n = 10 ? wide CIs.
- **Training data**: networks are trained on the synthetic generator only (18 train / 6 val / 6 test clips of 30 s). The SCAMPS?UBFC transfer from the charter was not possible (no UBFC videos).
- **SNN training length**: the 12-epoch SNN under-performed, so 40-epoch SNN/ANN runs were added (`syn_long`). Checkpoints are chosen by best *synthetic validation* MAE. The ANN's best epoch (7) is identical in both runs (same seed), hence identical ANN numbers. Reported as post hoc in the paper.
- **ANN control** has ReLU hidden layers (no hidden state) plus the same leaky-integrator read-out; part of any SNN advantage may be architectural.
- **Energy proxy**: add per ROI event for event estimators, add per ROI pixel/channel/frame for RGB (64�64 crop), 600 / 60 MACs per sample for POS / CHROM (analytic), SOPs measured for the SNN, dense MACs for the ANN; 4.6 pJ/MAC, 0.9 pJ/AC. Sensor, memory and state-update costs excluded.
- **Tectonic** (portable, `tools/tectonic`, SHA-256 verified) compiles the paper; it downloads TeX packages on first run. `pymupdf` was installed only to render PDF pages for visual checking (not in requirements).
- `paper/claims.json` is written by every evaluation command; concurrent runs can race. Re-run the producing command if a claim is missing (the paper build fails loudly).
- Failure analysis: on the hardest SCAMPS clip (lowest HR in the set) RGB POS locks to �2� the reference rate.

## Session 3 (2026-10-23/24): real data, app, corrected evaluation

- **Real data**: UBFC videos remain blocked by Google Drive quota (retry log: no videos after >24 h). Used **MCD-rPPG** (Hugging Face `kyegorov/mcd_rppg`, CC BY 4.0): 60 random subjects (seed 1337), frontal "FullHDwebcam" videos (640x480, MPEG-4), `ppg_sync` per frame. Container fps headers are wrong for some clips (24 vs ~30): the effective rate = frames / capture-timestamp duration (`datasets.mcd_effective_fs`). Split 36/12/12 by subject, frozen in `data/splits/mcd_seed1337.json`.
- **Test-set hygiene**: `mcd:test` requires `--final`. One test-subject clip (1097_before) was opened once while debugging the frame-rate bug; no accuracy-based decision followed. Test reference-HR range is shown in the dataset table (descriptive). The final test evaluation was run twice with identical models because the first run exposed a **summary bug**: `hr_metrics()['mae']` (window-pooled) overwrote the subject-level `mae` in each result row (tests/strata/per-clip data were unaffected). Fixed in `src/evaluate.py::summarise` (`win_mae` is the pooled value); validation and test were re-run; earlier outputs are in `out/superseded/`.
- **HR picking**: sub-harmonic summation (SHS) instead of plain peak picking, chosen on train+val subjects (small gain; it cannot recover an absent fundamental � see test).
- **Event path**: `emulate_event_frames` streams events straight into 33 ms frame bins (no event lists; exact same events as the list-based emulator, verified); channels are emulated as one (3H x W) image. Real video produces 35-75 events/pixel/s at theta 0.02, far above the simulator.
- **Model input**: `tanh(x/4)` soft clip (`snn.input_scale`) for the MCD networks. Earlier SCAMPS/synthetic checkpoints use raw counts (the checkpoint config records which).
- **Seeds**: `--seed` drives data/emulation, `--train-seed` the network init and window sampling. 3 seeds x {SNN, ANN} x theta {0.02, 0.05, 0.1}; controls: un-quantised log-difference input (1 seed), 4x training budget (ANN, theta 0.05).
- **Checkpoint hygiene**: a duplicate ANN theta=0.05 seed-2 checkpoint from an accidentally duplicated launch was deleted before evaluation (identical configuration).
- **App**: local web UI served on 127.0.0.1 (stdlib `http.server`); POST needs `X-CARDION: 1` and a local Host header. Windows camera access via DirectShow; **no camera was ever opened for capture during development** (device discovery and simulated failures only). Clean-room install (fresh folder, `requirements-app.txt`) + self-test passed 12/12. No standalone .exe: torch + mediapipe make a PyInstaller build very large/fragile; the launcher builds a private environment instead.
- **Live estimator**: the last 10 s are filtered together with 6 s of earlier context (avoids band-pass edge transients); this mattered (first version: 51 % of shown values within 5 BPM; with context 92 % on validation clips).
- **References**: all 16 bibliography entries verified against Crossref/arXiv (`scripts/verify_refs.py`).
- **Known weakness**: the networks may simply be under-trained/too small; a 4x budget did not help, a different architecture was not explored.

## Session 4: start-up failure ("[CARDION] Problem: Traceback (most recent call last):")

**Root cause (reproduced).** `scripts/launch.ps1` ran its environment check as `& $py -c "import ..." 2>$null` under
`$ErrorActionPreference = "Stop"`. In Windows PowerShell 5.1 a redirected native stderr line becomes a *terminating* error whose message is only the
first stderr line - here `Traceback (most recent call last):` - so the `catch` printed that one line and the real error (the failing import) was
lost. Reproduced on a disposable copy by breaking `numpy` inside an existing `.venv`: the old launcher printed exactly
`[CARDION] Problem: Traceback (most recent call last):`. The most likely trigger on a user machine is an interrupted/half-finished first install
(a `.venv` whose `python.exe` exists but whose packages are incomplete); the old launcher then never repaired it. (Whether that was the exact
trigger on the user's computer cannot be proven from here; the masking bug and the missing repair are proven.)

**Fix.** New launcher (no native stream redirection; programs write straight to the console or to files; every step logged to
`logs/launcher_*.log`; any failure writes `logs/launcher_error_*.log` with the complete error and opens it), `scripts/doctor.py` (quick check on
every launch; full diagnose/repair: completes the pinned install, force-reinstalls broken packages in dependency order, clears the uv cache
if a corrupted cache would restore the same files, rebuilds a damaged `.venv`, re-downloads a missing face model, falls back to pip + a system
Python if uv is blocked/offline), `Test and Repair CARDION.bat`, `src/cardion_app/cameratest.py` (+ UI button "Test my camera").
App/self-test crashes are written in full to `logs/app_*.log` / `logs/selftest_*.log`.

**Other bugs found while testing.** (1) Pipeline raised `IndexError` when the camera never saw a face (found by the real-webcam test); now reports
"No face found" (and does so after 3 s). (2) UI status line was overwritten by stale server messages. (3) `Stop` left the message "Running.".
(4) Browser disconnects logged `ConnectionResetError` tracebacks. (5) MediaPipe/OpenCV can fail on non-ASCII Windows paths: the face model is now
loaded from a byte buffer and video files are copied to an ASCII path if needed.

**Gotchas.** uv hard-links packages from its cache into every venv: editing a package file in place corrupts the cache and every environment
(this happened to the dev environment during a sabotage test and was repaired with `uv cache clean numpy`; sabotage tests now unlink first).
Repository files have mixed CRLF/LF line endings; scripted edits must normalise.

## Session 6 - guided positioning and quality feedback

**Added (advice only - the estimators are untouched):** `src/cardion_app/guidance.py` (face geometry from the MediaPipe landmarks CARDION already computes, smoothing, hysteresis, persistence, seven-way quality breakdown), multi-face handling (`LandmarkSkinROI.detect_all`, `num_faces=2`; the tracked face is followed, never silently swapped; a second face blocks the reading through a stricter gate status `multi_face`), browser overlay (mirrored preview, face box, target box, banner, progress strip, stepper), "last reliable reading", extra columns in recorded CSVs, `status_counts` in session summaries.

**Evidence used to set thresholds (out/guidance_calibration_*.json, `scripts/calibrate_guidance.py`; 5 research clips re-framed like a webcam shot, then moved/scaled/darkened synthetically):**
- Moving the whole face sideways/up/down (up to 0.3 of the frame) left the RGB-POS estimate within 1 BPM of the centred run on all 5 clips, so centring advice is given only when the face approaches the frame edge (enter at 30% offset, release at 20%), and never while a reliable pulse is being read.
- Face height of 22% of the frame gave estimates within 0.9 BPM of the full-size run; at 18% four of five clips were unchanged but the weakest clip shifted by 15.8 BPM; at about 16% the share of reliable seconds fell (0.20 on average) and below that the face was no longer detected. "Move a little closer" therefore starts below 20% of the frame height.
- Darkening to 25% brightness is blocked by the existing reliability gate; guidance simply mirrors that gate.
- Limits of this evidence: 5 clips, one dataset, synthetic repositioning, no reference monitor in the sweep.

**Not validated on real footage:** head-turn ("Face the camera directly") - thresholds come from the yaw spread of real frontal faces (p95 0.10; limits 0.20 enter / 0.14 exit) and synthetic landmark tests; a real sideways head was not available. Uneven-lighting advice is a measured left/right brightness difference, not a proven cause of weak signals.

**Bugs found by testing the real app and fixed:** (1) "Reliable pulse detected." lingered 1-2 s after the gate withdrew the BPM; (2) `/api/state` failed with two faces (numpy floats in JSON); (3) MediaPipe deadlocked when the garbage collector closed an old FaceLandmarker (test run hung in `FaceLandmarker.__del__`) - owners now call `close()` explicitly (`LandmarkSkinROI.close`, `Pipeline.close`); (4) the notice "this computer processed only 10 fps" was wrong when the *camera* was the bottleneck (dim room: the raw webcam read rate was 8.9 fps with CARDION idle); now reported as a camera note and the event methods are no longer paused for it.

**Estimates unchanged:** the same 600 real frames of two recordings through the old tree (tag `backup-before-guidance`) and the new tree give identical status and BPM for all 7 methods every second (`scripts/compare_estimates.py`, `out/estimate_equivalence_*.json`). Overhead: no measurable change in pipeline time per frame (5 ms/frame RGB path, 17-18 ms/frame with events; `scripts/bench_guidance.py`); the second-face detector costs +0.7 ms per detection (every 6th frame).

## Session 5 - WinError 206 (project folder too deep) and misleading reports

**Root cause (from the user's diagnostic output):** the project lived in a ~203-character folder. torch, mediapipe, Pillow (`_imaging`) and
matplotlib contain files ~125 characters deep, so inside `<project>\.venv` the full path passed 260 characters. Windows cannot load DLLs from
such paths (`WinError 206 The filename or extension is too long`) even when "long paths" are enabled for other programs. Reinstalling in the
same folder can never fix it. Reproduced on purpose in a 215-character folder: cv2, mediapipe, torch, snntorch and matplotlib failed exactly
as reported; numpy/scipy/yaml (shallow files) imported fine.

**Fixes**
- `scripts/launch.ps1`: if the project path is > 80 characters, or a previous run hit the limit (`logs/env_location.txt`), the environment is
  built in `%USERPROFILE%\.cardion\env_<hash>` instead. Models/data/results always stay in the project folder.
- `scripts/doctor.py`: measures the longest file path inside the environment; recognises WinError 206 / DLL-load errors in deep folders; does
  NOT reinstall packages in that case; exit code 11 makes the launcher relocate the environment. Repair attempts for an identical error are
  limited to two (`logs/repair_state.json`).
- Reports: every test deletes its previous report first, the report carries a run id, and a crashed test shows "NO REPORT WAS PRODUCED BY THIS
  RUN" instead of an older PASS. `out/test_and_repair_<runid>.md` is kept per run.
- `Move CARDION to a short folder.bat` copies the project (without the environment) to `%USERPROFILE%\CARDION`.
- `CARDION_MAX_ROOT_LEN` env var: test switch that makes the launcher keep the environment inside a long project folder (used to test relocation).

## Session 7 - smoothness, validation mode, discrepancy analysis

**Profiling findings (before changing anything; mock camera replaying a real face at true 30 fps, `scripts/profile_live.py`, tag `backup-before-validation`):**
* PyTorch used 12 threads for two tiny networks: the app burned 5.8-11.3 CPU cores and ran at ~36 ms/frame (slower than real time). With 2 threads: ~20 ms/frame on ~2 cores, SNN/ANN outputs and all estimates **bit-identical** (`scripts/bench_threads.py`). Now capped by `app.torch_threads: 2`.
* Camera reading, pipeline and JPEG encoding shared one loop, so any slow step (the once-per-second estimate, peaks of 100-180 ms) stalled the preview: 19-20 fps delivered, p95 gap 83-121 ms. Now a capture thread reads/draws the preview (time stamps taken at capture) and a processing thread consumes a short queue: 30 fps, p95 gap 36 ms, no dropped frames. MJPEG streams wake on new frames instead of polling every 40 ms; the event picture is only encoded while someone watches it.
* Page: 92 DOM writes/s and a 26.7 KB state payload twice a second (47.6 KB/s) -> 14.6 DOM writes/s, a 1 KB `/api/live` poll 4x/s plus the 23.7 KB full state about once a second (25.3 KB/s). The page's main thread was never overloaded (no long tasks), so "jumpy" was the display logic, not browser speed.
* **BPM display flicker** came from the status flipping ok/not-ok (8-13 flips per minute on gate-limited clips); the shown number itself changed by 1-2 BPM. The readout now keeps the last reliable number visible but clearly marked as older, with its age.
* **Delay to first reading**: the gate needs >= 8 s of face data and then agreement of >= 2 strong RGB methods. In good conditions the first reliable number appears at ~9-10 s after Start; in weak-signal conditions it is later or absent. This is necessary and was not shortened. The reliable share (after warm-up) was 26-51 % on the two profiled clips (research validation split: 35 %).
* Frame-rate effect (`scripts/fps_effect.py`, true time stamps): dropping to 20 or 15 fps does not systematically change availability (mean reliable share 0.34 / 0.37 / 0.30 at 30 / 20 / 15 fps) and shifts the shown BPM by a median of <= 0.4 BPM. Accurate time stamps matter more than frame count; the old loop stamped frames when it *read* them, which drifts under load. Differences in availability between "before" and "after" in the mock-camera profiles are partly a harness effect and are **not** claimed as an improvement.
* Bug found by testing: `MockCapture` started its 30 fps clock before a multi-second set-up, then replayed a burst of frames. Fixed (test harness only).
* Bug found by testing: un-closed `FaceLandmarker` objects hang the interpreter at exit; `Engine.stop()` now closes the pipeline.

**Estimates unchanged:** `scripts/compare_estimates.py` on the old and new trees gives identical status and BPM for all 7 methods every second (2 clips x 600 real frames). `Pipeline.estimate`, the quality gate, thresholds and the models were not touched.

**Which estimator produces the displayed number?** The mean of the RGB methods (green, POS, CHROM) that agree within 8 BPM and have spectral peak ratio >= 8, when at least 2 do (`configs/quality_calibrated.json`). The event, SNN and ANN methods are computed and shown for comparison but **do not feed the headline**. The app's accuracy is therefore a classical-rPPG result, not an SNN result, and says nothing about event-camera hardware.

**Discrepancy analysis (`scripts/analyze_discrepancy.py`, validation split only; test numbers quoted from existing files):**
* Ungated, per window: RGB POS 6.6 BPM MAE, CHROM 14.0, green 14.8, ANN 10.2, SNN 12.9, event POS 18.7, event green 23.7 (12 validation subjects). The research tables report these ungated numbers.
* The app displays a number for 35 % of validation windows (41 % on the frozen test split) with 3.0 BPM MAE (2.4 on test), 72 % of shown values within 1 BPM, 88 % within 3 BPM.
* Most of the gap is **selection**: on exactly the windows the gate lets through, RGB POS alone has 2.3 BPM MAE (6.6 ungated) - the gate keeps the easy windows. Averaging with green and CHROM does not beat POS alone there (2.98 vs 2.31). On those windows SNN 11.9, ANN 6.0, event POS 11.5: the neural/event methods do not benefit and are not what makes live readings accurate.
* Higher heart rates and after-activity recordings are measured better; resting low-rate windows are the hardest (gated MAE 4.4 at HR < 76 vs 1.3 at HR >= 90).
* Not separable with these data: lighting, camera, compression, watch-versus-oximeter reference timing and watch error, and the informal nature of the live comparison. Validation mode exists to measure the last two properly.
* An observation, not acted on: POS alone is better than the consensus on gate-passing validation windows. Changing the displayed estimator would be a tuning decision and needs a fresh, untouched evaluation set; nothing was changed.

**Privacy observation:** MediaPipe's native library logs `Network error: 502 for url: https://play.googleapis.com/log` to stderr when the face tracker starts. CARDION sends no video or readings, but whether the library transmits anything cannot be verified here; mentioned in `docs/VALIDATION_PROTOCOL.md` for IRB purposes.

**Not tested:** a real person at the real webcam (the camera was opened and delivered 29.7 fps with nobody in view -> "no face", no number); an actual Apple Watch comparison; the Apple Health import with a real export file (tested with a synthetic XML of the documented record format); validation mode with several real participants.

## Session 8 - paper rewrite (see docs/PAPER_REVISION_NOTES.md, docs/PAPER_REVIEW.md, docs/LITERATURE_REVIEW.md)

  see docs/AI_DISCLOSURE_AND_AUTHORSHIP.md before using any of this text in a competition.
* Authoritative sources: `paper/main.tex.in` (+ `refs.bib`, `literature.json`, `claims.json`), built by `scripts/build_paper.py` / `scripts/build_paper.ps1`. Old `paper/sections/*.tex.in` were folded into the template.
* New scripts: `scripts/paper_analysis.py` (independent recomputation of reported statistics, paired effect sizes, gate audit, accuracy-availability curve, 5 figures, 2 tables, ~200 claims),
  `scripts/fetch_bib.py` (Crossref/arXiv metadata), `scripts/register_literature.py`, `scripts/make_reference_table.py`, `scripts/replot_noise.py`.
* Audit findings: statistics reproduce (p-values exactly; MAE/CI within 0.005 BPM); the frozen gate re-implemented independently reproduces 486/1198 displayed windows; Table 1 window counts differ from
  evaluation windows because one validation clip was recorded at 18.4 fps (90 windows); the 8-clip real-time check is not representative of the 12-subject gate result (MAE 1.0 vs 2.37 BPM).
* Findings added to the paper: gate improves accuracy mostly by selection; POS alone beats the 3-method consensus on the windows the gate shows (validation 2.31 vs 2.98 BPM, test 1.46 vs 2.37) and a POS-only
  peak-ratio gate has a better accuracy-availability curve on validation windows (exploratory). The application and its estimators were NOT changed.
* No model retrained; no threshold changed; the test split was used only to re-read frozen outputs (gate audit, paired effect sizes) and this is disclosed.

## Session 9 - publication preparation (final audit)
* Paper corrections: test-set history (abstract and Section V), window counts reconciled exactly (`scripts/reconcile_windows.py`; Table II now uses the evaluation's fixed-1794-frame excerpts), Spiking-PhysFormer figures (journal 10.1 % vs preprint 12.4 %; block 12.2x), "information loss" wording replaced by observed-accuracy / hypothesis language, AI-disclosure paragraph extended, appendix reorganised (12 pages). Details: publication/SCIENTIFIC_CORRECTIONS.md.
* New scripts: `scripts/test_history_audit.py`, `scripts/reconcile_windows.py`, `scripts/make_release.py` (staging area, scans, arXiv zip), `scripts/finalize_availability.py` (writes real links after publication; refuses placeholders).
* `release/` holds the public README, licensing files and Zenodo metadata templates; `publication/` the private guides. Staging output: `../CARDION_release/`.
* Tests: 98 passed / 1 skipped; self-test 18/18; clean-environment run of the staged tree 82-83 passed, 16-17 skipped, 0 failed (docs/REPRODUCIBILITY_REPORT.md).
* Nothing was uploaded, reserved or published.

## Session 10 - publication-package cleanup
* BOMs removed from release templates (zenodo.json, CITATION.cff, gitignore_public, LICENSE) and make_release.scrub() now strips BOMs / normalises LF in staged text files; both metadata files validated (strict json.loads; cffconvert, CFF 1.2.0).
* scripts/verify_release.py inspects the finished zips (BOM, metadata, README = template, private strings, arXiv layout/compile/PDF-text identity, optional final-link consistency); make_release.py runs it and fails on any problem.
* MCD-rPPG licence verified on the official page (cc-by-4.0, Hugging Face API, 2026-10-25); one sentence in the paper, docs and checklists updated. README (root + public) statements on test-split history and live-webcam validation corrected.
* test_history_audit.py / reconcile_windows.py exit with an explanation when history/data are absent (maintainer-only); publication sequence rewritten as two releases (v1.0.0 -> DOI -> v1.0.1 with links), enforced by finalize_availability.py + verify_release.py.
