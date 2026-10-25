> Note: this spec dates from the original plan, in which UBFC-rPPG was the primary dataset. The final study uses **MCD-rPPG** (36/12/12 subject-disjoint split; see `data/README.md` and the paper); UBFC-rPPG and PURE loaders remain but were not used for any reported result.

# Spec 02 — classical rPPG (`src/capture.py`, `src/rppg_classical.py`)

- `capture.LandmarkSkinROI(cfg).mask(frame_rgb) -> uint8 (H,W) | None`: MediaPipe FaceLandmarker (Tasks API) face-oval polygon minus eyes, brows, lips; landmarks refreshed every `face.redetect_every` frames.
- `capture.extract_rgb_trace(path, cfg) -> ((T,3) mean skin RGB, fps)`; `video_frames`, `WebcamCapture` for sources.
- `rppg_classical.pos / chrom / green (rgb, fs, ...) -> pulse (T,)`: windowing mirrors rPPG-Toolbox (POS stride 1, window 1.6 s).
- `fft_hr(x, fs, band) -> (bpm, peak_ratio)`; `sliding_hr(x, fs, win_s=10, stride_s=1)`; `ppg_reference_hr(ppg, ...)` applies the *same* windows to the contact PPG.
- `estimate(rgb, fs, method, cfg) -> (pulse, window_times, bpm)`.
- Correctness anchor: `python scripts/anchor_toolbox.py --clips ...` feeds the identical RGB trace to the Toolbox `POS_WANG`/`CHROME_DEHAAN` and reports waveform correlation and MAE.
CLI: `python -m src.rppg_classical --clip synthetic:0|ubfc:subject1`.
