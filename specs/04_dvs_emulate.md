# Spec 04 — DVS emulation (`src/dvs_emulate.py`, `src/events.py`)

- `events.EventStream(t, x, y, p, height, width, duration)`; encoders `event_frame`, `voxel_grid`, `tbr` (temporal binary representation: 1 if a pixel fired in a sub-bin), `binned_rates`, `frame_sequence`, `render_event_frame`.
- `AnalyticDVS(threshold, sigma_rel, refractory_s, leak_hz, shot_noise_hz, cutoff_hz, upsample, log_eps, ref_init, seed)`: `emulate(frames(T,H,W), fs) -> EventStream`, streaming `start()/step()`.
  Log intensity `ln(I+eps)`; ON/OFF when the reference is passed by the (mismatched) threshold; refractory; leak (spontaneous ON); Poisson shot noise; optional photoreceptor low-pass; linear frame interpolation (`upsample`).
- `V2EDVS`: wraps v2e `EventEmulator` (CPU). Same interface. Backends agree on event rate at θ=0.02 (1.40 vs 1.38 ev/px/s on synthetic:0).
- `emulate_rgb(face(T,H,W,3), fs, cfg, backend, threshold, seed) -> [R, G, B] streams` (independent per-channel emulation).
- `pulse_correlation(streams, mask, fs, ppg, null_ppg)` -> |r| of integrated net events vs PPG at best lag (±0.5 s) plus mismatched-PPG chance level.
- CLI: `sweep` (rate-vs-threshold, event frames, rate-vs-pulse figures), `ablate` (leak/shot/ref-init), `smoke` (both backends).
