# Spec 05 — event-domain pulse (`src/rppg_events.py`)

- `event_log_traces(streams, mask, bin_s)`: integrate per-channel net (ON-OFF) counts over the ROI -> (T,3) log-intensity-like traces.
- `pos_from_traces(tr, fs)` POS-style projection (mean-subtracted log traces replace normalised C_n); `pos_events(streams, mask, bin_s)`; `green_events`.
- Bin width `event_rppg.window_ms`: 33.3 ms (frame rate, main comparison) and 10 ms (latency study). Synthetic events from 30 fps video carry no information above 15 Hz.
- CLI: `python -m src.rppg_events --clip synthetic:0`.
