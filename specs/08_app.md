# Spec 08 - application (`src/cardion_app/`)

* `sources.py`: `CameraSource(index)`, `VideoSource(path)`, `DemoSource(cfg, bpm)` -> `read() = (frame_rgb, t_seconds, roi_override|None)`; `list_cameras()`; `SourceError` carries a user-readable message (never raised into the UI).
* `pipeline.py::Pipeline(cfg, models, use_landmarks, threshold)`: `push(frame, t, roi_override)` per frame, `estimate()` about once per second -> `Assessment`; estimators on the last 10 s (+6 s filter context): RGB green/POS/CHROM, classical event green/POS, streaming SNN/ANN; `costs()` (ms per second of video, SOPs/MACs per second).
* `quality.py::assess(estimates, env, params)`: statuses `ok, collecting, no_face, motion, dark, bright, unstable_light, weak_signal, disagree`; a BPM is returned only for `ok` (>= `min_agree` RGB voters with peak ratio >= `min_peak_ratio` within `agree_bpm`). Thresholds: `configs/quality_calibrated.json` (written by `scripts/calibrate_quality.py` from validation data).
* `engine.py::Engine`: owns the source thread, latest JPEGs, state JSON, recording (CSV + summary + plot [+ mp4]) and session listing.
* `server.py`: `/`, `/api/state|cameras|sessions|source|stop|record|upload|open_sessions|shutdown`, `/stream/rgb.mjpg`, `/stream/events.mjpg`, `/api/session/<id>.zip`.
* `selftest.py`: 12 checks, writes `out/selftest_report.{md,json}`; exit code 0 only if all pass.
Tests: `tests/test_app.py`, self-test, `scripts/app_real_video_check.py` (real MCD videos through the live pipeline).
