"""Streaming CARDION pipeline: frames in, per-second heart-rate estimates and quality out (no threads, no I/O -> testable).

Per frame: face landmarks (every few frames) -> smoothed square face crop -> three emulated DVS channels (R, G, B) -> events binned
into 33.3 ms frames. Estimators (all on the same last <=10 s window): RGB green / POS / CHROM, event green / POS (classical), and the
streaming SNN / matched ANN. Every number shown by the app comes from here.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ..dvs_emulate import AnalyticDVS
from ..efficiency import E_AC_PJ, E_MAC_PJ, macs_per_step
from ..events import render_event_frame
from ..rppg_classical import bandpass, chrom, detrend, fft_hr, green, pos
from ..rppg_events import green_events, pos_from_traces
from .guidance import GuidanceParams, GuidanceTracker, geometry_from_landmarks, side_luma
from .quality import Assessment, QualityParams, assess

BIN_FS = 30.0                      # event bin rate (33.3 ms), the rate the networks were trained at
METHODS = ["rgb_green", "rgb_pos", "rgb_chrom", "event_green", "event_pos", "snn", "ann"]
LABELS = {"rgb_green": "RGB green", "rgb_pos": "RGB POS", "rgb_chrom": "RGB CHROM", "event_green": "Event green (classical)",
          "event_pos": "Event POS (classical)", "snn": "Spiking NN (SNN)", "ann": "Matched ANN"}


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    """Linear resampling of irregular samples (``t``, ``x`` shape (N,) or (N,C)) onto a uniform grid [t0, t1)."""
    grid = np.arange(t0, t1, 1.0 / fs)
    if x.ndim == 1:
        return np.interp(grid, t, x)
    return np.stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])], axis=1)


class Pipeline:
    """Online heart-rate pipeline. Call :meth:`push` per frame and :meth:`estimate` about once per second."""

    def __init__(self, cfg: dict, models: dict | None = None, use_landmarks: bool = True, threshold: float | None = None,
                 guidance: bool = False) -> None:
        self.cfg = cfg
        self.guidance_enabled = bool(guidance and use_landmarks)      # live positioning advice (camera only); never changes the estimates
        self.app = cfg.get("app", {})
        qcfg = {k: v for k, v in self.app.get("quality", {}).items() if k != "voters"}
        cal = Path(__file__).resolve().parents[2] / "configs" / "quality_calibrated.json"      # written by scripts/calibrate_quality.py (validation only)
        if cal.exists():
            c = json.loads(cal.read_text(encoding="utf-8-sig"))
            qcfg.update({"min_peak_ratio": c["min_peak_ratio"], "agree_bpm": c["agree_bpm"], "min_agree": c["min_agree"]})
            qcfg["voters"] = tuple(c["voters"])
        self.qp = QualityParams(**qcfg)
        self.window_s = float(cfg["rppg"]["hr_window_s"])
        self.context_s = 6.0                     # extra history filtered together with the window (avoids band-pass edge transients)
        self.crop = int(cfg["dvs"]["crop_size"])
        self.threshold = float(threshold if threshold is not None else self.app.get("threshold", cfg["dvs"]["threshold"]))
        self.models = models or {}
        self.events_enabled = True               # event emulation + SNN/ANN; can be paused when the CPU cannot keep up in real time
        self.detect_every = int(cfg["face"].get("live_detect_every", 6))
        self.roi = None
        if use_landmarks:
            from ..capture import LandmarkSkinROI
            self.roi = LandmarkSkinROI(cfg)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget everything (new source)."""
        self.n = 0
        self.rec: deque = deque(maxlen=int((self.window_s + self.context_s) * 40))          # (t, r, g, b, face_ok, cx, cy, side, luma, another_face_visible)
        gp = GuidanceParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in self.app.get("guidance", {}).items()})
        self.guide: GuidanceTracker | None = GuidanceTracker(gp) if self.guidance_enabled else None
        self.multi_now = False                                           # another (second) face is currently visible
        self.bins_net: deque = deque(maxlen=int((self.window_s + self.context_s + 2) * BIN_FS))
        self.bin_out = {k: deque(maxlen=int((self.window_s + self.context_s + 2) * BIN_FS)) for k in ("snn", "ann")}
        self.box: np.ndarray | None = None                                # smoothed (cx, cy, side) in full-frame pixels
        self.mask_full: np.ndarray | None = None
        self.mask_crop = np.ones((self.crop, self.crop), bool)
        self.dvs: AnalyticDVS | None = None
        self.prev_face: np.ndarray | None = None
        self.prev_t = 0.0
        self.cur_bin = 0
        self.bin_acc = np.zeros((3, self.crop, self.crop), np.float32)
        self.net_acc = np.zeros(3)
        self.disp = np.zeros((self.crop, self.crop), np.float32)         # decaying event image for display
        self.events_total = 0
        self.last_face_frame = -999
        self.history: list[dict] = []
        self.timing = {k: 0.0 for k in ("roi", "dvs", "snn", "ann", "est")}
        self.t_first: float | None = None
        for m in self.models.values():
            m.reset_state(1)
        self._ops0 = self._ops_now()
        self.last_assessment: Assessment | None = None
        self.last_est: dict = {}
        self.last_pulses: dict = {}
        self.last_env: dict = {}

    def close(self) -> None:
        """Release the face tracker deterministically (see ``LandmarkSkinROI.close``). The pipeline must not be pushed to afterwards."""
        if self.roi is not None:
            self.roi.close()
            self.roi = None

    def _ops_now(self) -> dict[str, float]:
        out = {}
        if "snn" in self.models:
            from ..efficiency import sops_from_counts
            out["snn"] = sops_from_counts(self.models["snn"])
        if "ann" in self.models:
            out["ann"] = macs_per_step(self.models["ann"], self.crop // 2, self.crop // 2) * self._bins_pushed()
        return out

    def _bins_pushed(self) -> int:
        return len(self.bin_out["ann"]) if "ann" in self.models else 0

    def set_events(self, enabled: bool) -> None:
        """Pause or resume the event path (emulation, event estimators, SNN/ANN). Resuming restarts the sensor and the networks cleanly."""
        if enabled and not self.events_enabled:
            self.dvs = None
            self.prev_face = None
            self.cur_bin = int(np.floor(self.rec[-1][0] * BIN_FS)) if self.rec else 0
            self.bin_acc[:] = 0
            self.net_acc[:] = 0
            self.bins_net.clear()
            for m in self.models.values():
                m.reset_state(1)
            for q in self.bin_out.values():
                q.clear()
        self.events_enabled = enabled

    # ------------------------------------------------------------------ per frame
    def _pick_face(self, faces: list[np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray | None, list[tuple]]:
        """Choose which face is analysed. One face: that face (unchanged behaviour). Several: keep following the face that was tracked
        (never silently jump to another person); with no face tracked yet, take the largest. Returns (landmarks or None, boxes of the other faces)."""
        if not faces:
            return None, []
        H, W = shape
        boxes = [(p.min(axis=0), p.max(axis=0)) for p in faces]
        if len(faces) == 1:
            return faces[0], []
        heights = [b[1][1] - b[0][1] for b in boxes]
        tracked = self.box is not None and self.n - self.last_face_frame <= int(1.0 * 30)
        if tracked:
            d = [float(np.hypot((lo[0] + hi[0]) / 2 - self.box[0], (lo[1] + hi[1]) / 2 - self.box[1])) for lo, hi in boxes]
            main = int(np.argmin(d))
            if d[main] > 0.8 * self.box[2]:
                main = -1                                           # the tracked person left; do not adopt someone else instantly
        else:
            main = int(np.argmax(heights))
        ref = max(heights) if main < 0 else heights[main]
        others = [(float(lo[0] / W), float(lo[1] / H), float(hi[0] / W), float(hi[1] / H)) for i, (lo, hi) in enumerate(boxes)
                  if i != main and heights[i] >= GuidanceParams.second_face_min * ref]
        return (faces[main] if main >= 0 else None), others

    def _update_face(self, frame: np.ndarray, roi_override: tuple | None, t: float = 0.0) -> bool:
        """Refresh landmarks on schedule; returns whether a face is currently tracked."""
        h, w = frame.shape[:2]
        if roi_override is not None:
            mask, box = roi_override
            self.mask_full, self.box = mask, np.asarray(box, float)
            self.last_face_frame = self.n
            self._update_crop_mask()
            return True
        if self.roi is not None and (self.n % self.detect_every == 0 or self.box is None):
            pts, others = self._pick_face(self.roi.detect_all(frame), frame.shape[:2])
            self.multi_now = bool(others)
            geom = None
            if pts is not None:
                m = self.roi.mask_from_landmarks(pts, frame.shape[:2])
                ys, xs = np.nonzero(m)
                if len(xs) > 50:
                    new = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, 1.1 * max(xs.max() - xs.min(), ys.max() - ys.min())])
                    self.box = new if self.box is None else 0.6 * self.box + 0.4 * new       # smooth: no crop jumps
                    self.mask_full = m
                    self.last_face_frame = self.n
                    self._update_crop_mask()
                    if self.guide is not None:
                        geom = geometry_from_landmarks(pts, frame.shape[:2], side_luma(frame, m, float(new[0])))
            if self.guide is not None:
                self.guide.aspect = w / h
                self.guide.update(t, geom, others)
        return self.n - self.last_face_frame <= int(1.0 * 30)

    def _crop_box(self, shape: tuple[int, int]) -> tuple[int, int, int]:
        h, w = shape
        cx, cy, side = self.box
        s = int(round(min(max(side, 16), min(h, w))))
        x0 = int(min(max(round(cx - side / 2), 0), w - s))
        y0 = int(min(max(round(cy - side / 2), 0), h - s))
        return x0, y0, s

    def _update_crop_mask(self) -> None:
        if self.mask_full is None or self.box is None:
            return
        x0, y0, s = self._crop_box(self.mask_full.shape)
        self.mask_crop = cv2.resize(self.mask_full[y0:y0 + s, x0:x0 + s], (self.crop, self.crop), interpolation=cv2.INTER_NEAREST).astype(bool)

    def push(self, frame: np.ndarray, t: float, roi_override: tuple | None = None) -> None:
        """Process one RGB frame captured at time ``t`` (seconds)."""
        if self.t_first is None:
            self.t_first = t
        t = t - self.t_first
        t0 = time.perf_counter()
        face_ok = self._update_face(frame, roi_override, t)
        self.timing["roi"] += time.perf_counter() - t0
        if self.box is None or self.mask_full is None:
            self.rec.append((t, np.nan, np.nan, np.nan, False, np.nan, np.nan, np.nan, np.nan, float(self.multi_now)))
            self.n += 1
            return
        sel = frame[self.mask_full.astype(bool)]
        rgb = sel.mean(axis=0) if len(sel) else np.full(3, np.nan)
        luma = float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        self.rec.append((t, rgb[0], rgb[1], rgb[2], face_ok, self.box[0], self.box[1], self.box[2], luma, float(self.multi_now)))
        if not self.events_enabled:
            self.n += 1
            return
        # --- event path (the three colour channels are emulated as one tall image: one vectorised pass per sub-step)
        t1 = time.perf_counter()
        x0, y0, s = self._crop_box(frame.shape[:2])
        face = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (self.crop, self.crop), interpolation=cv2.INTER_AREA).astype(np.float32)
        stacked = np.concatenate([face[..., c] for c in range(3)], axis=0)            # (3*crop, crop)
        if self.dvs is None:
            self.dvs = AnalyticDVS.from_cfg(self.cfg, self.threshold, seed=self.cfg["seed"])
            self.dvs.start(stacked, 30.0)
        elif face_ok:
            up = self.dvs.up
            dt = max(t - self.prev_t, 1e-3)
            for k in range(1, up + 1):
                f = k / up
                img = (1 - f) * self.prev_face + f * stacked
                for te, ex, ey, ep in self.dvs.step(img, self.prev_t + (k - 1) * dt / up, self.prev_t + k * dt / up):
                    c, y = ey // self.crop, ey % self.crop
                    inside = self.mask_crop[y, ex]
                    if inside.any():
                        self._add_events(te[inside], c[inside], ex[inside], y[inside], ep[inside])
        self.prev_face, self.prev_t = stacked, t
        self.timing["dvs"] += time.perf_counter() - t1
        self._flush_bins(t)
        self.n += 1

    def _add_events(self, te: np.ndarray, c: np.ndarray, ex: np.ndarray, y: np.ndarray, ep: np.ndarray) -> None:
        """Assign events to 33.3 ms bins (flushing earlier bins as time advances)."""
        n = self.crop
        b = np.floor(te * BIN_FS).astype(np.int64)
        for bi in np.unique(b):
            if bi > self.cur_bin:
                self._flush_to(bi)
            sel = b == bi
            w = ep[sel].astype(np.float64)
            self.bin_acc += np.bincount((c[sel] * n + y[sel]) * n + ex[sel], weights=w, minlength=3 * n * n).reshape(3, n, n).astype(np.float32)
            self.net_acc += np.bincount(c[sel], weights=w, minlength=3)
            g = sel & (c == 1)
            self.disp += np.bincount(y[g] * n + ex[g], weights=ep[g].astype(np.float64), minlength=n * n).reshape(n, n).astype(np.float32)
        self.events_total += len(te)

    def _flush_bins(self, t: float) -> None:
        """Flush bins that ended before time ``t``."""
        target = int(np.floor(t * BIN_FS))
        if target > self.cur_bin:
            self._flush_to(target)
        self.disp *= 0.8

    def _flush_to(self, new_bin: int) -> None:
        """Close bins ``cur_bin .. new_bin-1`` (empty bins are pushed as zeros) and advance the networks."""
        while self.cur_bin < new_bin:
            area = max(float(self.mask_crop.sum()), 1.0)
            self.bins_net.append(self.net_acc / area)
            x = self.bin_acc.reshape(3, self.crop // 2, 2, self.crop // 2, 2).sum(axis=(2, 4))
            for name, m in self.models.items():
                ts = time.perf_counter()
                with torch.no_grad():
                    y = m.step(torch.from_numpy(x[None].astype(np.float32)))
                self.bin_out[name].append(float(y[0]))
                self.timing[name] += time.perf_counter() - ts
            self.bin_acc[:] = 0
            self.net_acc[:] = 0
            self.cur_bin += 1

    # ------------------------------------------------------------------ estimates
    def seconds(self) -> float:
        """Seconds of data currently buffered."""
        return (self.rec[-1][0] - self.rec[0][0]) if len(self.rec) > 1 else 0.0

    def valid_seconds(self) -> float:
        """Seconds of buffered data in which a face was actually tracked (the progress shown to the user; never feeds an estimate)."""
        rows = list(self.rec)
        if len(rows) < 2:
            return 0.0
        return float(sum(b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[4] and np.isfinite(b[1])))

    def _window_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(self.rec, dtype=float)
        t_end = arr[-1, 0]
        keep = arr[:, 0] >= t_end - self.window_s - self.context_s
        return arr[keep], arr[keep, 0]

    def estimate(self) -> Assessment:
        """Compute all estimates on the last window and assess reliability. Also appends to ``self.history``."""
        t0 = time.perf_counter()
        r = self.cfg["rppg"]
        band, hr_band, method = tuple(r["band_hz"]), tuple(r["hr_band_hz"]), r.get("hr_method", "peak")
        est: dict[str, dict] = {}
        pulses: dict[str, np.ndarray] = {}
        arr, t = self._window_arrays() if len(self.rec) > 30 else (np.zeros((0, 10)), np.zeros(0))
        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        ok = np.isfinite(arr[:, 1]) if len(arr) else np.zeros(0, bool)
        if dur >= 4.0 and ok.sum() >= int(2.0 * BIN_FS):             # no face for most of the window -> nothing to estimate (never crash)
            tt, rgb = t[ok], arr[ok, 1:4]
            n = int(dur * BIN_FS)
            tail = int(self.window_s * BIN_FS)
            grid_t0 = tt[0]
            rgbu = resample_uniform(tt, rgb, BIN_FS, grid_t0, grid_t0 + n / BIN_FS)

            def put(name: str, pulse: np.ndarray) -> None:
                """Band-pass the full (window + context) pulse, then pick the HR from the last window only."""
                pulse = bandpass(pulse - pulse.mean(), BIN_FS, band[0], band[1])[-tail:]
                bpm, ratio = fft_hr(pulse, BIN_FS, hr_band, method)
                est[name] = {"bpm": bpm, "peak_ratio": ratio}
                pulses[name] = pulse
            for name, fn in (("rgb_green", lambda: green(rgbu, BIN_FS, band)), ("rgb_pos", lambda: pos(rgbu, BIN_FS, r["pos_window_s"], band)),
                             ("rgb_chrom", lambda: chrom(rgbu, BIN_FS, r["pos_window_s"], band))):
                try:
                    put(name, fn())
                except Exception:  # noqa: BLE001 - a failed estimator must not kill the app
                    est[name] = {"bpm": float("nan"), "peak_ratio": 0.0}
            net = np.array(self.bins_net)[-n:] if len(self.bins_net) else np.zeros((0, 3))
            if len(net) >= 4 * BIN_FS:
                tr = np.cumsum(net, axis=0)
                try:
                    put("event_green", detrend(tr[:, 1], 100.0))
                    put("event_pos", pos_from_traces(tr, BIN_FS, r["pos_window_s"], band))
                except Exception:  # noqa: BLE001
                    pass
            for name in ("snn", "ann"):
                out = np.array(self.bin_out[name])[-n:] if name in self.models else np.zeros(0)
                if len(out) >= 4 * BIN_FS:
                    put(name, out)
        # --- environment
        last = arr[arr[:, 0] >= (arr[-1, 0] - 3.0)] if len(arr) else arr
        face_fraction = float(np.mean(arr[:, 4])) if len(arr) else 0.0
        have_face = len(arr) > 0 and bool(np.isfinite(arr[:, 8]).any())
        lf = last[np.isfinite(last[:, 7])] if len(last) else last
        motion = float(np.nanstd(lf[:, 5:7], axis=0).max() / max(np.nanmean(lf[:, 7]), 1.0)) if len(lf) > 5 else 0.0
        luma = float(np.nanmean(arr[:, 8])) if have_face else 0.0
        luma_var = float(np.nanstd(arr[:, 8]) / max(luma, 1.0)) if have_face else 0.0
        multi = float(np.mean(last[:, 9])) if len(last) else 0.0           # share of the last 3 s with a second face in view
        env = {"seconds": dur if len(arr) else 0.0, "face_fraction": face_fraction, "motion": motion, "luma": luma, "luma_var": luma_var,
               "multi_face": multi}
        a = assess(est, env, self.qp)
        if self.guide is not None:
            self.guide.set_assessment(a.status, env, {**a.detail, "min_seconds": self.qp.min_seconds}, float(self.rec[-1][0]) if self.rec else 0.0)
        self.timing["est"] += time.perf_counter() - t0
        self.last_est, self.last_pulses, self.last_env = est, pulses, env
        self.last_assessment = a
        self.history.append({"t": float(self.rec[-1][0]) if self.rec else 0.0, "status": a.status, "bpm": a.bpm, "confidence": a.confidence,
                             **{f"{k}_bpm": v["bpm"] for k, v in est.items()}, **{f"{k}_q": v["peak_ratio"] for k, v in est.items()}})
        return a

    def event_view(self, size: int = 256) -> np.ndarray:
        """RGB image of the recent green-channel events on the face crop (white background, red ON / blue OFF)."""
        return cv2.resize(render_event_frame(self.disp, scale=3.0), (size, size), interpolation=cv2.INTER_NEAREST)

    def costs(self) -> dict[str, dict]:
        """Per-method cost accounting over the run so far (per second of video): wall time and proxy operations."""
        secs = max(self.rec[-1][0], 1e-6) if self.rec else 1e-6
        ops = self._ops_now()
        ev_per_s = self.events_total / secs
        out = {
            "shared_roi": {"ms_per_s": 1000 * self.timing["roi"] / secs, "note": "face landmarks + skin mask (all methods)"},
            "dvs_emulation": {"ms_per_s": 1000 * self.timing["dvs"] / secs, "events_per_s": ev_per_s,
                              "note": "software emulation of the sensor; a real event camera does this in hardware"},
            "estimators": {"ms_per_s": 1000 * self.timing["est"] / secs, "note": "FFT/POS/CHROM/event estimators, once per second"},
            "snn": {"ms_per_s": 1000 * self.timing["snn"] / secs, "sops_per_s": (ops.get("snn", 0) - self._ops0.get("snn", 0)) / secs,
                    "energy_uj_per_s_proxy": (ops.get("snn", 0) - self._ops0.get("snn", 0)) / secs * E_AC_PJ * 1e-6},
            "ann": {"ms_per_s": 1000 * self.timing["ann"] / secs, "macs_per_s": (ops.get("ann", 0) - self._ops0.get("ann", 0)) / secs,
                    "energy_uj_per_s_proxy": (ops.get("ann", 0) - self._ops0.get("ann", 0)) / secs * E_MAC_PJ * 1e-6},
        }
        return out
