"""Real-time demo: webcam (or a video file / synthetic source) -> live BPM, with RGB and event views side by side.

Left panel: RGB frame with the skin ROI outlined and the POS BPM. Right panel: synthetic DVS event view of the face
crop (red ON / blue OFF) and the event-domain BPM. Optionally records the composite to an mp4 under ``out/``.

    python -m src.demo_live --source camera            # press q to quit
    python -m src.demo_live --source synthetic --max-frames 450 --no-display
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

from .capture import LandmarkSkinROI, WebcamCapture, masked_mean_rgb, video_frames
from .common import add_common_args, out_path, resolve_cfg, set_seed, setup_logging
from .dvs_emulate import AnalyticDVS
from .events import render_event_frame
from .rppg_classical import bandpass, fft_hr, pos
from .rppg_events import pos_from_traces

log = logging.getLogger("cardion.demo")


class LiveEstimator:
    """Streaming RGB-POS and event-POS heart-rate estimator over a rolling window."""

    def __init__(self, cfg: dict, fs: float, crop: int = 64) -> None:
        self.cfg, self.fs, self.crop = cfg, fs, crop
        n = int(cfg["rppg"]["hr_window_s"] * fs)
        self.rgb: deque = deque(maxlen=n)
        self.ev: deque = deque(maxlen=n)
        self.dvs = [AnalyticDVS.from_cfg(cfg, seed=cfg["seed"] + c) for c in range(3)]
        self.prev: np.ndarray | None = None
        self.t = 0.0
        self.bpm_rgb = self.bpm_ev = float("nan")

    def step_events(self, face: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Advance the three DVS channels by one frame; return the signed green event image (crop x crop)."""
        face = cv2.resize(face, (self.crop, self.crop), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST).astype(bool)
        net, img_g = np.zeros(3), np.zeros((self.crop, self.crop), np.float32)
        up, dt = self.dvs[0].up, 1.0 / (self.fs * self.dvs[0].up)
        if self.prev is None:
            for c in range(3):
                self.dvs[c].start(face[..., c], self.fs)
        else:
            for c in range(3):
                for s in range(1, up + 1):
                    f = s / up
                    img = (1 - f) * self.prev[..., c] + f * face[..., c]
                    for _, x, y, p in self.dvs[c].step(img, self.t + (s - 1) * dt, self.t + s * dt):
                        keep = m[y, x]
                        net[c] += p[keep].sum()
                        if c == 1:
                            np.add.at(img_g, (y, x), p.astype(np.float32))
            self.t += 1.0 / self.fs
        self.prev = face.astype(np.float32)
        self.ev.append(net / max(m.sum(), 1))
        return img_g

    def update_bpm(self) -> None:
        """Recompute both BPM estimates once at least 6 s of data are buffered."""
        r = self.cfg["rppg"]
        if len(self.rgb) >= int(6 * self.fs):
            pulse = pos(np.asarray(self.rgb), self.fs, r["pos_window_s"], tuple(r["band_hz"]))
            self.bpm_rgb = fft_hr(pulse, self.fs, tuple(r["hr_band_hz"]))[0]
            tr = np.cumsum(np.asarray(self.ev), axis=0)
            self.bpm_ev = fft_hr(pos_from_traces(tr, self.fs, r["pos_window_s"], tuple(r["band_hz"])), self.fs,
                                 tuple(r["hr_band_hz"]))[0]


def open_source(args: argparse.Namespace, cfg: dict):
    """Return (frame iterator yielding RGB uint8, fps, closer)."""
    if args.source == "camera":
        cam = WebcamCapture(args.camera, 640, 480, 30)

        def gen():
            while True:
                f = cam.read()
                if f is None:
                    return
                yield f
        return gen(), 30.0, cam.close
    if args.source == "video":
        it, fps, _ = video_frames(args.video)
        return it, fps, lambda: None
    from .datasets import make_synthetic_clip
    sc = {**cfg, "datasets": {**cfg["datasets"], "synthetic": {**cfg["datasets"]["synthetic"], "size": 240, "clip_seconds": 40}}}
    clip = make_synthetic_clip(0, sc)
    return iter(clip.face), clip.fs, lambda: None


def main() -> None:
    """CLI entry point."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--source", default="camera", choices=["camera", "video", "synthetic"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--record", action="store_true", help="write the composite view to out/demo_*.mp4")
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    frames, fs, close = open_source(args, cfg)
    roi = LandmarkSkinROI(cfg)
    est = LiveEstimator(cfg, fs)
    writer, n, t_start, last_log = None, 0, time.perf_counter(), 0.0
    for frame in frames:
        mask = roi.mask(frame)
        if mask is None:
            continue
        est.rgb.append(masked_mean_rgb(frame, mask))
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        ev_img = est.step_events(frame[y0:y1, x0:x1], mask[y0:y1, x0:x1])
        if n % int(fs) == 0:
            est.update_bpm()
        vis = frame.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        ev_rgb = cv2.resize(render_event_frame(ev_img, scale=2.0), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        for img, label in ((vis, f"RGB POS: {est.bpm_rgb:5.1f} BPM"), (ev_rgb, f"Event POS: {est.bpm_ev:5.1f} BPM")):
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        comp = cv2.cvtColor(np.hstack([vis, ev_rgb]), cv2.COLOR_RGB2BGR)
        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(str(out_path(cfg, "demo", "mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fs,
                                         (comp.shape[1], comp.shape[0]))
            writer.write(comp)
        n += 1
        elapsed = time.perf_counter() - t_start
        if elapsed - last_log >= 5.0:
            last_log = elapsed
            log.info("frame %d  processing %.1f fps  RGB %.1f BPM  event %.1f BPM", n, n / elapsed, est.bpm_rgb, est.bpm_ev)
        if not args.no_display:
            cv2.imshow("CARDION", comp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and n >= args.max_frames:
            break
    close()
    roi.close()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    el = time.perf_counter() - t_start
    log.info("done: %d frames in %.1fs = %.1f fps; final RGB %.1f BPM, event %.1f BPM", n, el, n / max(el, 1e-9), est.bpm_rgb, est.bpm_ev)


if __name__ == "__main__":
    main()
