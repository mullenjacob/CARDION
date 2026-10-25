"""Event-stream container and encodings: event frames, voxel grids, binned rates and TBR.

An :class:`EventStream` holds sorted events ``(t [s], x, y, p in {+1,-1})`` for one sensor channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EventStream:
    """Sorted address-event stream for one channel."""
    t: np.ndarray      # float64 seconds, ascending
    x: np.ndarray      # int16 column
    y: np.ndarray      # int16 row
    p: np.ndarray      # int8, +1 ON / -1 OFF
    height: int
    width: int
    duration: float    # seconds covered (first frame to last frame)

    def __len__(self) -> int:
        return len(self.t)

    def save(self, path: str | Path) -> None:
        """Write the stream to a compressed .npz file."""
        np.savez_compressed(path, t=self.t, x=self.x, y=self.y, p=self.p, height=self.height, width=self.width,
                            duration=self.duration)

    @staticmethod
    def load(path: str | Path) -> "EventStream":
        """Read a stream written by :meth:`save`."""
        z = np.load(path)
        return EventStream(z["t"], z["x"], z["y"], z["p"], int(z["height"]), int(z["width"]), float(z["duration"]))

    def select(self, mask: np.ndarray) -> "EventStream":
        """Keep only events whose pixel is inside the boolean ``mask`` (H,W)."""
        keep = mask[self.y, self.x].astype(bool)
        return EventStream(self.t[keep], self.x[keep], self.y[keep], self.p[keep], self.height, self.width,
                           self.duration)

    def rate_per_pixel(self) -> float:
        """Mean event rate in events / pixel / second."""
        return len(self) / (self.height * self.width * max(self.duration, 1e-9))


def event_frame(ev: EventStream, t0: float, t1: float) -> np.ndarray:
    """Signed event-count image on ``[t0, t1)``: shape (H,W), ON=+1 per event, OFF=-1."""
    lo, hi = np.searchsorted(ev.t, [t0, t1])
    img = np.zeros((ev.height, ev.width), np.float32)
    np.add.at(img, (ev.y[lo:hi], ev.x[lo:hi]), ev.p[lo:hi].astype(np.float32))
    return img


def voxel_grid(ev: EventStream, t0: float, t1: float, n_bins: int) -> np.ndarray:
    """Polarity-weighted voxel grid (n_bins, H, W) with bilinear weighting in time (Zhu et al. 2019)."""
    lo, hi = np.searchsorted(ev.t, [t0, t1])
    grid = np.zeros((n_bins, ev.height, ev.width), np.float32)
    if hi <= lo:
        return grid
    tn = (ev.t[lo:hi] - t0) / max(t1 - t0, 1e-12) * (n_bins - 1)
    b0 = np.floor(tn).astype(int)
    w1 = (tn - b0).astype(np.float32)
    pol = ev.p[lo:hi].astype(np.float32)
    ys, xs = ev.y[lo:hi], ev.x[lo:hi]
    np.add.at(grid, (b0, ys, xs), pol * (1 - w1))
    b1 = np.minimum(b0 + 1, n_bins - 1)
    np.add.at(grid, (b1, ys, xs), pol * w1)
    return grid


def tbr(ev: EventStream, t0: float, t1: float, n_bins: int) -> np.ndarray:
    """Temporal binary representation: (n_bins, H, W) uint8, 1 where the pixel fired in that sub-bin."""
    lo, hi = np.searchsorted(ev.t, [t0, t1])
    out = np.zeros((n_bins, ev.height, ev.width), np.uint8)
    if hi > lo:
        b = np.minimum(((ev.t[lo:hi] - t0) / max(t1 - t0, 1e-12) * n_bins).astype(int), n_bins - 1)
        out[b, ev.y[lo:hi], ev.x[lo:hi]] = 1
    return out


def binned_rates(ev: EventStream, bin_s: float, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin ON count and OFF count over the (optionally masked) region.

    Returns (bin centres [s], on_counts, off_counts), each of length ``floor(duration / bin_s)``.
    """
    e = ev.select(mask) if mask is not None else ev
    n = int(np.floor(ev.duration / bin_s))
    edges = np.arange(n + 1) * bin_s
    on, _ = np.histogram(e.t[e.p > 0], bins=edges)
    off, _ = np.histogram(e.t[e.p < 0], bins=edges)
    return (edges[:-1] + edges[1:]) / 2, on.astype(np.float64), off.astype(np.float64)


def frame_sequence(ev: EventStream, fs: float, n_frames: int, pool: int = 2) -> np.ndarray:
    """Signed event counts per frame-period bin: (n_frames, H/pool, W/pool), sum-pooled spatially."""
    seq = np.zeros((n_frames, ev.height, ev.width), np.float32)
    b = np.minimum((ev.t * fs).astype(int), n_frames - 1)
    np.add.at(seq, (b, ev.y, ev.x), ev.p.astype(np.float32))
    h, w = ev.height // pool, ev.width // pool
    return seq[:, :h * pool, :w * pool].reshape(n_frames, h, pool, w, pool).sum(axis=(2, 4))


def render_event_frame(img: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Render a signed event image as RGB (white background, red = ON, blue = OFF)."""
    s = scale or max(float(np.abs(img).max()), 1.0)
    v = np.clip(img / s, -1, 1)
    rgb = np.ones(img.shape + (3,), np.float32)
    rgb[..., 1] = 1 - np.abs(v)
    rgb[..., 0] = np.where(v < 0, 1 + v, 1)
    rgb[..., 2] = np.where(v > 0, 1 - v, 1)
    return (rgb * 255).astype(np.uint8)
