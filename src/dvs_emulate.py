"""Video -> synthetic DVS events: an analytic emulator plus a wrapper around v2e.

Physics (both backends): per-pixel log-intensity ``L = ln(I + eps)``; an event of polarity +/-1 fires when
``L`` departs from a per-pixel reference by one contrast threshold, and the reference then steps by that
threshold. The analytic backend adds threshold mismatch, a refractory period, a leak current (spontaneous ON
events), Poisson shot noise and an optional photoreceptor low-pass. Each RGB channel is emulated as its own
stream, which is what makes chrominance projection possible in the event domain.

Limits that matter for interpretation (also stated in the paper): input video has finite frame rate and bit
depth, so intermediate frames are linearly interpolated and synthetic events carry no temporal information
above the source frame rate; and ``ref_init="random"`` assumes pixel references are already phase-randomised
by prior activity, as in a running sensor.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .common import REPO_ROOT, add_common_args, out_path, register_claim, resolve_cfg, save_json, set_seed, setup_logging
from .events import EventStream, binned_rates, event_frame, render_event_frame

log = logging.getLogger("cardion.dvs")


def _interp_frames(frames: np.ndarray, fs: float, up: int):
    """Yield (t_prev, t_now, intensity) sub-steps with linear interpolation between consecutive frames."""
    for k in range(1, len(frames)):
        a, b = frames[k - 1].astype(np.float32), frames[k].astype(np.float32)
        for s in range(1, up + 1):
            f = s / up
            yield (k - 1 + (s - 1) / up) / fs, (k - 1 + f) / fs, (1 - f) * a + f * b


def _expand(ys: np.ndarray, xs: np.ndarray, counts: np.ndarray, sign: np.ndarray, t_prev: float, t_now: float):
    """Expand per-pixel event counts into individual events spread uniformly across (t_prev, t_now]."""
    total = int(counts.sum())
    ry, rx, rp = np.repeat(ys, counts), np.repeat(xs, counts), np.repeat(sign, counts)
    j = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    tt = t_prev + (t_now - t_prev) * (j + 1) / np.repeat(counts, counts)
    return tt, rx, ry, rp


class AnalyticDVS:
    """Vectorised DVS pixel-array model for a single intensity channel."""

    def __init__(self, threshold: float = 0.02, sigma_rel: float = 0.2, refractory_s: float = 0.001,
                 leak_hz: float = 0.05, shot_noise_hz: float = 0.05, cutoff_hz: float = 0.0, upsample: int = 4,
                 log_eps: float = 1.0, ref_init: str = "random", seed: int = 0) -> None:
        self.theta, self.sigma_rel, self.refr = threshold, sigma_rel, refractory_s
        self.leak_hz, self.shot_hz, self.cutoff_hz = leak_hz, shot_noise_hz, cutoff_hz
        self.up, self.eps, self.ref_init, self.seed = upsample, log_eps, ref_init, seed

    @classmethod
    def from_cfg(cls, cfg: dict, threshold: float | None = None, seed: int = 0, **kw) -> "AnalyticDVS":
        """Build from the ``dvs`` config section (``threshold`` overrides the default operating point)."""
        d = cfg["dvs"]
        args = dict(threshold=threshold if threshold is not None else d["threshold"], sigma_rel=d["sigma_threshold"],
                    refractory_s=d["refractory_s"], leak_hz=d["leak_hz"], shot_noise_hz=d["shot_noise_hz"],
                    cutoff_hz=d["cutoff_hz"], upsample=d["upsample"], log_eps=d["log_eps"], seed=seed)
        args.update(kw)
        return cls(**args)

    def start(self, first_frame: np.ndarray, fs: float) -> None:
        """Initialise pixel state from the first frame (H,W); call once before :meth:`step`."""
        H, W = first_frame.shape
        self._rng = np.random.default_rng(self.seed)
        rng = self._rng
        self._th_on = self.theta * np.clip(1 + self.sigma_rel * rng.standard_normal((H, W)), 0.2, None)
        self._th_off = self.theta * np.clip(1 + self.sigma_rel * rng.standard_normal((H, W)), 0.2, None)
        L0 = np.log(first_frame.astype(np.float64) + self.eps)
        self._ref = L0.copy()
        if self.ref_init == "random":
            self._ref = L0 - rng.uniform(-self._th_off, self._th_on)  # L - ref ~ U(-th_off, th_on) at start
        self._lp = L0.copy()
        self._last_t = np.full((H, W), -np.inf)
        self._dt = 1.0 / (fs * self.up)
        self._alpha = 1.0 if self.cutoff_hz <= 0 else 1 - np.exp(-2 * np.pi * self.cutoff_hz * self._dt)

    def step(self, img: np.ndarray, t_prev: float, t_now: float) -> list[tuple]:
        """Advance one sub-step with intensity ``img``; returns a list of (t, x, y, p) event chunks."""
        H, W = img.shape
        th_on, th_off, ref, dt = self._th_on, self._th_off, self._ref, self._dt
        chunks: list[tuple] = []
        L = np.log(img.astype(np.float64) + self.eps)
        if self._alpha < 1.0:
            self._lp += self._alpha * (L - self._lp)
            L = self._lp
        ref -= self.leak_hz * th_on * dt
        d = L - ref
        on_n = np.floor(np.clip(d, 0, None) / th_on).astype(np.int64)
        off_n = np.floor(np.clip(-d, 0, None) / th_off).astype(np.int64)
        n = np.where(on_n > 0, on_n, off_n)
        sign = np.where(on_n > 0, 1, -1).astype(np.int8)
        fire = n > 0
        if self.refr > 0:
            fire &= (t_now - self._last_t) >= self.refr
            n = np.minimum(n, max(1, int(dt / self.refr)))
        ys, xs = np.nonzero(fire)
        if len(ys):
            c, sg = n[ys, xs], sign[ys, xs]
            ref[ys, xs] += sg * c * np.where(sg > 0, th_on[ys, xs], th_off[ys, xs])
            self._last_t[ys, xs] = t_now
            tt, rx, ry, rp = _expand(ys, xs, c, sg, t_prev, t_now)
            chunks.append((tt, rx, ry, rp))
        if self.shot_hz > 0:
            ny, nx = np.nonzero(self._rng.random((H, W)) < self.shot_hz * dt)
            if len(ny):
                chunks.append((self._rng.uniform(t_prev, t_now, len(ny)), nx, ny,
                               np.where(self._rng.random(len(ny)) < 0.5, 1, -1).astype(np.int8)))
        return chunks

    def emulate(self, frames: np.ndarray, fs: float) -> EventStream:
        """Emulate events from ``frames`` (T,H,W) intensity in [0,255] sampled at ``fs`` Hz."""
        T, H, W = frames.shape
        self.start(frames[0], fs)
        out_t, out_x, out_y, out_p = [], [], [], []
        for t_prev, t_now, img in _interp_frames(frames, fs, self.up):
            for tt, rx, ry, rp in self.step(img, t_prev, t_now):
                out_t.append(tt); out_x.append(rx); out_y.append(ry); out_p.append(rp)
        return _pack(out_t, out_x, out_y, out_p, H, W, (T - 1) / fs)


class V2EDVS:
    """Wrapper over v2e's ``EventEmulator`` (CPU). Same interface as :class:`AnalyticDVS`."""

    def __init__(self, threshold: float = 0.02, sigma_abs: float = 0.004, refractory_s: float = 0.001,
                 leak_hz: float = 0.05, shot_noise_hz: float = 0.05, cutoff_hz: float = 0.0, upsample: int = 4,
                 seed: int = 1) -> None:
        src = REPO_ROOT / "third_party" / "v2e_src" / "v2e-master"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        logging.getLogger("v2ecore").setLevel(logging.WARNING)
        from v2ecore.emulator import EventEmulator
        self._cls = EventEmulator
        self.kw = dict(pos_thres=threshold, neg_thres=threshold, sigma_thres=sigma_abs, cutoff_hz=cutoff_hz,
                       leak_rate_hz=leak_hz, refractory_period_s=refractory_s, shot_noise_rate_hz=shot_noise_hz,
                       device="cpu", seed=max(1, seed))
        self.up = upsample

    @classmethod
    def from_cfg(cls, cfg: dict, threshold: float | None = None, seed: int = 1) -> "V2EDVS":
        """Build from the ``dvs`` config section."""
        d = cfg["dvs"]
        th = threshold if threshold is not None else d["threshold"]
        return cls(threshold=th, sigma_abs=d["sigma_threshold"] * th, refractory_s=d["refractory_s"],
                   leak_hz=d["leak_hz"], shot_noise_hz=d["shot_noise_hz"], cutoff_hz=d["cutoff_hz"],
                   upsample=d["upsample"], seed=seed)

    def emulate(self, frames: np.ndarray, fs: float) -> EventStream:
        """Feed interpolated frames to v2e; v2e's own linear-log mapping (<20 counts is linear) applies."""
        T, H, W = frames.shape
        em = self._cls(**self.kw)
        em.generate_events(frames[0].astype(np.float32), 0.0)
        chunks = []
        for _, t_now, img in _interp_frames(frames, fs, self.up):
            ev = em.generate_events(img, t_now)
            if ev is not None and len(ev):
                chunks.append(np.asarray(ev))
        if not chunks:
            return _pack([], [], [], [], H, W, (T - 1) / fs)
        e = np.concatenate(chunks)
        return _pack([e[:, 0]], [e[:, 1].astype(np.int64)], [e[:, 2].astype(np.int64)],
                     [np.where(e[:, 3] > 0, 1, -1).astype(np.int8)], H, W, (T - 1) / fs)


def _pack(ts, xs, ys, ps, H: int, W: int, duration: float) -> EventStream:
    """Concatenate chunk lists into a time-sorted :class:`EventStream`."""
    if not ts:
        z = np.zeros(0)
        return EventStream(z, z.astype(np.int16), z.astype(np.int16), z.astype(np.int8), H, W, duration)
    t = np.concatenate(ts).astype(np.float64)
    order = np.argsort(t, kind="stable")
    return EventStream(t[order], np.concatenate(xs).astype(np.int16)[order], np.concatenate(ys).astype(np.int16)[order],
                       np.concatenate(ps).astype(np.int8)[order], H, W, duration)


def emulate_rgb(face: np.ndarray, fs: float, cfg: dict, backend: str | None = None, threshold: float | None = None,
                seed: int = 0, **kw) -> list[EventStream]:
    """Emulate the three colour channels of ``face`` (T,H,W,3) as independent event streams (R, G, B)."""
    backend = backend or cfg["dvs"]["backend"]
    if backend == "analytic":
        # The pixel model has no spatial coupling, so the three colour channels are emulated as one tall image (3H x W) in a single
        # vectorised pass (independent per-pixel threshold mismatch / noise for every channel) and split back by row. ~3x faster.
        T, H, W = face.shape[:3]
        stacked = np.concatenate([face[..., c] for c in range(3)], axis=1)
        full = AnalyticDVS.from_cfg(cfg, threshold, seed=seed, **kw).emulate(stacked, fs)
        out = []
        for c in range(3):
            sel = (full.y >= c * H) & (full.y < (c + 1) * H)
            out.append(EventStream(full.t[sel], full.x[sel], (full.y[sel] - c * H).astype(np.int16), full.p[sel], H, W, full.duration))
        return out
    streams = []
    for c in range(3):
        dvs = (V2EDVS.from_cfg(cfg, threshold, seed + c + 1) if backend == "v2e"
               else AnalyticDVS.from_cfg(cfg, threshold, seed + c, **kw))
        streams.append(dvs.emulate(face[..., c], fs))
    return streams


def prepare_face(clip, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize a clip's face crop and skin mask to ``size`` x ``size``."""
    face = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in clip.face])
    mask = cv2.resize(clip.face_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return face, mask


def pulse_correlation(streams: list[EventStream], mask: np.ndarray, fs: float, ppg: np.ndarray,
                      max_lag_s: float = 0.5, null_ppg: np.ndarray | None = None) -> dict:
    """How well does the event signal track the pulse?

    Net signed rate (ON-OFF) per frame-sized bin is integrated, bandpassed to 0.7-4 Hz and correlated with the
    bandpassed contact PPG at the best lag within +/-``max_lag_s`` (sign-agnostic: |r|). Also returns the same for
    the green channel alone, and the unsigned total event rate vs |d PPG/dt|.
    """
    from .rppg_classical import bandpass, detrend
    ref = bandpass(detrend(ppg - ppg.mean()), fs, 0.7, 4.0)
    n_lag = int(max_lag_s * fs)

    def best(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        n = min(len(x), len(y))
        best_r, best_l = 0.0, 0
        for lag in range(-n_lag, n_lag + 1):
            a, b = (x[lag:n], y[:n - lag]) if lag >= 0 else (x[:n + lag], y[-lag:n])
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_l = r, lag
        return best_r, best_l / fs

    nets, tot = [], 0.0
    for s in streams:
        _, on, off = binned_rates(s, 1.0 / fs, mask)
        nets.append(bandpass(detrend(np.cumsum(on - off)), fs, 0.7, 4.0))
        tot = tot + on + off
    out = {}
    out["r_net_green"], out["lag_green_s"] = best(nets[1], ref)
    out["r_net_sum"], out["lag_sum_s"] = best(sum(nets), ref)
    rate = bandpass(detrend(tot - tot.mean()), fs, 0.7, 4.0)
    out["r_unsigned_vs_absderiv"], _ = best(rate, np.abs(np.gradient(ref)) - np.abs(np.gradient(ref)).mean())
    if null_ppg is not None:  # chance level: same lag search against another clip's (unrelated) pulse
        nref = bandpass(detrend(null_ppg - null_ppg.mean()), fs, 0.7, 4.0)
        out["r_null_green"], _ = best(nets[1], nref)
    return out


def run_sweep(cfg: dict, clips: list[str], backend: str) -> dict:
    """Event rate and pulse correlation as a function of contrast threshold, over several clips."""
    from .datasets import load_clip
    size = int(cfg["dvs"]["crop_size"])
    res: dict = {"backend": backend, "thresholds": cfg["dvs"]["thresholds"], "clips": clips, "per_clip": {}}
    loaded = [load_clip(spec, cfg) for spec in clips]
    for i, (spec, clip) in enumerate(zip(clips, loaded)):
        face, mask = prepare_face(clip, size)
        null_ppg = loaded[(i + 1) % len(loaded)].ppg if len(loaded) > 1 else None
        rows = []
        for th in cfg["dvs"]["thresholds"]:
            st = emulate_rgb(face, clip.fs, cfg, backend, th, seed=cfg["seed"])
            corr = pulse_correlation(st, mask, clip.fs, clip.ppg, null_ppg=null_ppg)
            row = {"threshold": th, "events_per_pixel_per_s": float(np.mean([s.rate_per_pixel() for s in st])),
                   "n_events": int(sum(len(s) for s in st)), **corr}
            rows.append(row)
            log.info("%s th=%.3f rate=%.3f ev/px/s r_green=%.3f r_sum=%.3f lag=%.2fs null=%s", spec, th,
                     row["events_per_pixel_per_s"], corr["r_net_green"], corr["r_net_sum"], corr["lag_sum_s"],
                     f"{corr['r_null_green']:.3f}" if "r_null_green" in corr else "n/a")
        res["per_clip"][spec] = rows
    return res


ABLATION_SETTINGS = [  # (label, leak_hz, shot_noise_hz, ref_init)
    ("default", None, None, "random"), ("no leak", 0.0, None, "random"), ("no leak, no shot", 0.0, 0.0, "random"),
    ("no leak, no shot, aligned refs", 0.0, 0.0, "aligned"), ("strong leak 0.5 Hz", 0.5, 0.0, "random")]


def run_ablation(cfg: dict, clips: list[str], thresholds: list[float]) -> dict:
    """Which emulator ingredients carry the pulse? |r| (net events vs PPG) and chance level under each setting."""
    from .datasets import load_clip
    loaded = [load_clip(c, cfg) for c in clips]
    res: dict = {"clips": clips, "thresholds": thresholds, "settings": [s[0] for s in ABLATION_SETTINGS], "rows": []}
    for i, clip in enumerate(loaded):
        face, mask = prepare_face(clip, int(cfg["dvs"]["crop_size"]))
        null_ppg = loaded[(i + 1) % len(loaded)].ppg
        for th in thresholds:
            for label, leak, shot, ref_init in ABLATION_SETTINGS:
                kw = {"ref_init": ref_init}
                if leak is not None:
                    kw["leak_hz"] = leak
                if shot is not None:
                    kw["shot_noise_hz"] = shot
                st = emulate_rgb(face, clip.fs, cfg, "analytic", th, seed=cfg["seed"], **kw)
                c = pulse_correlation(st, mask, clip.fs, clip.ppg, null_ppg=null_ppg)
                row = {"clip": clip.name, "threshold": th, "setting": label, "rate": float(np.mean([s.rate_per_pixel() for s in st])),
                       "abs_r_green": abs(c["r_net_green"]), "abs_r_unsigned_vs_absderiv": abs(c["r_unsigned_vs_absderiv"]),
                       "abs_r_null": abs(c["r_null_green"])}
                res["rows"].append(row)
                log.info("%s th=%.2f %-34s rate=%.3f |r|=%.3f |r_unsigned|=%.3f null=%.3f", clip.name, th, label, row["rate"],
                         row["abs_r_green"], row["abs_r_unsigned_vs_absderiv"], row["abs_r_null"])
    return res


def plot_all(cfg: dict, res: dict, demo_clip: str, backend: str) -> dict[str, Path]:
    """Write the three Phase-4 figures (rate vs threshold, event frames, rate vs pulse)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .datasets import load_clip
    from .rppg_classical import bandpass, detrend

    paths: dict[str, Path] = {}
    ths = np.array(res["thresholds"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for spec, rows in res["per_clip"].items():
        ax[0].loglog(ths, [r["events_per_pixel_per_s"] for r in rows], "o-", alpha=0.6, label=spec)
        ax[1].semilogx(ths, [abs(r["r_net_sum"]) for r in rows], "o-", alpha=0.6)
    null = np.array([[abs(r.get("r_null_green", np.nan)) for r in rows] for rows in res["per_clip"].values()])
    if np.isfinite(null).any():
        ax[1].semilogx(ths, np.nanmean(null, axis=0), "k--", label="chance (mismatched PPG)")
        ax[1].legend(fontsize=7)
    ax[0].set_xlabel("contrast threshold (log-intensity units)")
    ax[0].set_ylabel("event rate (events / pixel / s)")
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel("contrast threshold (log-intensity units)")
    ax[1].set_ylabel("|r| (integrated net events vs contact PPG)")
    ax[1].set_ylim(0, 1)
    fig.suptitle(f"Event rate and pulse correlation vs threshold ({backend} emulator)")
    fig.tight_layout()
    paths["rate_vs_threshold"] = out_path(cfg, "dvs_rate_vs_threshold", "png")
    fig.savefig(paths["rate_vs_threshold"], dpi=150)
    plt.close(fig)

    clip = load_clip(demo_clip, cfg)
    face, mask = prepare_face(clip, int(cfg["dvs"]["crop_size"]))
    st = emulate_rgb(face, clip.fs, cfg, backend, seed=cfg["seed"])
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    ax[0].imshow(face[len(face) // 2]); ax[0].set_title("input frame")
    t0 = 10.0
    for k, (name, w) in enumerate((("G events, 33 ms", 1 / 30), ("G events, 200 ms", 0.2), ("G events, 1 s", 1.0))):
        ax[k + 1].imshow(render_event_frame(event_frame(st[1], t0, t0 + w)))
        ax[k + 1].set_title(name)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    paths["event_frames"] = out_path(cfg, "dvs_event_frames", "png")
    fig.savefig(paths["event_frames"], dpi=150)
    plt.close(fig)

    corr = pulse_correlation(st, mask, clip.fs, clip.ppg)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fs = clip.fs
    _, on, off = binned_rates(st[1], 1 / fs, mask)
    t = np.arange(len(on)) / fs
    ax[0].plot(t, on + off, lw=0.8)
    ax[0].set_ylabel("events / frame-bin (G, ROI)")
    ref = bandpass(detrend(clip.ppg - clip.ppg.mean()), fs, 0.7, 4.0)
    net = bandpass(detrend(np.cumsum(on - off)), fs, 0.7, 4.0)
    lag = int(round(corr["lag_green_s"] * fs))
    shifted = np.roll(ref, lag)[:len(net)] * np.sign(corr["r_net_green"])
    ax[1].plot(t[:len(net)], net / net.std(), label="integrated net events (G)")
    ax[1].plot(t[:len(shifted)], shifted / ref.std(), "k", lw=0.8, label="contact PPG (lag/sign aligned)")
    ax[1].set_xlim(0, 15)
    ax[1].set_xlabel("time (s)")
    ax[1].legend(fontsize=8)
    ax[0].set_title(f"{demo_clip}: |r| = {abs(corr['r_net_green']):.3f} at lag {corr['lag_green_s']:.2f} s")
    fig.tight_layout()
    paths["rate_vs_pulse"] = out_path(cfg, "dvs_rate_vs_pulse", "png")
    fig.savefig(paths["rate_vs_pulse"], dpi=150)
    plt.close(fig)
    res["demo"] = {"clip": demo_clip, **corr}
    return paths


def main() -> None:
    """CLI: ``sweep`` runs the threshold sweep and writes figures + JSON; ``smoke`` compares both backends."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("command", choices=["sweep", "smoke", "ablate"])
    ap.add_argument("--clips", nargs="+", default=[f"synthetic:{i}" for i in range(3)])
    ap.add_argument("--backend", default=None, choices=["analytic", "v2e"])
    ap.add_argument("--ablate-thresholds", nargs="+", type=float, default=[0.02, 0.1])
    args = ap.parse_args()
    cfg = resolve_cfg(args)
    setup_logging()
    set_seed(cfg["seed"])
    backend = args.backend or cfg["dvs"]["backend"]
    if args.command == "ablate":
        res = run_ablation(cfg, args.clips, args.ablate_thresholds)
        jpath = save_json(res, out_path(cfg, "dvs_ablation", "json"))
        cmd = "python -m src.dvs_emulate ablate --clips " + " ".join(args.clips)
        for th in args.ablate_thresholds:
            for label, *_ in ABLATION_SETTINGS:
                v = float(np.mean([r["abs_r_green"] for r in res["rows"] if r["threshold"] == th and r["setting"] == label]))
                register_claim(f"dvs.ablation.th{th}.{label.replace(' ', '_').replace(',', '')}.abs_r_green", v, jpath, cmd,
                               note="mean over listed clips; synthetic clips")
        log.info("wrote %s", jpath)
        return
    if args.command == "smoke":
        from .datasets import load_clip
        clip = load_clip(args.clips[0], cfg)
        face, mask = prepare_face(clip, int(cfg["dvs"]["crop_size"]))
        res = {"clip": args.clips[0], "threshold": cfg["dvs"]["threshold"], "frames": 300}
        for b in ("analytic", "v2e"):
            st = emulate_rgb(face[:300], clip.fs, cfg, b, seed=cfg["seed"])
            res[b] = float(np.mean([s.rate_per_pixel() for s in st]))
            log.info("%s: events per channel %s, rate %.3f ev/px/s", b, [len(s) for s in st], res[b])
        jpath = save_json(res, out_path(cfg, "dvs_backend_smoke", "json"))
        cmd = f"python -m src.dvs_emulate smoke --clips {args.clips[0]}"
        for b in ("analytic", "v2e"):
            register_claim(f"dvs.backend_rate.{b}", res[b], jpath, cmd, note="events/pixel/s, first 300 frames")
        return
    res = run_sweep(cfg, args.clips, backend)
    paths = plot_all(cfg, res, args.clips[0], backend)
    jpath = save_json(res, out_path(cfg, "dvs_sweep", "json"))
    cmd = f"python -m src.dvs_emulate sweep --backend {backend} --clips " + " ".join(args.clips)
    register_claim(f"dvs.{backend}.demo_abs_r_green", abs(res["demo"]["r_net_green"]), jpath, cmd)
    for i, th in enumerate(res["thresholds"]):
        rows = [res["per_clip"][c][i] for c in res["per_clip"]]
        for name, vals in (("rate", [x["events_per_pixel_per_s"] for x in rows]), ("abs_r", [abs(x["r_net_sum"]) for x in rows]),
                           ("abs_r_chance", [abs(x["r_null_green"]) for x in rows if "r_null_green" in x])):
            register_claim(f"dvs.sweep.{backend}.th{th}.{name}", float(np.mean(vals)), jpath, cmd, note=f"mean over {len(rows)} synthetic clips")
    register_claim(f"dvs.sweep.{backend}.n_clips", len(res["per_clip"]), jpath, cmd)
    log.info("wrote %s and %s", jpath, {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
