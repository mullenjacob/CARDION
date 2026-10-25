"""App tests: quality gating, streaming pipeline on a known-pulse simulation, streaming == batch network, source error handling."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.common import load_config


def test_streaming_network_equals_batch() -> None:
    from src.ann_model import AnnPulseNet
    from src.snn_model import SpikingPulseNet
    torch.manual_seed(0)
    for cls in (SpikingPulseNet, AnnPulseNet):
        m = cls()
        x = (torch.rand(1, 20, 3, 32, 32) < 0.1).float() - (torch.rand(1, 20, 3, 32, 32) < 0.1).float()
        y = m(x)
        m.reset_state(1)
        z = torch.stack([m.step(x[:, t]) for t in range(20)], 1)
        assert torch.allclose(y, z)


def test_quality_gates() -> None:
    from src.cardion_app.quality import QualityParams, assess
    p = QualityParams()
    env = {"seconds": 12, "face_fraction": 1.0, "motion": 0.01, "luma": 120, "luma_var": 0.01}
    good = {k: {"bpm": 70.0 + d, "peak_ratio": 8.0} for k, d in (("rgb_green", 0.5), ("rgb_pos", -0.5), ("event_green", 1.0))}
    a = assess(good, env, p)
    assert a.status == "ok" and abs(a.bpm - 70.3) < 1
    assert assess(good, {**env, "seconds": 2}, p).bpm is None
    assert assess(good, {**env, "motion": 0.5}, p).status == "motion"
    assert assess({k: {"bpm": 70.0, "peak_ratio": 0.5} for k in good}, env, p).status == "weak_signal"


@pytest.mark.parametrize("bpm", [58.0, 96.0])
def test_pipeline_recovers_known_pulse(bpm: float) -> None:
    from src.cardion_app.pipeline import Pipeline
    from src.datasets import make_synthetic_clip
    cfg = load_config(overrides=["datasets.synthetic.size=128", "datasets.synthetic.clip_seconds=22", "datasets.synthetic.noise_sigma_range=[1.0,1.0]"])
    clip = make_synthetic_clip(2, cfg, hr0=bpm)
    h, w = clip.face.shape[1:3]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((((xx - w / 2) / (w * 0.29)) ** 2 + ((yy - h / 2) / (h * 0.37)) ** 2) <= 1).astype(np.uint8)
    pl = Pipeline(cfg, models=None, use_landmarks=False)
    a = None
    for i, f in enumerate(clip.face):
        pl.push(f, i / 30.0, (mask, (w / 2, h / 2, 0.9 * w)))
        if i % 30 == 29:
            a = pl.estimate()
    assert a is not None and a.status == "ok", a
    assert abs(a.bpm - bpm) < 3.0


def test_sources_report_errors_not_crash(tmp_path) -> None:
    from src.cardion_app.sources import SourceError, VideoSource
    with pytest.raises(SourceError):
        VideoSource(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"nope")
    with pytest.raises(SourceError):
        VideoSource(bad)


def test_engine_rejects_unknown_source() -> None:
    from src.cardion_app.engine import Engine
    e = Engine(load_config(), fast=True)
    r = e.start({"type": "nonsense"})
    assert not r["ok"] and "Unknown source" in r["message"]
    assert e.state()["error"]
