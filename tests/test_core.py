"""Fast unit tests: imports, config overrides, subject-disjoint splits, synthetic pulse recovery, DVS behaviour."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.common import apply_overrides, load_config

MODULES = ["common", "capture", "record", "rppg_classical", "datasets", "dvs_emulate", "events", "rppg_events",
           "snn_model", "ann_model", "train", "evaluate", "efficiency", "stats", "demo_live"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"src.{name}")


def test_config_overrides() -> None:
    cfg = load_config(overrides=["dvs.threshold=0.07", "rppg.hr_band_hz=[0.8, 2.5]", "seed=5"])
    assert cfg["dvs"]["threshold"] == 0.07 and cfg["rppg"]["hr_band_hz"] == [0.8, 2.5] and cfg["seed"] == 5
    assert apply_overrides({"a": {"b": 1}}, ["a.c=2"]) == {"a": {"b": 1, "c": 2}}


def test_split_is_subject_disjoint_and_seeded() -> None:
    from src.datasets import subject_split
    subs = [f"s{i}" for i in range(42)]
    a, b = subject_split(subs, [0.6, 0.2, 0.2], 1), subject_split(subs, [0.6, 0.2, 0.2], 1)
    assert a == b
    assert not (set(a["train"]) & set(a["val"]) or set(a["train"]) & set(a["test"]) or set(a["val"]) & set(a["test"]))
    assert sum(len(v) for v in a.values()) == 42


def test_synthetic_pulse_recovered_by_pos() -> None:
    from src.datasets import make_synthetic_clip
    from src.rppg_classical import estimate, ppg_reference_hr
    cfg = load_config(overrides=["datasets.synthetic.clip_seconds=30"])
    clip = make_synthetic_clip(1, cfg)
    _, _, bpm = estimate(clip.rgb, clip.fs, "pos", cfg)
    _, ref = ppg_reference_hr(clip.ppg, clip.fs, 10.0, 1.0)
    k = min(len(bpm), len(ref))
    assert np.mean(np.abs(bpm[:k] - ref[:k])) < 1.5


def test_dvs_deterministic_and_polarity() -> None:
    from src.dvs_emulate import AnalyticDVS
    t = np.arange(60) / 30.0
    frames = (120 * (1 + 0.1 * np.sin(2 * np.pi * 1.0 * t)))[:, None, None] * np.ones((60, 8, 8))
    a = AnalyticDVS(threshold=0.05, seed=3).emulate(frames, 30.0)
    b = AnalyticDVS(threshold=0.05, seed=3).emulate(frames, 30.0)
    assert len(a) == len(b) > 0 and np.array_equal(a.t, b.t)
    assert np.all(np.diff(a.t) >= 0) and set(np.unique(a.p)) <= {-1, 1}
    assert (a.p > 0).any() and (a.p < 0).any()


def test_dvs_more_events_at_lower_threshold() -> None:
    from src.dvs_emulate import AnalyticDVS
    rng = np.random.default_rng(0)
    frames = 100 + rng.normal(0, 2, (40, 8, 8))
    lo = AnalyticDVS(threshold=0.02, shot_noise_hz=0, leak_hz=0, seed=1).emulate(frames, 30.0)
    hi = AnalyticDVS(threshold=0.2, shot_noise_hz=0, leak_hz=0, seed=1).emulate(frames, 30.0)
    assert len(lo) > len(hi)


def test_stats() -> None:
    from src.stats import bootstrap_ci, hr_metrics, lins_ccc
    ref = np.linspace(60, 100, 50)
    assert hr_metrics(ref, ref)["mae"] == 0 and lins_ccc(ref, ref) == pytest.approx(1.0)
    mean, lo, hi = bootstrap_ci(np.arange(20.0), n_boot=200)
    assert lo <= mean <= hi


def test_snn_ann_match_parameters() -> None:
    from src.ann_model import AnnPulseNet
    from src.snn_model import SpikingPulseNet
    assert sum(p.numel() for p in AnnPulseNet().parameters()) == sum(p.numel() for p in SpikingPulseNet().parameters())

def test_paper_placeholders_fill_and_reject_unknown() -> None:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("build_paper", Path(__file__).resolve().parent.parent / "scripts" / "build_paper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    claims = {"a.b": {"value": 3.14159}, "c.d": {"value": 2.0}}
    assert mod.fill("x={{c:a.b|.2f}} r={{r:a.b/c.d|.3f}}", claims) == "x=3.14 r=1.571"
    with pytest.raises(KeyError):
        mod.fill("{{c:missing.key}}", claims)
