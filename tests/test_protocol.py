"""Protocol tests: protected test split, subject-disjoint splits on real data (if present), reference-window validity, SHS picking."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from src.common import REPO_ROOT, load_config


def test_final_test_split_is_protected() -> None:
    from src.evaluate import run_single
    cfg = load_config()
    args = argparse.Namespace(dataset="mcd", split="test", final=False, models=[], thresholds=[0.02], max_clips=2, tag="x", reference="rgb_pos",
                              overrides=[])
    with pytest.raises(SystemExit) as e:
        run_single(args, cfg)
    assert "protected" in str(e.value)


@pytest.mark.skipif(not (REPO_ROOT / "data" / "splits" / "mcd_seed1337.json").exists(), reason="MCD manifest not present")
def test_mcd_split_manifest_subject_disjoint_and_complete() -> None:
    import json
    m = json.loads((REPO_ROOT / "data" / "splits" / "mcd_seed1337.json").read_text())
    s = m["split"]
    sets = {k: set(v) for k, v in s.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    assert sum(len(v) for v in s.values()) == m["n_subjects"] == 60
    assert m["subject_disjoint"]


def test_clip_specs_keep_a_subject_in_one_split() -> None:
    from src.datasets import clip_specs, subject_of
    cfg = load_config()
    try:
        tr, va, te = (clip_specs("mcd", sp, cfg) for sp in ("train", "val", "test"))
    except SystemExit:
        pytest.skip("MCD data not present")
    if not tr:
        pytest.skip("MCD data not present")
    subj = lambda specs: {subject_of(s) for s in specs}  # noqa: E731
    assert not (subj(tr) & subj(va) or subj(tr) & subj(te) or subj(va) & subj(te))
    assert all(len([s for s in tr if subject_of(s) == u]) == 2 for u in subj(tr))        # rest + post-exercise clip together


def test_reference_windows_flag_corrupt_ppg() -> None:
    from src.rppg_classical import reference_windows
    cfg = load_config()
    fs = 30.0
    t = np.arange(int(30 * fs)) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t) + 0.02 * np.random.default_rng(0).normal(size=len(t))
    bpm, valid = reference_windows(ppg, fs, cfg)
    assert valid.all() and abs(np.median(bpm) - 72) < 2
    bad = np.random.default_rng(1).normal(size=len(t))                                       # pure noise = no pulse
    assert not reference_windows(bad, fs, cfg)[1].all()


def test_shs_reduces_octave_error_when_fundamental_is_present() -> None:
    from src.rppg_classical import fft_hr
    fs, t = 30.0, np.arange(300) / 30.0
    f0 = 0.85                                                   # 51 BPM; waveform dominated by its 2nd harmonic
    x = 0.8 * np.sin(2 * np.pi * f0 * t) + 1.0 * np.sin(2 * np.pi * 2 * f0 * t)           # fundamental present but weaker than the 2nd harmonic
    assert fft_hr(x, fs, (0.75, 3.0), "peak")[0] > 90          # plain peak picking locks to 2x
    assert abs(fft_hr(x, fs, (0.75, 3.0), "shs")[0] - 51) < 3  # sub-harmonic summation recovers the fundamental
