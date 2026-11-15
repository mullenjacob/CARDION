"""Validation mode: the maths (pairing, metrics, availability, refusal to over-claim), reference-entry handling, imports, and a full demo session."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import pytest

from src.cardion_app import validation_metrics as vm
from src.cardion_app.validation import ValidationError, ValidationManager, normalise_participant


def rows_const(bpm: float, n: int = 60, ok_from: int = 10, status_after: str = "ok") -> list[dict]:
    """One-per-second rows: collecting for ``ok_from`` seconds, then reliable at ``bpm``."""
    out = []
    for i in range(n):
        good = i >= ok_from
        out.append({"t": float(i), "status": status_after if good else "collecting", "bpm": bpm if good and status_after == "ok" else None, "rgb_pos": bpm + 1.0})
    return out


def entry(i: int, t: float, v: float, deleted: bool = False) -> dict:
    return {"id": i, "t": t, "value": v, "source": "live", "deleted": deleted}


# ---------------------------------------------------------------------------- pairing and metrics
def test_pairing_uses_nearest_reliable_reading_within_tolerance():
    rows = rows_const(72.0)
    pairs, unp = vm.pair_readings([entry(1, 30.2, 71.0), entry(2, 3.0, 70.0)], rows, tol_s=5.0)
    assert len(pairs) == 1 and pairs[0]["cardion_t"] == 30.0 and pairs[0]["error"] == pytest.approx(1.0)
    assert unp[0]["entry_id"] == 2 and unp[0]["reason"] == "collecting"      # before the first reliable reading: not compared, and says why


def test_unpaired_entries_are_not_in_error_statistics_but_are_counted():
    rows = rows_const(72.0, n=40)
    ents = [entry(1, 20.0, 72.0), entry(2, 38.0, 80.0), entry(3, 90.0, 70.0)]
    rep = vm.session_report(ents, rows)
    assert rep["accuracy"]["n_pairs"] == 2
    assert rep["reference_availability"]["n_reference_entries"] == 3 and rep["reference_availability"]["n_paired"] == 2
    assert rep["reference_availability"]["unpaired_reasons"] == {"no_estimate_in_window": 1}
    assert "2 of 3" in vm.headline(rep)                                      # availability always sits next to accuracy


def test_metrics_values():
    pairs = [{"error": e, "abs_error": abs(e), "ref_bpm": 70.0 + i, "cardion_bpm": 70.0 + i + e, "dt_s": 0.0} for i, e in enumerate([0.0, 1.0, -2.0, 4.0, -6.0])]
    m = vm.accuracy_metrics(pairs)
    assert m["n_pairs"] == 5 and m["mae_bpm"] == pytest.approx(2.6) and m["bias_bpm"] == pytest.approx(-0.6)
    assert m["rmse_bpm"] == pytest.approx(np.sqrt((0 + 1 + 4 + 16 + 36) / 5)) and m["median_abs_error_bpm"] == 2.0 and m["max_abs_error_bpm"] == 6.0
    assert m["within_1_bpm"] == pytest.approx(0.4) and m["within_3_bpm"] == pytest.approx(0.6) and m["within_5_bpm"] == pytest.approx(0.8)
    assert m["pearson_r"] is None                                            # fewer than 10 pairs: no correlation is reported


def test_no_reference_readings_gives_no_accuracy_claim():
    rep = vm.session_report([], rows_const(72.0))
    assert rep["accuracy"] == {"n_pairs": 0} and "cannot be assessed" in vm.headline(rep)


def test_deleted_entries_are_excluded():
    rep = vm.session_report([entry(1, 30.0, 72.0), entry(2, 31.0, 100.0, deleted=True)], rows_const(72.0))
    assert rep["accuracy"]["n_pairs"] == 1 and rep["accuracy"]["mae_bpm"] == 0.0


def test_difficult_stretches_count_against_availability_not_hidden():
    rows = rows_const(72.0, n=60)
    for r in rows[30:45]:
        r["status"], r["bpm"] = "weak_signal", None
    av = vm.availability(rows, min_seconds=8.0)
    assert av["time_to_first_reliable_s"] == 10.0 and av["withheld_episodes"] == 1 and av["withheld_episode_max_s"] == pytest.approx(15.0)
    assert av["reliable_share_after_warmup"] == pytest.approx((60 - 10 - 15) / (60 - 8))        # rows at t=8 and 9 still "collecting" count against availability
    assert av["withheld_reasons_after_warmup"] == {"collecting": 2, "weak_signal": 15}
    assert av["max_change_per_second_bpm"] == 0.0 and av["mean_abs_change_per_second_bpm"] == 0.0


def test_tolerance_sensitivity_and_ungated_context():
    rows = rows_const(72.0)
    ents = [entry(1, 30.0, 72.0), entry(2, 5.0, 70.0)]          # the second is only reachable with a wide window (first reliable row is t=10)
    sens = {s["tolerance_s"]: s for s in vm.sensitivity_to_tolerance(ents, rows)}
    assert sens[3.0]["n_pairs"] == 1 and sens[5.0]["n_pairs"] == 2 and sens[10.0]["n_pairs"] == 2
    ung = vm.ungated_context(ents, rows)
    assert ung["rgb_pos"]["n"] == 2 and ung["rgb_pos"]["bias_bpm"] == pytest.approx(2.0)   # rgb_pos says 73 everywhere (gate or no gate): errors +1 and +3


def test_bland_altman_and_ci_refuse_small_samples():
    pairs = [{"error": 1.0}] * 5
    assert vm.bland_altman(pairs)["computed"] is False
    assert vm.bootstrap_ci([1.0, 2.0, 3.0])["computed"] is False
    ci = vm.bootstrap_ci(list(np.linspace(1, 3, 12)), n_boot=500)
    assert ci["computed"] and ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert vm.bland_altman([{"error": float(e)} for e in np.linspace(-2, 2, 30)])["computed"] is True


def test_study_counts_participants_not_readings():
    def sess(p, err):
        return {"participant": p, "n_reference_entries": 4, "reliable_share_after_warmup": 0.8,
                "pairs": [{"error": err, "abs_error": abs(err), "ref_bpm": 70.0, "cardion_bpm": 70.0 + err, "dt_s": 0.0, "ref_t": 1.0, "cardion_t": 1.0, "entry_id": i} for i in range(4)]}
    s = vm.study_summary([sess("P001", 1.0), sess("P001", 1.0), sess("P002", 3.0)])
    assert s["n_participants"] == 2 and s["n_sessions"] == 3 and s["n_pairs"] == 12
    assert s["participant_mean_of_mae"] == pytest.approx(2.0)               # each person counts once: (1 + 3) / 2, not weighted by readings
    assert s["participant_mae_ci"]["computed"] is False


# ---------------------------------------------------------------------------- the manager with a stand-in engine
class FakeEngine:
    """Just enough of Engine for the recorder: no camera, rows are injected by the test."""

    def __init__(self) -> None:
        self.cfg = {"app": {}}
        self.listeners: list = []
        self.wall_t0 = None
        self.thread = None
        self.error = None
        self.message = ""
        self.source_info = {"label": "fake"}
        self.true_bpm = None
        self.pipeline = None
        self.video_sink = None
        self.stopped = False

    def start(self, spec):
        self.wall_t0 = time.time()
        self.thread = type("T", (), {"is_alive": staticmethod(lambda: True)})()
        return {"ok": True, "message": "started"}

    def stop(self):
        self.stopped = True
        self.thread = None


def make_manager(tmp_path) -> tuple[ValidationManager, FakeEngine]:
    eng = FakeEngine()
    return ValidationManager(eng, root=tmp_path / "val"), eng


def start_params(**kw) -> dict:
    return {"participant": "Participant 001", "reference_device": "Apple Watch", "condition": "Resting, seated", "duration_s": 60, "consent_confirmed": True,
            "source": {"type": "camera", "index": 0}, **kw}


def feed(mgr: ValidationManager, eng: FakeEngine, bpm: float, seconds: int) -> None:
    for i in range(seconds):
        r = {"t": float(i), "wall": eng.wall_t0 + i, "status": "ok" if i >= 10 else "collecting", "bpm": bpm if i >= 10 else None, "rgb_pos": bpm}
        mgr._on_row(r)


def test_participant_codes():
    assert normalise_participant("Participant 001") == "P001" and normalise_participant("p7") == "P007" and normalise_participant("12") == "P012"
    assert normalise_participant("subj_A-3") == "subj_A-3"
    for bad in ("", "Jane Doe", "a/b", "x" * 40):
        with pytest.raises(ValidationError):
            normalise_participant(bad)


def test_start_requires_consent_and_valid_settings(tmp_path):
    mgr, _ = make_manager(tmp_path)
    with pytest.raises(ValidationError, match="agreed"):
        mgr.start(start_params(consent_confirmed=False))
    with pytest.raises(ValidationError, match="between 20 s"):
        mgr.start(start_params(duration_s=5))
    with pytest.raises(ValidationError, match="anonymous"):
        mgr.start(start_params(participant="Jane Doe"))


def test_full_session_files_edit_trail_and_report(tmp_path):
    mgr, eng = make_manager(tmp_path)
    mgr.start(start_params())
    eng.wall_t0 -= 30                                           # pretend 30 s have passed
    feed(mgr, eng, 72.0, 30)
    r1 = mgr.add_reference(72, ago_s=0)
    assert r1["entry"]["source"] == "live" and 29.0 < r1["entry"]["t"] < 31.0
    mgr.add_reference(70, ago_s=15)                              # read 15 s earlier
    with pytest.raises(ValidationError, match="not a valid heart rate"):
        mgr.add_reference("abc")
    with pytest.raises(ValidationError):
        mgr.add_reference(500)
    with pytest.raises(ValidationError, match="future"):
        mgr.add_reference(70, t_s=200)
    with pytest.raises(ValidationError, match="between 0 and"):
        mgr.add_reference(70, ago_s=500)
    mgr.edit_reference(2, value=73, note="re-read the watch")
    mgr.delete_reference(1)
    snap = mgr.snapshot()
    assert snap["n_entries"] == 1 and snap["active"]
    mgr.stop("completed")
    d = next((tmp_path / "val" / "P001").glob("S*"))
    for name in ("session.json", "estimates.csv", "reference_readings.csv", "audit.jsonl", "pairs.csv", "report.json", "report.html", "report.png"):
        assert (d / name).exists(), name
    rows = list(csv.DictReader(open(d / "reference_readings.csv", encoding="utf-8")))
    assert rows[0]["deleted"] == "True" and rows[1]["edited"] == "True" and rows[1]["original_bpm"] == "70.0" and rows[1]["bpm"] == "73.0"
    audit = [json.loads(x)["action"] for x in open(d / "audit.jsonl", encoding="utf-8")]
    assert audit[0] == "start" and "edit_reference" in audit and "delete_reference" in audit and audit[-1] == "stop"
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    assert rep["accuracy"]["n_pairs"] == 1 and rep["accuracy"]["mae_bpm"] == pytest.approx(1.0)       # only the edited entry (73 vs CARDION 72) counts
    assert "Apple Watch" in (d / "report.html").read_text(encoding="utf-8") and eng.stopped


def test_reopen_session_and_add_later_reading(tmp_path):
    mgr, eng = make_manager(tmp_path)
    mgr.start(start_params())
    eng.wall_t0 -= 40
    feed(mgr, eng, 70.0, 40)
    mgr.stop("stopped by operator")
    sid = mgr.s.meta["session_id"]
    mgr2 = ValidationManager(FakeEngine(), root=tmp_path / "val")
    mgr2.load("P001", sid)
    assert mgr2.add_reference(71, t_s=25)["ok"]                  # typed afterwards, with an explicit time
    assert mgr2.s.report["accuracy"]["n_pairs"] == 1 and mgr2.s.report["accuracy"]["bias_bpm"] == pytest.approx(-1.0)
    with pytest.raises(ValidationError, match="outside the session"):
        mgr2.edit_reference(1, t_s=999)


def test_import_csv_and_apple_health(tmp_path):
    mgr, eng = make_manager(tmp_path)
    mgr.start(start_params())
    eng.wall_t0 -= 50
    feed(mgr, eng, 72.0, 50)
    t0 = eng.wall_t0
    import datetime as dt
    csvf = tmp_path / "ref.csv"
    csvf.write_text("time_s,bpm\n20,72\n30,73\n500,70\n35,999\n", encoding="utf-8")
    res = mgr.import_reference(str(csvf))
    assert res["added"] == 2 and res["skipped"] == 2               # outside the session and an impossible value are ignored
    fmt = lambda s: dt.datetime.fromtimestamp(t0 + s).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    xml = tmp_path / "export.xml"
    xml.write_text(f'<?xml version="1.0"?><HealthData><Record type="HKQuantityTypeIdentifierHeartRate" value="71" endDate="{fmt(40)}" startDate="{fmt(40)}"/>'
                   f'<Record type="HKQuantityTypeIdentifierStepCount" value="9" endDate="{fmt(41)}" startDate="{fmt(41)}"/>'
                   f'<Record type="HKQuantityTypeIdentifierHeartRate" value="70" endDate="{fmt(5000)}" startDate="{fmt(5000)}"/></HealthData>', encoding="utf-8")
    assert mgr.import_reference(str(xml))["added"] == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="needs a 'bpm' column"):
        mgr.import_reference(str(bad))
    mgr.stop("completed")
    assert mgr.s.report["accuracy"]["n_pairs"] == 3


def test_interrupted_session_is_saved_and_flagged(tmp_path):
    mgr, eng = make_manager(tmp_path)
    mgr.start(start_params())
    eng.wall_t0 -= 20
    feed(mgr, eng, 72.0, 20)
    mgr.add_reference(72)
    mgr.stop("interrupted: The camera stopped delivering frames")
    assert mgr.s.meta["state"] == "interrupted" and mgr.s.report["accuracy"]["n_pairs"] == 1 and not mgr.active


def test_second_start_while_active_is_refused_and_demo_excluded_from_study(tmp_path):
    mgr, eng = make_manager(tmp_path)
    mgr.start(start_params(source={"type": "demo"}))
    with pytest.raises(ValidationError, match="already running"):
        mgr.start(start_params(participant="P002"))
    eng.wall_t0 -= 20
    feed(mgr, eng, 72.0, 20)
    mgr.stop("completed")
    assert mgr.study()["n_sessions"] == 0 and mgr.study()["excluded_non_camera_sessions"] == 1


# ---------------------------------------------------------------------------- a real session on the simulated face (slow: real time)
def test_demo_session_end_to_end(tmp_path):
    from src.cardion_app.engine import Engine
    from src.common import load_config
    eng = Engine(load_config(), use_landmarks=False)
    mgr = ValidationManager(eng, root=tmp_path / "val")
    mgr.start(start_params(duration_s=30, source={"type": "demo", "bpm": 72}, prompt_times=[22]))
    try:
        for _ in range(60):
            time.sleep(0.5)
            if eng.wall_t0 and time.time() - eng.wall_t0 >= 21:
                break
        mgr.add_reference(72, ago_s=0)                           # the known truth of the simulated face
        for _ in range(60):
            time.sleep(0.5)
            if not mgr.active and mgr.s.report:
                break
    finally:
        if mgr.active:
            mgr.stop("test cleanup")
    rep = mgr.s.report
    assert mgr.s.meta["state"] == "completed", mgr.s.meta
    assert rep["accuracy"]["n_pairs"] == 1 and rep["accuracy"]["mae_bpm"] <= 3.0
    assert rep["cardion_availability"]["time_to_first_reliable_s"] is not None
