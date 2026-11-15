"""Files written by a validation session: pairs.csv, report.json, report.png and a self-contained report.html (and the study-level exports)."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

CAVEATS = [
    "CARDION is a research prototype, not a medical device. These numbers describe agreement with the reference device used here, not clinical accuracy.",
    "A wrist-worn consumer watch is a useful reference but not the same as clinical ECG ground truth; some of any difference may be the reference's own error.",
    "Reference values were read and typed by a person, so their timing is only approximately known (see the pairing window and its sensitivity table).",
    "Many readings from one person are not independent observations. Judge repeatability by the number of participants, not the number of readings.",
]


def _f(v, d=1, unit="") -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}{unit}"


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def make_plot(rep: dict, rows: list[dict], path: Path | None = None) -> bytes:
    """Two panels: heart rate over time (CARDION reliable readings, withheld periods, reference points) and the error of each pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tol = rep["tolerance_s"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.2), gridspec_kw={"height_ratios": [3, 1.6]})
    warm = rep["meta"].get("min_seconds", 8.0)
    t = np.array([r["t"] for r in rows], float) if rows else np.array([])
    ok = np.array([(r.get("status") == "ok" and r.get("bpm") is not None) for r in rows], bool) if rows else np.array([], bool)
    if len(t):
        step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        for r in rows:
            if r["t"] >= warm and not (r.get("status") == "ok" and r.get("bpm") is not None):
                a1.axvspan(r["t"] - step / 2, r["t"] + step / 2, color="#f4b6b4", alpha=0.35, lw=0)
        y = np.array([r["bpm"] if (r.get("status") == "ok" and r.get("bpm") is not None) else np.nan for r in rows], float)
        a1.plot(t, y, "-", color="#1f6fd1", lw=2, label="CARDION (reliable readings only)")
        a1.plot(t[ok], y[ok], ".", color="#1f6fd1", ms=4)
        pos = np.array([r.get("rgb_pos") if r.get("rgb_pos") is not None else np.nan for r in rows], float)
        a1.plot(t, pos, "--", color="#999", lw=1, label="RGB POS alone (no reliability gate)")
    ents = [e for e in rep["entries"] if not e.get("deleted")]
    paired = {p["entry_id"] for p in rep["pairs"]}
    for e in ents:
        a1.errorbar([e["t"]], [e["value"]], xerr=[[tol], [tol]], fmt="D", color="black" if e["id"] in paired else "#b00020", capsize=3, ms=6,
                    label=None)
    a1.plot([], [], "kD", label="reference reading (bars = pairing window)")
    if any(e["id"] not in paired for e in ents):
        a1.plot([], [], "D", color="#b00020", label="reference reading with no reliable CARDION reading in range")
    a1.fill_between([], [], [], color="#f4b6b4", alpha=0.35, label="CARDION withheld a reading")
    a1.set_xlabel("session time (s)")
    a1.set_ylabel("beats per minute")
    a1.legend(fontsize=7, loc="best")
    a1.set_title(f"{rep['meta'].get('participant')} · {rep['meta'].get('reference_device')} · {rep['meta'].get('condition')}", fontsize=10)
    a1.grid(alpha=0.25)
    pairs = rep["pairs"]
    if pairs:
        x = np.arange(len(pairs))
        err = np.array([p["error"] for p in pairs])
        a2.axhspan(-5, 5, color="#d7ecd8", alpha=0.6, lw=0)
        a2.axhspan(-1, 1, color="#a9d7ac", alpha=0.7, lw=0)
        a2.bar(x, err, color=["#1f6fd1" if abs(v) <= 5 else "#b00020" for v in err])
        a2.set_xticks(x)
        a2.set_xticklabels([f"{p['ref_t']:.0f}s\nref {p['ref_bpm']:.0f}" for p in pairs], fontsize=7)
        a2.axhline(0, color="black", lw=0.8)
    else:
        a2.text(0.5, 0.5, "no paired readings", ha="center", va="center", transform=a2.transAxes)
    a2.set_ylabel("CARDION − reference (BPM)")
    a2.set_title("error of each paired reading (green band: within 5 BPM, dark green: within 1 BPM)", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        path.write_bytes(data)
    return data


def _history_text(e: dict) -> str:
    """'value: 70 -> 72; ...' for an entry that was edited."""
    return "; ".join("{}: {} → {}".format(h["field"], h["from"], h["to"]) for h in e.get("history", []))


def _table(rows: list[tuple[str, str]], cls: str = "kv") -> str:
    return f'<table class="{cls}">' + "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def render_html(rep: dict, png: bytes) -> str:
    """A self-contained report page (open in any browser; print to PDF if needed)."""
    m, a, ba = rep["meta"], rep["accuracy"], rep["bland_altman"]
    ra, ca = rep["reference_availability"], rep["cardion_availability"]
    acc = [("Paired readings", str(a.get("n_pairs", 0)))]
    if a.get("n_pairs"):
        acc += [("Mean absolute error (MAE)", _f(a["mae_bpm"], 2, " BPM")), ("Root mean squared error (RMSE)", _f(a["rmse_bpm"], 2, " BPM")),
                ("Signed bias (CARDION − reference)", _f(a["bias_bpm"], 2, " BPM")), ("Median absolute error", _f(a["median_abs_error_bpm"], 2, " BPM")),
                ("Largest absolute error", _f(a["max_abs_error_bpm"], 2, " BPM")), ("Within 1 BPM", _pct(a["within_1_bpm"])), ("Within 3 BPM", _pct(a["within_3_bpm"])),
                ("Within 5 BPM", _pct(a["within_5_bpm"])), ("Mean absolute error as % of reference (MAPE)", _f(a["mape_percent"], 1, " %")),
                ("Average time offset between paired readings", _f(a["mean_abs_pair_offset_s"], 1, " s")),
                ("Reference range in this session", f"{a['ref_range_bpm'][0]:.0f}–{a['ref_range_bpm'][1]:.0f} BPM")]
        acc.append(("Correlation (Pearson r)", _f(a.get("pearson_r"), 2) if a.get("pearson_r") is not None else "not given (needs ≥ 10 pairs and some spread in the reference)"))
    ba_txt = (f"bias {ba['bias_bpm']:+.1f} BPM, 95% limits of agreement {ba['loa_low_bpm']:+.1f} to {ba['loa_high_bpm']:+.1f} BPM ({html.escape(ba['caveat'])})"
              if ba["computed"] else html.escape(ba["reason"]))
    avail = [("Reference readings entered", str(ra["n_reference_entries"])),
             ("... with a reliable CARDION reading within the window", f"{ra['n_paired']} ({_pct(ra['paired_share'])})"),
             ("... without (reason)", html.escape(", ".join(f"{k}: {v}" for k, v in ra["unpaired_reasons"].items()) or "none")),
             ("CARDION: time to first reliable reading", _f(ca.get("time_to_first_reliable_s"), 1, " s")),
             ("CARDION: reliable share, whole session", _pct(ca.get("reliable_share_whole_session"))),
             (f"CARDION: reliable share after the {ca.get('warmup_s', 8):.0f} s warm-up", _pct(ca.get("reliable_share_after_warmup"))),
             ("Withheld periods after the first reading", f"{ca.get('withheld_episodes', 0)} (mean {_f(ca.get('withheld_episode_mean_s'), 1, ' s')}, longest {_f(ca.get('withheld_episode_max_s'), 1, ' s')})"),
             ("Why readings were withheld (after warm-up, seconds)", html.escape(", ".join(f"{k}: {v}" for k, v in ca.get("withheld_reasons_after_warmup", {}).items()) or "none")),
             ("Steadiness of the shown number", f"mean change {_f(ca.get('mean_abs_change_per_second_bpm'), 2, ' BPM')} per second, largest {_f(ca.get('max_change_per_second_bpm'), 1, ' BPM')}; "
                                                f"spread of all reliable readings (SD) {_f(ca.get('reliable_bpm_sd'), 1, ' BPM')}")]
    sens = "".join(f"<tr><td>±{s['tolerance_s']:.0f} s</td><td>{s['n_pairs']} of {s['n_entries']}</td><td>{_f(s['mae_bpm'], 2)}</td><td>{_pct(s['within_5_bpm'])}</td></tr>" for s in rep["tolerance_sensitivity"])
    ung = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['n']}</td><td>{_f(v['mae_bpm'], 2)}</td><td>{_f(v['bias_bpm'], 2)}</td><td>{_pct(v['within_5_bpm'])}</td></tr>" for k, v in rep["ungated_context"].items())
    prs = "".join(f"<tr><td>{p['ref_t']:.1f}</td><td>{p['ref_bpm']:.1f}</td><td>{p['cardion_t']:.1f}</td><td>{p['cardion_bpm']:.1f}</td><td>{p['error']:+.1f}</td></tr>" for p in rep["pairs"])
    unp = "".join(f"<tr><td>{u['ref_t']:.1f}</td><td>{u['ref_bpm']:.1f}</td><td>{html.escape(u['reason'])}</td></tr>" for u in rep["unpaired"])
    ents = "".join(f"<tr><td>{e['id']}</td><td>{e['t']:.1f}</td><td>{e['value']:.0f}</td><td>{html.escape(e['source'])}</td><td>{'deleted' if e.get('deleted') else ('edited' if e.get('edited') else '')}</td>"
                   f"<td>{html.escape(_history_text(e))}</td></tr>" for e in rep["entries"])
    img = base64.b64encode(png).decode("ascii")
    sim = ("<p class='warn'><b>This session used a simulated face (built-in demo), not a person. Its numbers test the software, not people, and are excluded from the study summary.</b></p>"
           if m.get("source_type") == "demo" else "")
    cav = "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CARDION validation report {html.escape(str(m.get('participant')))} {html.escape(str(m.get('session_id')))}</title>
<style>body{{font:15px/1.5 "Segoe UI",system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1b2430}}h1{{font-size:22px}}h2{{font-size:16px;margin-top:26px;border-bottom:1px solid #d5dbe3;padding-bottom:3px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #e4e8ee;padding:5px 8px;text-align:left;vertical-align:top}}table.kv th{{width:46%;font-weight:500;color:#44505f}}
.head{{background:#eef5ff;border:1px solid #b9d3f5;border-radius:8px;padding:10px 14px;font-size:16px}}.warn{{background:#fff3cd;border:1px solid #f0d58a;border-radius:8px;padding:8px 12px}}
img{{max-width:100%}}small,.muted{{color:#5d6a7a}}</style></head><body>
<h1>CARDION accuracy validation report</h1>
<p class="muted">Participant <b>{html.escape(str(m.get('participant')))}</b> &middot; session {html.escape(str(m.get('session_id')))} &middot; reference: <b>{html.escape(str(m.get('reference_device')))}</b> {html.escape(str(m.get('device_note') or ''))}
 &middot; condition: {html.escape(str(m.get('condition')))} &middot; visit: {html.escape(str(m.get('visit')))} &middot; planned length {m.get('duration_s'):.0f} s &middot; source: {html.escape(str(m.get('source_label')))} &middot; ended: {html.escape(str(m.get('end_reason')))}</p>
{sim}<div class="head">{html.escape(rep['headline'])}</div>
<h2>Accuracy (paired readings only)</h2>{_table(acc)}<p class="muted">Bland&ndash;Altman: {ba_txt}</p>
<h2>Availability (always read together with accuracy)</h2>{_table(avail)}
<h2>Figure</h2><img alt="Heart rate over time and error of each pair" src="data:image/png;base64,{img}">
<h2>Does the pairing window matter?</h2><table><tr><th>Window</th><th>Readings paired</th><th>MAE (BPM)</th><th>Within 5 BPM</th></tr>{sens}</table>
<p class="muted">The main result uses ±{rep['tolerance_s']:.0f} s. If these rows differ a lot, the timing of manual entry is limiting the comparison.</p>
<h2>What the reliability gate hides</h2><table><tr><th>Method (no gate)</th><th>Entries compared</th><th>MAE (BPM)</th><th>Bias</th><th>Within 5 BPM</th></tr>{ung}</table>
<p class="muted">The same reference readings compared with each individual method at the nearest second, whether or not CARDION judged the reading reliable. CARDION's displayed number is the average of the agreeing RGB methods (green, POS, CHROM) when it is judged reliable; the event-based, SNN and ANN methods are shown for comparison and do not feed it.</p>
<h2>Paired readings</h2><table><tr><th>Reference time (s)</th><th>Reference BPM</th><th>CARDION time (s)</th><th>CARDION BPM</th><th>Error</th></tr>{prs or '<tr><td colspan=5>none</td></tr>'}</table>
<h2>Reference readings that could not be compared</h2><table><tr><th>Time (s)</th><th>Reference BPM</th><th>Reason</th></tr>{unp or '<tr><td colspan=3>none</td></tr>'}</table>
<h2>All reference entries (with edit history)</h2><table><tr><th>#</th><th>Time (s)</th><th>BPM</th><th>Source</th><th>Flag</th><th>Changes</th></tr>{ents or '<tr><td colspan=6>none</td></tr>'}</table>
<h2>How to read this</h2><ul>{cav}</ul>
<p class="muted">Anonymous code only. No identifying information, face recognition or (unless the operator chose it) video is stored with this session.</p></body></html>"""


def write_report_files(d: Path, rep: dict, rows: list[dict]) -> None:
    """Write pairs.csv, report.json, report.png and report.html next to the session data."""
    with open(d / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["entry_id", "ref_t", "ref_bpm", "cardion_t", "cardion_bpm", "dt_s", "error", "abs_error"])
        wr.writeheader()
        wr.writerows(rep["pairs"])
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    png = make_plot(rep, rows, d / "report.png")
    (d / "report.html").write_text(render_html(rep, png), encoding="utf-8")


def write_study_files(root: Path, summ: dict, sessions: list[dict]) -> None:
    """Write the participant-level study exports (study_summary.json, all_pairs.csv, all_sessions.csv, study_report.html) into ``root/exports``."""
    ex = root / "exports"
    ex.mkdir(parents=True, exist_ok=True)
    (ex / "study_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    with open(ex / "all_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["participant", "session_id", "condition", "visit", "reference_device", "ref_t", "ref_bpm", "cardion_t", "cardion_bpm", "error"])
        for s in sessions:
            for p in s["pairs"]:
                wr.writerow([s["participant"], s["session_id"], s["condition"], s["visit"], s["reference_device"], p["ref_t"], p["ref_bpm"], p["cardion_t"], p["cardion_bpm"], p["error"]])
    with open(ex / "all_sessions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["participant", "session_id", "condition", "visit", "reference_device", "reference_entries", "paired", "reliable_share_after_warmup", "time_to_first_reliable_s"])
        for s in sessions:
            wr.writerow([s["participant"], s["session_id"], s["condition"], s["visit"], s["reference_device"], s["n_reference_entries"], len(s["pairs"]),
                         s.get("reliable_share_after_warmup"), s.get("time_to_first_reliable_s")])
    ci = summ["participant_mae_ci"]
    ci_txt = (f"{summ['participant_mean_of_mae']:.2f} BPM (95% bootstrap CI over {ci['n_participants']} participants: {ci['ci_low']:.2f}–{ci['ci_high']:.2f})" if ci["computed"]
              else html.escape(ci["reason"]))
    rows = "".join(f"<tr><td>{r['participant']}</td><td>{r['n_pairs']}</td><td>{r['mae_bpm']:.2f}</td><td>{r['bias_bpm']:+.2f}</td><td>{100 * r['within_5_bpm']:.0f}%</td></tr>" for r in summ["participant_rows"])
    cond = "".join(f"<tr><td>{html.escape(c)}</td><td>{v['n_participants']}</td><td>{v['n_pairs']}</td><td>{_f(v['participant_mean_of_mae'], 2)}</td><td>{_pct(v['mean_reliable_share_after_warmup'])}</td></tr>"
                   for c, v in summ["by_condition"].items())
    (ex / "study_report.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>CARDION study summary</title>
<style>body{{font:15px/1.5 "Segoe UI",sans-serif;max-width:860px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:5px 8px;text-align:left}}</style></head><body>
<h1>CARDION validation study summary</h1>
<p>{summ['n_participants']} participants, {summ['n_sessions']} real-camera sessions ({summ['excluded_non_camera_sessions']} demo/video sessions excluded), {summ['n_reference_entries']} reference readings, {summ['n_pairs']} paired.
Each participant counts once in the headline: mean of per-participant MAE = {ci_txt}.</p>
<p>Average share of session time with a reliable reading (after warm-up): {_pct(summ['mean_reliable_share_after_warmup'])}. Reference readings that could be compared: {_pct(summ['paired_share'])}.</p>
<h2>Per participant</h2><table><tr><th>Participant</th><th>Pairs</th><th>MAE (BPM)</th><th>Bias</th><th>Within 5</th></tr>{rows}</table>
<h2>By condition</h2><table><tr><th>Condition</th><th>Participants</th><th>Pairs</th><th>Mean of participant MAE</th><th>Reliable share</th></tr>{cond}</table>
<p><small>Informal pilot data unless the protocol in docs/VALIDATION_PROTOCOL.md was followed. Not a medical device evaluation.</small></p></body></html>""", encoding="utf-8")
