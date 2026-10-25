"use strict";
/* Validation page. Shared live components (preview overlay, readout, checklist, quality checks) come from shared.js. */
let V = null, L = null, S = null, started = false, pollingV = false, pollingL = false, pollingS = false, lastShownSession = null, studyShown = false;
const REASON = Object.assign({collecting: "CARDION was still collecting data", no_estimate_in_window: "no CARDION estimate near that time"}, REJECT_WHY);
const mmss = (s) => { s = Math.max(0, Math.round(s)); return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); };
const sgn = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);

/* ---------------------------------------------------------------- setup form */
async function loadParticipants() {
  const j = await api("/api/validation/participants");
  if (!$("vParticipant").value) $("vParticipant").value = j.next;
  setHTML("vParticipants", (j.participants || []).map((p) => `<option value="${esc(p.participant)}">${p.sessions} session(s)</option>`).join(""));
  if (!$("vDevice").options.length) {
    $("vDevice").innerHTML = (j.devices || []).map((d) => `<option>${esc(d)}</option>`).join("");
    $("vCondition").innerHTML = (j.conditions || []).map((d) => `<option>${esc(d)}</option>`).join("");
    $("vVisit").value = new Date().toISOString().slice(0, 10);
  }
}
async function loadCams() {
  const j = await api("/api/cameras");
  $("vCam").innerHTML = (j.cameras || []).map((c) => `<option value="${c.index}">${esc(c.name)} (${c.width}×${c.height})</option>`).join("") || '<option value="0">Camera 0</option>';
}
$("vSource").onchange = () => $("vCamRow").classList.toggle("hidden", $("vSource").value !== "camera");

$("vStart").onclick = async () => {
  const src = $("vSource").value === "camera" ? {type: "camera", index: +($("vCam").value || 0)} : {type: "demo", bpm: 72};
  $("vMsg").textContent = "Starting…";
  const j = await api("/api/validation/start", {participant: $("vParticipant").value, reference_device: $("vDevice").value, device_note: $("vDeviceNote").value,
    condition: $("vCondition").value, visit: $("vVisit").value, duration_s: +$("vLength").value, consent_confirmed: $("vConsent").checked, save_video: $("vVideo").checked, source: src});
  $("vMsg").textContent = j.message || "";
  if (j.ok) { started = true; $("rgbImg").src = "/stream/rgb.mjpg?" + Date.now(); setOn($("placeholder"), "hidden", true); $("vRef").focus(); }
};
$("vStop").onclick = async () => { if (confirm("Stop this session now? The data collected so far is kept and the report is built from it.")) await api("/api/validation/stop", {}); };
$("vNew").onclick = () => { $("vParticipant").value = ""; loadParticipants(); window.scrollTo(0, 0); $("vParticipant").focus(); };
$("vFolder").onclick = () => api("/api/validation/open_folder", {});

/* ---------------------------------------------------------------- reference entry */
async function addRef() {
  const v = $("vRef").value.trim();
  if (!v) { $("vRefMsg").textContent = "Type the number shown on the reference monitor first."; return; }
  const j = await api("/api/validation/reference", {action: "add", value: v, ago_s: +$("vAgo").value});
  $("vRefMsg").textContent = j.ok ? `Logged ${v} BPM at ${mmss(j.entry.t)}.` : (j.message || j.error || "Could not log the reading.");
  $("vRefMsg").classList.toggle("err", !j.ok);
  if (j.ok) { $("vRef").value = ""; $("vAgo").value = "0"; pollV(); }
  $("vRef").focus();
}
$("vAdd").onclick = addRef;
$("vRef").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); addRef(); } };
async function editRef(id, cur, t) {
  const v = prompt("Correct BPM value (leave as is to keep):", cur); if (v === null) return;
  const ts = prompt("Time of the reading in seconds from the start of the session (leave as is to keep):", t); if (ts === null) return;
  const j = await api("/api/validation/reference", {action: "edit", id, value: v, t_s: ts});
  if (!j.ok) alert(j.message || j.error || "Could not change the reading.");
  pollV();
}
async function delRef(id) { if (confirm("Delete this reference reading? It stays in the saved files, marked as deleted, and is not used in the results.")) { await api("/api/validation/reference", {action: "delete", id}); pollV(); } }
window.editRef = editRef; window.delRef = delRef;

$("vImport").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  $("vImportMsg").textContent = `Reading ${f.name}…`;
  const up = await api("/api/upload?name=" + encodeURIComponent(f.name), f, true);
  if (!up.ok) { $("vImportMsg").textContent = "Upload failed."; return; }
  const j = await api("/api/validation/import", {path: up.path});
  $("vImportMsg").textContent = j.message || j.error || "";
  pollV();
};

/* ---------------------------------------------------------------- drawing */
function drawChart(v) {
  const c = $("vChart"), g = c.getContext("2d"), W = c.width, H = c.height; g.clearRect(0, 0, W, H);
  const hist = v.history || [], ents = (v.all_entries || []).filter((e) => !e.deleted);
  const tmax = Math.max(v.duration_s || 60, v.elapsed_s || 0, 30), tol = (v.session && v.session.tolerance_s) || 5;
  const vals = hist.filter((r) => r.bpm != null).map((r) => r.bpm).concat(ents.map((e) => e.value));
  let lo = 50, hi = 110; if (vals.length) { lo = Math.max(30, Math.min(...vals) - 8); hi = Math.min(220, Math.max(...vals) + 8); if (hi - lo < 30) { const m = (hi + lo) / 2; lo = m - 15; hi = m + 15; } }
  const X = (t) => 42 + t / tmax * (W - 54), Y = (b) => H - 26 - (b - lo) / (hi - lo) * (H - 40);
  g.font = "11px sans-serif"; g.fillStyle = "#8b97a8"; g.strokeStyle = "#222a35"; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const b = lo + (hi - lo) * i / 4; g.beginPath(); g.moveTo(40, Y(b)); g.lineTo(W - 8, Y(b)); g.stroke(); g.fillText(b.toFixed(0), 6, Y(b) + 4); }
  for (let t = 0; t <= tmax; t += tmax > 120 ? 30 : 15) g.fillText(mmss(t), X(t) - 12, H - 8);
  const warm = (v.session && v.session.min_seconds) || 8;
  g.fillStyle = "rgba(120,40,40,.45)";
  hist.forEach((r, i) => { if (r.t >= warm && r.bpm == null) { const nx = hist[i + 1] ? hist[i + 1].t : r.t + 1; g.fillRect(X(r.t - 0.5), H - 24, Math.max(2, X(nx) - X(r.t)), 8); } });
  g.strokeStyle = "#4da3ff"; g.lineWidth = 2.5; g.beginPath(); let st = false;
  hist.forEach((r) => { if (r.bpm != null) { st ? g.lineTo(X(r.t), Y(r.bpm)) : g.moveTo(X(r.t), Y(r.bpm)); st = true; } else st = false; }); g.stroke();
  ents.forEach((e) => { const x = X(e.t), y = Y(e.value);
    g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(e.t - tol), y); g.lineTo(X(e.t + tol), y); g.stroke();
    g.fillStyle = "#fff"; g.beginPath(); g.moveTo(x, y - 6); g.lineTo(x + 6, y); g.lineTo(x, y + 6); g.lineTo(x - 6, y); g.closePath(); g.fill(); });
}

function statRow(label, value) { return `<div><div class="muted small">${label}</div><div class="statnum">${value}</div></div>`; }

function renderEntries(v) {
  const pairById = {}; (v.entries || []).forEach((e) => { pairById[e.id] = e; });
  const rows = (v.all_entries || []).map((e) => {
    const x = pairById[e.id], p = x && x.pair;
    let card = "—", diff = "—";
    if (e.deleted) { card = "deleted"; }
    else if (p) { card = `${p.cardion_bpm.toFixed(1)} <span class="muted small">(${sgn(p.dt_s, 0)} s)</span>`; diff = `<b>${sgn(p.error, 1)}</b>`; }
    else if (x && x.waiting) card = '<span class="muted">waiting…</span>';
    else if (x) card = `<span class="muted small">no reliable reading near this time (${esc(REASON[x.unpaired_reason] || x.unpaired_reason || "")})</span>`;
    const acts = e.deleted ? "" : `<button class="mini" onclick="editRef(${e.id},${e.value},${e.t})">Edit</button> <button class="mini" onclick="delRef(${e.id})">Delete</button>`;
    return `<tr class="${e.deleted ? "deleted" : ""}"><td>${e.id}</td><td>${mmss(e.t)}${e.source === "live-delayed" ? " <span class='muted small'>(delayed)</span>" : ""}${e.source.startsWith("imported") ? " <span class='muted small'>(file)</span>" : ""}</td><td>${e.value.toFixed(0)}${e.edited ? " <span class='muted small'>edited</span>" : ""}</td><td>${card}</td><td>${diff}</td><td>${acts}</td></tr>`;
  }).join("");
  setHTML("vEntriesBody", rows || '<tr><td colspan="6" class="muted">No reference readings yet.</td></tr>');
}

function renderReport(v) {
  const r = v.report, show = !v.active && r;
  setOn($("reportCard"), "hidden", !show);
  if (!show) return;
  const a = r.accuracy, ra = r.reference_availability, ca = r.cardion_availability;
  setText("vHeadline", r.headline);
  const pc = (x) => x == null ? "—" : (100 * x).toFixed(0) + "%";
  const cell = (k, val) => `<tr><th>${k}</th><td>${val}</td></tr>`;
  const acc = a.n_pairs ? [cell("Paired readings", a.n_pairs), cell("Mean absolute error", a.mae_bpm.toFixed(2) + " BPM"), cell("RMSE", a.rmse_bpm.toFixed(2) + " BPM"), cell("Signed bias (CARDION − reference)", sgn(a.bias_bpm, 2) + " BPM"),
    cell("Median / largest absolute error", a.median_abs_error_bpm.toFixed(2) + " / " + a.max_abs_error_bpm.toFixed(2) + " BPM"), cell("Within 1 / 3 / 5 BPM", pc(a.within_1_bpm) + " / " + pc(a.within_3_bpm) + " / " + pc(a.within_5_bpm))] : [cell("Paired readings", "0")];
  const av = [cell("Reference readings compared", `${ra.n_paired} of ${ra.n_reference_entries} (${pc(ra.paired_share)})`), cell("Time to first reliable CARDION reading", ca.time_to_first_reliable_s == null ? "none" : ca.time_to_first_reliable_s.toFixed(1) + " s"),
    cell("Reliable share of the session (after warm-up)", pc(ca.reliable_share_after_warmup)), cell("Withheld periods", `${ca.withheld_episodes} (longest ${(ca.withheld_episode_max_s || 0).toFixed(0)} s)`),
    cell("Steadiness of the shown number", ca.mean_abs_change_per_second_bpm == null ? "—" : `${ca.mean_abs_change_per_second_bpm.toFixed(2)} BPM change per second on average`)];
  setHTML("vReport", `<table class="kv"><tbody>${acc.join("")}</tbody></table><h4>Availability</h4><table class="kv"><tbody>${av.join("")}</tbody></table>
    <p class="muted small">${esc(r.bland_altman.computed ? "" : "Bland\u2013Altman limits: " + r.bland_altman.reason + ". ")}Accuracy is computed only from readings CARDION judged reliable, so read it together with availability.</p>`);
  const m = v.session, base = `/validation/files/${m.participant}/${m.session_id}/`;
  $("vOpenReport").href = base + "report.html"; $("vPairs").href = base + "pairs.csv"; $("vEst").href = base + "estimates.csv";
  $("vZip").href = `/api/validation/export/${m.participant}/${m.session_id}.zip`;
}

function renderV(v) {
  V = v;
  const s = v.session, active = !!v.active;
  setOn($("setupCard"), "hidden", active);
  setOn($("runCard"), "hidden", !active);
  if (!s) { return; }
  setText("vWho", `${s.participant} · ${s.reference_device} · ${s.condition}`);
  setText("vTimer", mmss(v.elapsed_s)); setText("vDur", mmss(v.duration_s));
  setStyle($("vBar"), "width", Math.min(100, v.elapsed_s / v.duration_s * 100) + "%");
  setHTML("vPrompts", (v.prompts || []).map((p) => `<span class="prompt ${p.state}">${mmss(p.t)} ${p.state === "done" ? "✓ logged" : p.state === "due" ? "— read the monitor now" : p.state === "missed" ? "missed" : ""}</span>`).join(""));
  const due = (v.prompts || []).some((p) => p.state === "due");
  setOn($("runCard"), "due", due);
  const ents = (v.all_entries || []).filter((e) => !e.deleted);
  const last = (v.entries || []).filter((e) => !e.deleted).slice(-1)[0];
  setText("refLatest", last ? last.value.toFixed(0) : "—");
  if (last && last.pair) { setText("refMatch", last.pair.cardion_bpm.toFixed(1)); setText("refDiff", sgn(last.pair.error, 1)); setText("refWhy", `CARDION reading ${Math.abs(last.pair.dt_s).toFixed(0)} s ${last.pair.dt_s < 0 ? "before" : "after"} the reference reading.`); }
  else { setText("refMatch", "—"); setText("refDiff", "—");
    setText("refWhy", !last ? "Enter a reference reading to compare." : last.waiting ? "Waiting for a CARDION reading near that time…" : `Not compared: ${REASON[last.unpaired_reason] || "no reliable CARDION reading within the pairing window"}. This counts as 'not available', not as an error.`); }
  setHTML("vStats", [statRow("Reference readings", v.n_entries), statRow("Valid pairs", v.n_pairs), statRow("Reliable CARDION readings", `${v.n_reliable} of ${v.n_estimates}`),
    statRow("Availability so far", v.reliable_share == null ? "—" : (100 * v.reliable_share).toFixed(0) + "%"), statRow("First reliable reading", v.time_to_first_reliable_s == null ? "not yet" : v.time_to_first_reliable_s.toFixed(0) + " s")].join(""));
  renderEntries(v); drawChart(v); renderReport(v);
}

/* ---------------------------------------------------------------- study overview */
async function loadStudy() {
  const [st, ss] = await Promise.all([api("/api/validation/study"), api("/api/validation/sessions")]);
  const ci = st.participant_mae_ci || {};
  const pc = (x) => x == null ? "—" : (100 * x).toFixed(0) + "%";
  const rows = (st.participant_rows || []).map((r) => `<tr><td>${esc(r.participant)}</td><td>${r.n_pairs}</td><td>${r.mae_bpm.toFixed(2)}</td><td>${sgn(r.bias_bpm, 2)}</td><td>${pc(r.within_5_bpm)}</td></tr>`).join("");
  setHTML("vStudy", `<p><b>${st.n_participants}</b> participant(s), <b>${st.n_sessions}</b> real-camera session(s), <b>${st.n_pairs}</b> paired readings of ${st.n_reference_entries} reference readings (${pc(st.paired_share)} could be compared).
    Average reliable share of session time (after warm-up): ${pc(st.mean_reliable_share_after_warmup)}.</p>
    <p>${st.participant_mean_of_mae == null ? "No paired readings yet. " : `Mean of per-participant MAE: <b>${st.participant_mean_of_mae.toFixed(2)} BPM</b>. `}${ci.computed ? `95% bootstrap CI over participants: ${ci.ci_low.toFixed(2)}–${ci.ci_high.toFixed(2)} BPM.` : esc((ci.reason || "") + ".")}</p>
    ${rows ? `<table><thead><tr><th>Participant</th><th>Pairs</th><th>MAE</th><th>Bias</th><th>Within 5</th></tr></thead><tbody>${rows}</tbody></table>` : ""}`);
  setHTML("vSessions", (ss.sessions || []).slice(-8).reverse().map((x) => `<div class="s"><span>${esc(x.participant)} · ${esc(x.session_id)} · ${esc(x.condition || "")}${x.source_type === "demo" ? " (demo)" : ""} · ${x.n_pairs} pairs${x.mae_bpm != null ? " · MAE " + x.mae_bpm.toFixed(1) : ""}</span>
    <span><a href="/validation/files/${x.participant}/${x.session_id}/report.html" target="_blank" rel="noopener">report</a> · <a href="/api/validation/export/${x.participant}/${x.session_id}.zip">zip</a></span></div>`).join(""));
}
$("vStudyBtn").onclick = loadStudy;

/* ---------------------------------------------------------------- polling */
async function pollL() { if (pollingL) return; pollingL = true; try { L = await api("/api/live"); const act = !!(V && V.active); bpmReading(L); checklist(L); guidance(L); setText("why", L.error || (L.guidance ? L.guidance.message + (L.guidance.explain ? " " + L.guidance.explain : "") : "")); if (L.notice) setText("notice", L.notice); setOn($("notice"), "hidden", !L.notice); if (act && !$("rgbImg").getAttribute("src")) $("rgbImg").src = "/stream/rgb.mjpg?" + Date.now(); } catch (e) { /* offline */ } pollingL = false; }
async function pollV() { if (pollingV) return; pollingV = true; try { renderV(await api("/api/validation/state")); } catch (e) { /* offline */ } pollingV = false; }
async function pollS() { if (pollingS) return; pollingS = true; try { S = await api("/api/state"); checks(S); } catch (e) { /* offline */ } pollingS = false; }
loadParticipants(); loadCams(); $("vSource").onchange(); loadStudy();
setInterval(pollL, 250); setInterval(pollV, 500); setInterval(pollS, 1000); pollL(); pollV(); pollS();
$("vStudyLink").href = "/validation/exports/study_report.html";
