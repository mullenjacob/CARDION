"use strict";
/* Measurement page. Shared helpers and live components are in shared.js. Two polls: /api/live (a few hundred bytes, 4x per second:
   headline, positioning, progress) and /api/state (everything, only when a new one-per-second estimate exists or every 2 s). */
const COLORS = {rgb_green:"#6fd08c", rgb_pos:"#4da3ff", rgb_chrom:"#9b8cff", event_green:"#ffb454", event_pos:"#ff7a59", snn:"#ff4d6d", ann:"#d96bff"};
const NAMES = {rgb_green:"RGB green", rgb_pos:"RGB POS", rgb_chrom:"RGB CHROM", event_green:"Event green (classical)", event_pos:"Event POS (classical)", snn:"Spiking NN (SNN)", ann:"Matched ANN"};
const GROUP = {rgb_green:"RGB", rgb_pos:"RGB", rgb_chrom:"RGB", event_green:"events", event_pos:"events", snn:"events", ann:"events"};
let srcType = "camera", uploadedPath = null, S = null, L = null, lastServerMsg = null, lastRev = -1, lastFull = 0, pollingLive = false, pollingFull = false;

function setSrc(t) {
  srcType = t;
  document.querySelectorAll("#srcSeg button").forEach(b => b.classList.toggle("on", b.dataset.src === t));
  ["camera", "video", "demo"].forEach(k => $("panel-" + k).classList.toggle("hidden", k !== t));
  if (t === "camera" && !$("camSel").options.length) scanCams();
}
document.querySelectorAll("#srcSeg button").forEach(b => b.onclick = () => setSrc(b.dataset.src));

async function scanCams() {
  $("camMsg").textContent = "Looking for cameras…";
  const j = await api("/api/cameras");
  const sel = $("camSel"); sel.innerHTML = "";
  (j.cameras || []).forEach(c => { const o = document.createElement("option"); o.value = c.index; o.textContent = `${c.name} (${c.width}×${c.height})`; sel.appendChild(o); });
  $("camMsg").textContent = (j.cameras || []).length ? "" : "No camera found — connect one, allow camera access in Windows Settings, or try the demo / a video file.";
}
$("camScan").onclick = scanCams;
$("camTest").onclick = async () => {
  if (S && S.camera_check && S.camera_check.running) { await api("/api/cameracheck", {cancel: true}); return; }
  const idx = $("camSel").options.length ? +$("camSel").value : 0;
  $("camReport").classList.remove("hidden"); $("camReport").textContent = "Starting the camera check...";
  const j = await api("/api/cameracheck", {index: idx});
  if (!j.ok) $("camReport").textContent = j.message || j.error || "Could not start the camera check.";
};
function camReport(s) {
  const c = s.camera_check, box = $("camReport"), btn = $("camTest");
  if (!c || (!c.running && !c.done)) { setText("camTest", "Test my camera"); return; }
  setText("camTest", c.running ? "Stop camera test" : "Test my camera");
  const rows = (c.steps || []).map(st => `<li><span class="dot ${st.status === "PASS" ? "ok" : st.status === "WARN" ? "warn" : "bad"}"></span><span><b>${st.step}</b> - ${String(st.detail).split("\n")[0]}</span></li>`).join("");
  const head = c.running ? `Testing the camera for about ${c.seconds || 20} s - sit still, face the camera in good light...` : "Camera test finished (no video or images were saved or uploaded):";
  box.classList.remove("hidden"); setHTML("camReport", `<b>${head}</b><ul>${rows}</ul>`);
}

$("vidFile").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  $("vidMsg").textContent = `Uploading ${f.name}…`;
  const j = await api("/api/upload?name=" + encodeURIComponent(f.name), f, true);
  if (j.ok) { uploadedPath = j.path; $("vidMsg").textContent = `${f.name} ready`; } else { $("vidMsg").textContent = "Upload failed: " + (j.error || ""); }
};

$("startBtn").onclick = async () => {
  let spec = {type: srcType};
  if (srcType === "camera") { if (!$("camSel").options.length) await scanCams(); if (!$("camSel").options.length) { $("msg").textContent = "No camera available."; return; } spec.index = +$("camSel").value; }
  if (srcType === "video") { if (!uploadedPath) { $("msg").textContent = "Choose a video file first."; return; } spec.path = uploadedPath; spec.speed = +$("speedSel").value; }
  if (srcType === "demo") spec.bpm = +$("demoBpm").value;
  $("msg").textContent = "Starting…"; $("placeholder").classList.add("hidden");
  const j = await api("/api/source", spec);
  $("msg").textContent = j.message || (j.error || "");
  if (j.ok) { $("rgbImg").src = "/stream/rgb.mjpg?" + Date.now(); $("evImg").src = "/stream/events.mjpg?" + Date.now(); pollFull(true); }
  else $("placeholder").classList.remove("hidden");
};
$("evToggle").onchange = async () => { const j = await api("/api/events", {enabled: $("evToggle").checked}); $("msg").textContent = j.message || ""; };
$("stopBtn").onclick = async () => { await api("/api/stop", {}); $("msg").textContent = "Stopped."; };

$("recBtn").onclick = async () => {
  const rec = L && L.recording;
  const j = await api("/api/record", {action: rec ? "stop" : "start", video: $("recVideo").checked});
  $("msg").textContent = j.message || j.error || "";
  loadSessions();
};
$("openBtn").onclick = () => api("/api/open_sessions", {});

async function loadSessions() {
  const j = await api("/api/sessions");
  const html = (j.sessions || []).slice(0, 6).map(s => {
    const m = s.summary || {};
    return `<div class="s"><span>${s.id.replace("session_", "")} &middot; ${m.reliable_estimates ?? 0}/${m.estimates_total ?? 0} reliable${m.mean_reliable_bpm ? " &middot; mean " + m.mean_reliable_bpm.toFixed(1) + " BPM" : ""}</span><a href="/api/session/${s.id}.zip">download .zip</a></div>`; }).join("");
  setHTML("sessions", html || '<span class="muted">No saved sessions yet.</span>');
}

function drawWave(s) {
  const c = $("wave"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H); g.strokeStyle = "#222a35"; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(0, H * i / 4); g.lineTo(W, H * i / 4); g.stroke(); }
  const wf = s.waveforms || {}; const leg = [];
  const show = ["rgb_pos", "event_green", "snn"].concat(Object.keys(wf).filter(k => !["rgb_pos", "event_green", "snn"].includes(k)));
  show.forEach(k => {
    const w = wf[k]; if (!w || w.length < 5) return;
    g.strokeStyle = COLORS[k]; g.lineWidth = k === "snn" || k === "rgb_pos" ? 2 : 1; g.globalAlpha = k === "snn" || k === "rgb_pos" || k === "event_green" ? 1 : 0.45;
    g.beginPath(); w.forEach((v, i) => { const x = i / (w.length - 1) * W, y = H / 2 - Math.max(-3, Math.min(3, v)) / 3 * (H / 2 - 8); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    leg.push(`<span><i style="background:${COLORS[k]}"></i>${NAMES[k]}</span>`);
  });
  g.globalAlpha = 1;
  setHTML("legend", leg.join("") || '<span>waiting for enough data…</span>');
}

function drawHist(s) {
  const c = $("hist"), g = c.getContext("2d"), W = c.width, H = c.height; g.clearRect(0, 0, W, H);
  const h = s.history || []; if (h.length < 2) { g.fillStyle = "#8b97a8"; g.font = "14px sans-serif"; g.fillText("History appears after the first estimates", 14, H / 2); return; }
  const lo = 40, hi = 180, X = (i) => 40 + i / Math.max(h.length - 1, 1) * (W - 50), Y = (v) => H - 20 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (H - 34);
  g.strokeStyle = "#222a35"; g.fillStyle = "#8b97a8"; g.font = "11px sans-serif";
  [60, 90, 120, 150].forEach(v => { g.beginPath(); g.moveTo(38, Y(v)); g.lineTo(W, Y(v)); g.stroke(); g.fillText(v, 8, Y(v) + 4); });
  Object.keys(COLORS).forEach(k => { g.strokeStyle = COLORS[k]; g.globalAlpha = 0.5; g.lineWidth = 1; g.beginPath(); let st = false;
    h.forEach((r, i) => { const v = r[k]; if (v === null || v === undefined) { st = false; return; } st ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); st = true; }); g.stroke(); });
  g.globalAlpha = 1; g.strokeStyle = "#fff"; g.lineWidth = 3; g.beginPath(); let st = false;
  h.forEach((r, i) => { if (r.status === "ok" && r.bpm != null) { st ? g.lineTo(X(i), Y(r.bpm)) : g.moveTo(X(i), Y(r.bpm)); st = true; } else st = false; }); g.stroke();
  if (s.true_bpm) { g.setLineDash([6, 5]); g.strokeStyle = "#7ee2a0"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(38, Y(s.true_bpm)); g.lineTo(W, Y(s.true_bpm)); g.stroke(); g.setLineDash([]); }
}

function methods(s) {
  const est = s.estimates || {}, cost = s.costs || {};
  const rows = Object.keys(NAMES).map(k => {
    const e = est[k]; const bpm = e ? e.bpm : null, q = e ? e.quality : null;
    let comp = "—";
    if (k === "snn" && cost.snn) comp = `${fmt(cost.snn.ms_per_s, 1)} ms/s · ${fmt(cost.snn.sops_per_s / 1e3, 0)}k SOPs/s`;
    else if (k === "ann" && cost.ann) comp = `${fmt(cost.ann.ms_per_s, 1)} ms/s · ${fmt(cost.ann.macs_per_s / 1e6, 1)}M MACs/s`;
    else if (k.startsWith("event") && cost.dvs_emulation) comp = `events ${fmt(s.events_per_s, 0)}/s`;
    const disabled = (k === "snn" || k === "ann") && s.models && s.models[k] && !s.models[k].loaded;
    const cls = s.events_enabled === false && GROUP[k] === "events" ? ' class="paused"' : "";
    return `<tr${cls}><td><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${COLORS[k]};margin-right:7px"></i>${NAMES[k]}${disabled ? ' <span class="muted small">(model not found)</span>' : ""}</td>
      <td class="num">${fmt(bpm, 1)}</td><td><span class="bar"><i style="width:${Math.min(100, (q || 0) * 6)}%"></i></span>${q === null ? "" : q.toFixed(1)}</td><td class="muted small">${comp}</td></tr>`;
  }).join("");
  setHTML("methodsBody", rows);
  if (cost.dvs_emulation) setText("costNote", `Shared: face tracking ${fmt(cost.shared_roi.ms_per_s, 1)} ms/s; event emulation ${fmt(cost.dvs_emulation.ms_per_s, 1)} ms/s (software stand-in for the sensor). Energy numbers are operation-count proxies, not measured power.`);
}

/* ---- fast part: everything the eye follows while measuring */
function renderLive(l) {
  L = l;
  bpmReading(l); checklist(l); guidance(l);
  const h = l.headline || {};
  const why = {collecting: `Collecting data… stay still and face the camera.`, no_face: "Move into view, face the camera, and make sure your face is lit.",
    motion: "Hold still for a few seconds.", dark: "Add light (face a window or lamp).", bright: "Reduce glare / bright light on the face.", unstable_light: "Avoid flickering lights and auto-exposure changes.",
    weak_signal: "The pulse is not clearly visible yet; better light and staying still help.", disagree: "Estimators give different answers, so no number is shown.", ok: "Estimate is the average of the methods that agree.", multi_face: "Only one person should be in view; the measurement is paused until then."}[h.status] || "";
  setText("why", l.error || (l.guidance ? (l.guidance.message + (l.guidance.explain ? " " + l.guidance.explain : "")) : why));
  if ($("startBtn").disabled) $("startBtn").disabled = false; if ($("stopBtn").disabled === !!l.running) $("stopBtn").disabled = !l.running;
  setHTML("recBtn", l.recording ? "■ Stop recording" : "● Start recording"); setOn($("recBtn"), "on", l.recording);
  if (l.message !== lastServerMsg) { lastServerMsg = l.message; if (!l.running) $("msg").textContent = l.message; }      // show server messages once, so they do not overwrite local hints
  setOn($("placeholder"), "hidden", l.running || l.frames > 0);
  if (l.running && !$("rgbImg").getAttribute("src")) { $("rgbImg").src = "/stream/rgb.mjpg?" + Date.now(); $("evImg").src = "/stream/events.mjpg?" + Date.now(); }
  if (l.notice) setText("notice", l.notice); setOn($("notice"), "hidden", !l.notice);
  if (l.rev !== lastRev || (!l.running && Date.now() - lastFull > 2000) || Date.now() - lastFull > 2500) pollFull();
}

/* ---- slow part: tables, waveforms, history (only redrawn when a new estimate exists) */
function renderFull(s) {
  S = s;
  if (s.true_bpm) setText("truth", `Ground truth for this simulated face: ${s.true_bpm.toFixed(1)} BPM`); setOn($("truth"), "hidden", !s.true_bpm);
  if (s.running) $("evToggle").checked = s.events_enabled !== false;
  checks(s); stability(s); methods(s); drawWave(s); drawHist(s); camReport(s);
  const m = s.models || {}, p = s.perf || {};
  setHTML("foot", `${s.disclaimer}<br>SNN: ${m.snn && m.snn.loaded ? "loaded" : "not loaded"} · ANN: ${m.ann && m.ann.loaded ? "loaded" : "not loaded"} · processing ${s.fps || 0} fps` +
    (s.running && s.source && s.source.type === "camera" ? ` · camera delivering ${s.camera_fps || 0} fps${p.dropped_frames ? " · " + p.dropped_frames + " frames dropped" : ""}` : "") +
    (m.snn && m.snn.loaded && m.snn.note ? `<br>${m.snn.note}` : ""));
}

async function pollLive() {
  if (pollingLive) return; pollingLive = true;
  try { renderLive(await api("/api/live")); setOn($("msg"), "err", false); }
  catch (e) { $("msg").textContent = "Lost connection to CARDION (is the window still open?)"; }
  pollingLive = false;
}
async function pollFull(force) {
  if (pollingFull) return; pollingFull = true;
  try { const s = await api("/api/state"); lastRev = s.rev; lastFull = Date.now(); renderFull(s); } catch (e) { /* the live poll reports connection loss */ }
  pollingFull = false;
}
setSrc("camera"); loadSessions(); setInterval(pollLive, 250); pollLive(); pollFull();
