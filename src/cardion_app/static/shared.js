"use strict";
/* Helpers and live-view components shared by the measurement page (app.js) and the Validation page (validate.js).
   Every DOM write goes through setText/setHTML/setClass, which skip the write when nothing changed - this keeps the page from re-laying
   itself out several times a second, which is a main cause of a "jumpy" interface. */
const $ = (id) => document.getElementById(id);
const _last = new Map();

async function api(path, body, raw) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"X-CARDION": "1", ...(raw ? {} : {"Content-Type": "application/json"})}, body: raw ? body : JSON.stringify(body)};
  const r = await fetch(path, opt);
  let j = {}; try { j = await r.json(); } catch (e) {}
  if (!r.ok) j.ok = false;
  return j;
}
function fmt(v, d = 0) { return v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d); }
function setText(id, txt) { const el = $(id); if (el && _last.get("t" + id) !== txt) { _last.set("t" + id, txt); el.textContent = txt; } }
function setHTML(id, html) { const el = $(id); if (el && _last.get("h" + id) !== html) { _last.set("h" + id, html); el.innerHTML = html; } }
function setClass(id, cls) { const el = $(id); if (el && el.className !== cls) el.className = cls; }
function setAttr(el, name, val) { val = String(val); if (el.getAttribute(name) !== val) el.setAttribute(name, val); }
function setOn(el, cls, on) { if (el.classList.contains(cls) !== !!on) el.classList.toggle(cls, !!on); }
function setStyle(el, prop, val) { if (el.style[prop] !== val) el.style[prop] = val; }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
function ageText(sec) { return sec < 90 ? Math.round(sec) + " s" : Math.round(sec / 60) + " min"; }

/* ---------------------------------------------------------------- measurement quality (seven separate checks) */
const DIM_ICON = {good: ["✓", "good"], warn: ["!", "warn"], bad: ["✕", "bad"], pending: ["…", ""]};
const DIM_WORD = {good: "OK", warn: "adjust", bad: "problem", pending: "waiting"};
function checks(s) {
  const dims = s.quality_dims || [];
  setHTML("checks", dims.map((d) => { const [ic, cls] = DIM_ICON[d.state] || DIM_ICON.pending;
    return `<li><span class="ic ${cls}" aria-hidden="true">${ic}</span><span><b>${d.label}</b> <span class="muted small">(${DIM_WORD[d.state]})</span><small>${d.detail}</small></span></li>`; }).join(""));
}

/* ---------------------------------------------------------------- positioning overlay on the preview */
const G_ICON = {ok: "✓", warn: "!", bad: "✕", info: "i"};
const G_ARROW = {left: "←", right: "→", up: "↑", down: "↓", closer: "+", farther: "−"};
function setRect(el, b, dead = 0.35) {
  const v = {x: b.x0 * 100, y: b.y0 * 100, width: (b.x1 - b.x0) * 100, height: (b.y1 - b.y0) * 100};
  const p = el._rect;
  if (p && Object.keys(v).every((k) => Math.abs(v[k] - p[k]) < dead)) return;        // landmark noise below ~0.35% of the frame is not shown: a calmer box
  el._rect = v;
  for (const k in v) { setAttr(el, k, v[k].toFixed(2)); setStyle(el, k, v[k].toFixed(2) + "px"); }   // CSS copy lets the box glide between detections
}
function guidance(s) {
  const g = s.guidance, ov = $("overlay"), ban = $("guideBanner"), st = $("gStage"), pr = $("gProg"), ea = $("edgeArrow");
  const mirror = !!(g && g.mirrored);
  setOn($("rgbImg").parentElement, "guided", !!g);
  setOn($("rgbImg"), "mirror", mirror);
  setAttr($("ovMirror"), "transform", mirror ? "translate(100 0) scale(-1 1)" : "");
  if (!g) { [ov, ban, st, pr, ea].forEach((e) => setOn(e, "hidden", true)); return; }
  [ov, ban, st, pr].forEach((e) => setOn(e, "hidden", false));
  setText("gText", g.message); setText("gIcon", G_ICON[g.level] || "");
  setClass("guideBanner", "gbanner " + g.level);
  const reliable = g.stage === "reliable";
  setRect($("ovTarget"), g.target);
  setStyle($("ovTarget"), "display", (g.framing_ok || reliable) ? "none" : "");
  const f = $("ovFace"); if (g.face) { setRect(f, g.face); setStyle(f, "display", ""); } else setStyle(f, "display", "none");
  setAttr(f, "class", g.code === "multi_face" ? "multi" : reliable ? "reliable" : g.framing_ok ? "good" : "");
  const oth = $("ovOthers"), key = JSON.stringify(g.others || []);
  if (_last.get("others") !== key) { _last.set("others", key); oth.innerHTML = ""; (g.others || []).forEach((b) => { const r = document.createElementNS("http://www.w3.org/2000/svg", "rect"); setRect(r, b); r.setAttribute("rx", 2); oth.appendChild(r); }); }
  const arrow = G_ARROW[g.arrow];
  if (arrow) { setText("edgeArrow", arrow); setClass("edgeArrow", "edgearrow " + (["left", "right", "up", "down"].includes(g.arrow) ? g.arrow : "hidden")); } else setClass("edgeArrow", "edgearrow hidden");
  const need = s.need_s || 8, sec = s.valid_s ?? s.buffered_s ?? 0;
  const label = {positioning: "○ Positioning", collecting: `◔ Collecting ${Math.min(sec, need).toFixed(0)} / ${need} s`, adjusting: "◑ Adjust conditions",
    stabilizing: "◑ Stabilizing", reliable: "● Reliable pulse"}[g.stage] || "";
  setText("gStage", label);
  setClass("gProg", "gprog" + (g.stage === "reliable" ? " reliable" : g.stage === "stabilizing" ? " stabilizing" : ""));
  const w = (g.stage === "collecting" ? Math.min(sec / need, 1) * 100 : 0) + "%";
  setStyle(pr.firstElementChild, "width", w);
}

/* ---------------------------------------------------------------- the heart-rate readout */
const GRACE_S = 2.5;      // two one-per-second estimate cycles: a reading this fresh is still the latest confirmed value, not an "older" one
const OLDER_MAX_S = 120;  // an old reading is shown (dimmed, with its age) for at most this long
const REJECT_WHY = {no_face: "no face found", motion: "too much movement", dark: "lighting too dark", bright: "lighting too bright",
  unstable_light: "lighting is changing", weak_signal: "pulse signal too weak", disagree: "methods disagree", multi_face: "more than one face in view"};
function readingState(s) {
  const h = s.headline || {}, lr = s.last_reliable;
  if (s.running && h.status === "ok") return {kind: "current", bpm: h.bpm, age: 0};
  if (lr && lr.age_s !== undefined) {
    const age = lr.age_s;
    if (age <= GRACE_S && s.running) return {kind: "held", bpm: lr.bpm, age};
    if (age <= OLDER_MAX_S) return {kind: "older", bpm: lr.bpm, age};
  }
  if (!s.running || h.status === "collecting") return {kind: "waiting"};
  return {kind: "rejected"};
}
function bpmReading(s) {
  const h = s.headline || {}, st = readingState(s), why = REJECT_WHY[h.status] || "";
  let big = "—", label = "", cls = "bignum stale", badge = h.text || "Waiting", bcls = "badge";
  if (st.kind === "current") {
    big = Math.round(st.bpm); cls = "bignum"; label = "Current reliable estimate"; bcls = "badge ok";
  } else if (st.kind === "held") {
    big = Math.round(st.bpm); cls = "bignum"; label = `Latest confirmed estimate (${Math.round(st.age)} s ago, re-checking)`; bcls = "badge warn";
  } else if (st.kind === "older") {
    big = Math.round(st.bpm); cls = "bignum older"; label = `Older estimate from ${ageText(st.age)} ago — not current${why ? " (" + why + ")" : ""}`; bcls = "badge warn";
  } else if (st.kind === "waiting") {
    label = s.running ? `Not enough data yet — no number is shown until the signal has been checked` : "Press Start to begin"; bcls = "badge";
  } else {
    label = `Signal rejected: ${why || "not reliable"}. No number is shown.`; bcls = "badge " + (["weak_signal", "disagree", "unstable_light", "multi_face"].includes(h.status) ? "warn" : "bad");
  }
  setText("bpm", String(big)); setClass("bpm", cls);
  setText("status", badge); setClass("status", bcls);
  setText("bpmState", label); if ($("bpmState").dataset.kind !== st.kind) $("bpmState").dataset.kind = st.kind;
  setText("spread", st.kind === "current" && h.spread !== null && h.spread !== undefined ? `± ${h.spread.toFixed(1)} BPM spread between agreeing methods` : "");
  const lr = s.last_reliable;
  setText("lastOk", (st.kind === "rejected" || st.kind === "waiting") && lr ? `Last reliable reading: ${Math.round(lr.bpm)} BPM, ${ageText(lr.age_s)} ago` : "");
  return st;
}

/* ---------------------------------------------------------------- start-up checklist (driven by measured state, not a timer) */
const CHECK_ITEMS = [["camera", "Camera connected"], ["face", "Face detected"], ["position", "Positioning suitable"],
  ["collect", "Collecting pulse data"], ["stabilize", "Stabilizing the measurement"], ["reliable", "Reliable BPM available"]];
function checklist(s) {
  const g = s.guidance, h = s.headline || {}, need = s.need_s || 8, valid = s.valid_s ?? s.buffered_s ?? 0;
  const camera = !!s.running && (s.frames || 0) > 0;
  const face = camera && (g ? !!g.face : h.status !== "no_face");
  const position = face && (g ? !!(g.framing_ok || g.stage === "reliable") : true);
  const collect = position && valid >= need;
  const reliable = collect && h.status === "ok";
  const stabilize = reliable;                          // "stabilizing" is the stretch between enough data and the first reliable reading
  const done = [camera, face, position, collect, stabilize, reliable];
  let cur = done.indexOf(false); if (cur < 0) cur = 5;
  const sub = {collect: !collect && position ? ` ${Math.min(valid, need).toFixed(0)}/${need} s` : ""};
  setHTML("steps", CHECK_ITEMS.map(([k, t], i) => `<li class="${done[i] && (i < cur || reliable) ? "done" : i === cur ? "cur" : ""}">${t}${sub[k] || ""}</li>`).join(""));
  const pct = Math.min(valid / need, 1) * 100;
  const bar = $("collectBar");
  if (bar) {
    setOn(bar, "hidden", !s.running || reliable);
    setStyle(bar.firstElementChild, "width", (position ? pct : 0) + "%");
    setText("collectTxt", !s.running || reliable ? "" : !position ? "Waiting for a well-framed face before data collection counts." :
      collect ? "Enough data collected; checking that the signal is clear and the methods agree." :
      `Collected ${Math.min(valid, need).toFixed(1)} of ${need} s of usable face data (counts only while the face is tracked).`);
  }
}

/* ---------------------------------------------------------------- stability of the last reliable readings */
function stability(s) {
  const h = (s.history || []).filter((r) => r.status === "ok" && r.bpm != null).slice(-10).map((r) => r.bpm);
  setText("stability", h.length >= 5 ? `Last ${h.length} reliable readings: ${Math.min(...h).toFixed(0)}–${Math.max(...h).toFixed(0)} BPM` : "");
}
