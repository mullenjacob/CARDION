// Paste/execute in the CARDION page (used through the browser tool): starts the live source and measures how the PAGE behaves for ~25 s.
// Returns: animation-frame interval stats (page smoothness), long main-thread tasks, DOM changes per second, requests and bytes per second.
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  document.getElementById("startBtn").click();
  await sleep(3000);
  const frames = []; let run = true;
  (function loop(t) { frames.push(t); if (run) requestAnimationFrame(loop); })(performance.now());
  let longTasks = 0, longMs = 0, muts = 0;
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) { longTasks++; longMs += e.duration; } }).observe({entryTypes: ["longtask"]}); } catch (e) {}
  new MutationObserver((m) => { muts += m.length; }).observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true});
  const r0 = performance.getEntriesByType("resource").filter((e) => e.name.includes("/api/")).length;
  const b0 = performance.getEntriesByType("resource").filter((e) => e.name.includes("/api/")).reduce((a, e) => a + (e.encodedBodySize || 0), 0);
  const T = 25000, t0 = performance.now();
  await sleep(T);
  run = false;
  const secs = (performance.now() - t0) / 1000;
  const d = frames.slice(1).map((x, i) => x - frames[i]).sort((a, b) => a - b);
  const q = (p) => d.length ? d[Math.min(d.length - 1, Math.floor(p * d.length))] : null;
  const res = performance.getEntriesByType("resource").filter((e) => e.name.includes("/api/"));
  return {seconds: +secs.toFixed(1), raf_fps: +(d.length / secs).toFixed(1), raf_gap_ms_median: +q(0.5).toFixed(1), raf_gap_ms_p95: +q(0.95).toFixed(1), raf_gap_ms_max: +d[d.length - 1].toFixed(1),
    raf_gaps_over_50ms: d.filter((x) => x > 50).length, long_tasks: longTasks, long_task_ms: +longMs.toFixed(0), dom_changes_per_s: +(muts / secs).toFixed(0),
    api_requests_per_s: +((res.length - r0) / secs).toFixed(1), api_kb_per_s: +((res.reduce((a, e) => a + (e.encodedBodySize || 0), 0) - b0) / 1024 / secs).toFixed(1)};
})()
