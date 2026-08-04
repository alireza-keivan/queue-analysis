"use strict";

const $ = (id) => document.getElementById(id);
const state = { jobs: [], selected: null, videoUrls: {} };

/* ------------------------------------------------------------------ */
/* tiny SVG chart toolkit (no external charting library)               */
/* ------------------------------------------------------------------ */

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(max, count = 4) {
  if (max <= 0) return [0, 1];
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(v);
  return ticks;
}

function fmt(n, dp = 1) {
  if (!isFinite(n)) return "0";
  return Number.isInteger(n) ? String(n) : n.toFixed(dp);
}

function emptyChart(container, message) {
  container.innerHTML = `<div class="nodata">${message}</div>`;
}

/* ------------------------------------------------------------------ */
/* shared hover tooltip (Google-Analytics-style floating readout)      */
/* ------------------------------------------------------------------ */

function getHoverTooltip() {
  let node = document.getElementById("chart-hover-tooltip");
  if (!node) {
    node = document.createElement("div");
    node.id = "chart-hover-tooltip";
    node.className = "hover-tooltip";
    node.hidden = true;
    document.body.appendChild(node);
  }
  return node;
}

function idList(ids, max = 12) {
  if (!ids || !ids.length) return `<span class="hover-none">none</span>`;
  const shown = ids.slice(0, max).join(", ");
  const rest = ids.length > max ? ` +${ids.length - max} more` : "";
  return shown + rest;
}

function showHoverTooltip(clientX, clientY, point) {
  const node = getHoverTooltip();
  node.innerHTML = `
    <div class="hover-time">t = ${point.x.toFixed(1)}s</div>
    <div class="hover-row"><span class="hover-label in">Inside ROI (${point.inside.length})</span>
      <span class="hover-ids">${idList(point.inside)}</span></div>
    <div class="hover-row"><span class="hover-label out">Outside ROI (${point.outside.length})</span>
      <span class="hover-ids">${idList(point.outside)}</span></div>
  `;
  node.hidden = false;

  // Keep it near the cursor but fully on-screen (flip sides near edges).
  const pad = 14;
  const rect = node.getBoundingClientRect();
  let left = clientX + pad;
  let top = clientY + pad;
  if (left + rect.width > window.innerWidth - 8) left = clientX - rect.width - pad;
  if (top + rect.height > window.innerHeight - 8) top = clientY - rect.height - pad;
  node.style.left = `${Math.max(8, left)}px`;
  node.style.top = `${Math.max(8, top)}px`;
}

function hideHoverTooltip() {
  const node = document.getElementById("chart-hover-tooltip");
  if (node) node.hidden = true;
}

/**
 * Step chart of queue occupancy. Step (not smooth) because each snapshot is a
 * discrete sample that holds until the next one - interpolating would invent
 * data that was never measured.
 *
 * Hovering shows a floating readout of exactly which track IDs were inside
 * vs outside the ROI at that 0.1s sample, snapped to the nearest real
 * snapshot (there's one snapshot per processed frame, so "nearest" already
 * means "the actual sample at that instant", not an interpolation).
 */
function occupancyChart(container, points) {
  container.innerHTML = "";
  if (!points.length) return emptyChart(container, "No snapshots recorded for this job.");

  const W = 860, H = 260;
  const pad = { t: 14, r: 16, b: 34, l: 46 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  const maxX = Math.max(...points.map((p) => p.x), 1);
  const maxYraw = Math.max(...points.map((p) => p.y), 1);
  const yTicks = niceTicks(maxYraw, 4);
  const maxY = yTicks[yTicks.length - 1];

  const sx = (v) => pad.l + (v / maxX) * iw;
  const sy = (v) => pad.t + ih - (v / maxY) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);

  const defs = el("defs", {}, svg);
  const grad = el("linearGradient", { id: "occGrad", x1: "0", y1: "0", x2: "0", y2: "1" }, defs);
  el("stop", { offset: "0%", "stop-color": "var(--accent)", "stop-opacity": ".38" }, grad);
  el("stop", { offset: "100%", "stop-color": "var(--accent)", "stop-opacity": "0" }, grad);

  const axis = el("g", { class: "axis" }, svg);
  for (const t of yTicks) {
    el("line", { class: "gridline", x1: pad.l, y1: sy(t), x2: W - pad.r, y2: sy(t) }, axis);
    el("text", { x: pad.l - 8, y: sy(t) + 3.5, "text-anchor": "end" }, axis).textContent = fmt(t, 0);
  }
  const xTicks = niceTicks(maxX, 6);
  for (const t of xTicks) {
    if (t > maxX) continue;
    el("text", { x: sx(t), y: H - pad.b + 16, "text-anchor": "middle" }, axis).textContent = fmt(t, 0);
  }
  el("text", { x: pad.l + iw / 2, y: H - 3, "text-anchor": "middle", class: "axis-title" }, svg)
    .textContent = "seconds into video";

  // step path
  const sorted = [...points].sort((a, b) => a.x - b.x);
  let d = `M ${sx(sorted[0].x)} ${sy(sorted[0].y)}`;
  for (let i = 1; i < sorted.length; i++) {
    d += ` L ${sx(sorted[i].x)} ${sy(sorted[i - 1].y)} L ${sx(sorted[i].x)} ${sy(sorted[i].y)}`;
  }
  const last = sorted[sorted.length - 1];
  const area = `${d} L ${sx(last.x)} ${sy(0)} L ${sx(sorted[0].x)} ${sy(0)} Z`;

  el("path", { d: area, fill: "url(#occGrad)" }, svg);
  el("path", {
    d, fill: "none", stroke: "var(--accent)", "stroke-width": "1.8",
    "stroke-linejoin": "round", "stroke-linecap": "round",
  }, svg);

  // peak marker
  const peak = sorted.reduce((a, b) => (b.y > a.y ? b : a), sorted[0]);
  if (peak.y > 0) {
    el("circle", { cx: sx(peak.x), cy: sy(peak.y), r: 3.5, fill: "var(--accent)" }, svg);
  }

  // --- hover: crosshair + floating ID readout, snapped to nearest sample ---
  const crosshair = el("line", {
    x1: 0, x2: 0, y1: pad.t, y2: pad.t + ih, class: "crosshair", visibility: "hidden",
  }, svg);
  const crossDot = el("circle", { r: 3, class: "crosshair-dot", visibility: "hidden" }, svg);

  function nearestPoint(svgX) {
    const t = ((svgX - pad.l) / iw) * maxX;
    let best = sorted[0], bestDist = Infinity;
    for (const p of sorted) {
      const dist = Math.abs(p.x - t);
      if (dist < bestDist) { bestDist = dist; best = p; }
    }
    return best;
  }

  function onMove(evt) {
    const rect = svg.getBoundingClientRect();
    const fracX = (evt.clientX - rect.left) / rect.width;
    const svgX = fracX * W;
    if (svgX < pad.l || svgX > W - pad.r) { onLeave(); return; }

    const point = nearestPoint(svgX);
    const px = sx(point.x), py = sy(point.y);

    crosshair.setAttribute("x1", px);
    crosshair.setAttribute("x2", px);
    crosshair.setAttribute("visibility", "visible");
    crossDot.setAttribute("cx", px);
    crossDot.setAttribute("cy", py);
    crossDot.setAttribute("visibility", "visible");

    showHoverTooltip(evt.clientX, evt.clientY, point);
  }
  function onLeave() {
    crosshair.setAttribute("visibility", "hidden");
    crossDot.setAttribute("visibility", "hidden");
    hideHoverTooltip();
  }

  const capture = el("rect", {
    x: pad.l, y: pad.t, width: iw, height: ih, fill: "transparent", class: "hover-capture",
  }, svg);
  capture.addEventListener("pointermove", onMove);
  capture.addEventListener("pointerleave", onLeave);
}

/**
 * Dwell-time histogram. Bins that fall entirely under 1s are drawn in the warn
 * colour, because those are the ones most likely to be tracking churn rather
 * than real short visits.
 */
function dwellHistogram(container, values) {
  container.innerHTML = "";
  if (!values.length) return emptyChart(container, "No completed tracks for this job.");

  const W = 860, H = 240;
  const pad = { t: 14, r: 16, b: 40, l: 46 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  const maxV = Math.max(...values, 0.1);
  const binCount = Math.min(16, Math.max(5, Math.ceil(Math.sqrt(values.length))));
  const binW = maxV / binCount;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    x0: i * binW, x1: (i + 1) * binW, n: 0,
  }));
  for (const v of values) {
    let i = Math.floor(v / binW);
    if (i >= binCount) i = binCount - 1;
    if (i < 0) i = 0;
    bins[i].n++;
  }

  const yTicks = niceTicks(Math.max(...bins.map((b) => b.n), 1), 4);
  const maxY = yTicks[yTicks.length - 1];
  const sy = (v) => pad.t + ih - (v / maxY) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const axis = el("g", { class: "axis" }, svg);
  for (const t of yTicks) {
    el("line", { class: "gridline", x1: pad.l, y1: sy(t), x2: W - pad.r, y2: sy(t) }, axis);
    el("text", { x: pad.l - 8, y: sy(t) + 3.5, "text-anchor": "end" }, axis).textContent = fmt(t, 0);
  }

  const slot = iw / binCount;
  bins.forEach((b, i) => {
    const h = Math.max(b.n > 0 ? 2 : 0, (b.n / maxY) * ih);
    const x = pad.l + i * slot + slot * 0.12;
    const w = slot * 0.76;
    const churn = b.x1 <= 1.0;
    const rect = el("rect", {
      x, y: pad.t + ih - h, width: w, height: h, rx: 2,
      fill: churn ? "var(--warn)" : "var(--accent)",
      opacity: b.n ? (churn ? ".85" : ".7") : "0",
    }, svg);
    el("title", {}, rect).textContent =
      `${fmt(b.x0, 1)}–${fmt(b.x1, 1)}s : ${b.n} track${b.n === 1 ? "" : "s"}`;

    if (i % Math.ceil(binCount / 6) === 0) {
      el("text", { x: x + w / 2, y: H - pad.b + 16, "text-anchor": "middle" }, axis)
        .textContent = fmt(b.x0, 1);
    }
  });

  el("text", { x: pad.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" }, svg)
    .textContent = "dwell seconds";
}

/* ------------------------------------------------------------------ */
/* data + rendering                                                    */
/* ------------------------------------------------------------------ */

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(message, isError = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = "toast" + (isError ? " err" : "");
  node.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { node.hidden = true; }, isError ? 9000 : 5000);
}

async function loadHealth() {
  try {
    const h = await api("/api/health");
    $("pill-storage").className = "pill " + (h.storage_api ? "ok" : "bad");
    $("pill-runpod").className = "pill " + (h.runpod_configured ? "ok" : "bad");
    if (!h.runpod_configured) {
      $("pill-runpod").title = "RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID not set on the dashboard service";
    }
  } catch (_) {
    $("pill-storage").className = "pill bad";
  }
}

async function loadJobs() {
  const list = $("joblist");
  try {
    state.jobs = await api("/api/jobs");
  } catch (err) {
    list.innerHTML = `<li class="none">Could not reach storage-api.</li>`;
    return;
  }
  if (!state.jobs.length) {
    list.innerHTML = `<li class="none">No jobs yet. Submit a video to get started.</li>`;
    return;
  }
  list.innerHTML = "";
  for (const job of state.jobs) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = job.job_id === state.selected ? "active" : "";
    btn.innerHTML =
      `<span class="job-id">${job.job_id}</span>` +
      `<span class="job-meta">peak ${job.peak_queue} · ${job.track_count} tracks · ` +
      `${fmt(job.duration_seconds, 0)}s</span>`;
    btn.onclick = () => selectJob(job.job_id);
    li.appendChild(btn);
    list.appendChild(li);
  }
}

function renderTiles(job, snapshots, tracks) {
  const dwells = tracks.map((t) => t.dwell_seconds);
  const churn = dwells.filter((d) => d < 1.0).length;
  const sorted = [...dwells].sort((a, b) => a - b);
  const median = sorted.length
    ? sorted.length % 2
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : 0;
  const churnPct = dwells.length ? Math.round((churn / dwells.length) * 100) : 0;

  const tiles = [
    { label: "Peak queue", value: job ? job.peak_queue : Math.max(0, ...snapshots.map((s) => s.queue_count)), note: "people in ROI" },
    { label: "Avg queue", value: fmt(job ? job.avg_queue : 0, 2), note: "across samples" },
    { label: "Tracks", value: tracks.length, note: "completed dwells" },
    { label: "Median dwell", value: fmt(median, 2), unit: "s", note: "typical stay" },
    { label: "Max dwell", value: fmt(Math.max(0, ...dwells), 1), unit: "s", note: "longest stay" },
    { label: "Sub-1s tracks", value: `${churnPct}%`, note: `${churn} of ${dwells.length} · likely churn`, warn: churnPct > 50 },
  ];

  $("tiles").innerHTML = tiles.map((t) => `
    <div class="tile${t.warn ? " warn" : ""}">
      <div class="tile-label">${t.label}</div>
      <div class="tile-value">${t.value}${t.unit ? `<span class="unit">${t.unit}</span>` : ""}</div>
      <div class="tile-note">${t.note}</div>
    </div>`).join("");
}

function renderTrackTable(tracks) {
  const body = $("track-table").querySelector("tbody");
  if (!tracks.length) {
    body.innerHTML = `<tr><td colspan="3" style="color:var(--text-faint)">No tracks.</td></tr>`;
    return;
  }
  body.innerHTML = [...tracks]
    .sort((a, b) => b.dwell_seconds - a.dwell_seconds)
    .map((t) => {
      const low = t.dwell_seconds < 1.0;
      return `<tr>
        <td>${t.track_id}</td>
        <td class="num">${t.dwell_seconds.toFixed(2)}</td>
        <td><span class="tag ${low ? "tag-low" : "tag-ok"}">${low ? "likely churn" : "plausible"}</span></td>
      </tr>`;
    }).join("");
}

async function selectJob(jobId) {
  state.selected = jobId;
  $("empty-state").hidden = true;
  $("detail").hidden = false;
  $("detail-jobid").textContent = jobId;
  $("detail-title").textContent = "Run detail";

  const link = $("video-link");
  if (state.videoUrls[jobId]) {
    link.href = state.videoUrls[jobId];
    link.hidden = false;
  } else {
    link.hidden = true;
  }

  await loadJobs();

  let data;
  try {
    data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  } catch (err) {
    toast(`Could not load job: ${err.message}`, true);
    return;
  }

  const snapshots = data.snapshots || [];
  const tracks = data.tracks || [];
  const job = state.jobs.find((j) => j.job_id === jobId);

  renderTiles(job, snapshots, tracks);
  $("occupancy-sub").textContent =
    `${snapshots.length} samples over ${fmt(Math.max(0, ...snapshots.map((s) => s.timestamp)), 0)}s`;
  occupancyChart($("chart-occupancy"), snapshots.map((s) => ({
    x: s.timestamp, y: s.queue_count,
    inside: s.inside_ids || [], outside: s.outside_ids || [],
  })));
  dwellHistogram($("chart-dwell"), tracks.map((t) => t.dwell_seconds));
  renderTrackTable(tracks);
}

/* ------------------------------------------------------------------ */
/* actions                                                             */
/* ------------------------------------------------------------------ */

$("submit-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("submit-btn");
  const hint = $("submit-hint");
  const payload = {
    video_url: $("video-url").value.trim(),
    target_fps: parseInt($("target-fps").value, 10) || 10,
    annotate: $("annotate").checked,
  };

  btn.disabled = true;
  btn.textContent = "Running…";
  hint.className = "hint busy";
  hint.textContent = "Submitted. Waiting for the GPU worker to finish…";

  try {
    const res = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.annotated_video_url) state.videoUrls[res.job_id] = res.annotated_video_url;
    hint.className = "hint";
    hint.textContent = `Done: ${res.snapshot_count} snapshots, ${res.track_count} tracks.`;
    toast(`Job complete — ${res.track_count} tracks recorded.`);
    await loadJobs();
    await selectJob(res.job_id);
  } catch (err) {
    hint.className = "hint err";
    hint.textContent = err.message;
    toast(err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run analysis";
  }
});

$("delete-btn").addEventListener("click", async () => {
  const jobId = state.selected;
  if (!jobId) return;
  if (!confirm(`Delete all stored data for this run?\n\n${jobId}`)) return;
  try {
    const res = await api(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    toast(`Deleted ${res.snapshots_deleted} snapshots and ${res.tracks_deleted} tracks.`);
    state.selected = null;
    $("detail").hidden = true;
    $("empty-state").hidden = false;
    await loadJobs();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, true);
  }
});

$("refresh-btn").addEventListener("click", () => { loadJobs(); loadHealth(); });

loadHealth();
loadJobs();
setInterval(loadHealth, 30000);