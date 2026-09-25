import { number, gib, percent, unit, clamp, processRows, stepPath } from "/view-model.js";

const $ = id => document.getElementById(id);
// Build identity is independent of live telemetry; a failure must not hide readings.
fetch("/api/deployment", { cache: "no-store", signal: AbortSignal.timeout(5000) })
  .then(response => { if (!response.ok) throw new Error("Deployment unavailable"); return response.json(); })
  .then(({ environment, release }) => {
    $("deployment").textContent = environment ? `/ ${environment} · ${release?.slice(2, 10) ?? "unknown"}` : "/ local";
    $("deployment").title = release ?? "Unversioned local build";
  })
  .catch(() => { $("deployment").textContent = "/ version unavailable"; });
const theme = $("theme");
let appearance = null;
try { appearance = localStorage.getItem("periplous-theme"); } catch {}
function setTheme(value) {
  document.documentElement.dataset.theme = value;
  theme.textContent = value === "dark" ? "[ light ]" : "[ dark ]";
  theme.setAttribute("aria-label", `Switch to ${value === "dark" ? "light" : "dark"} appearance`);
}
setTheme(["light", "dark"].includes(appearance) ? appearance : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
theme.addEventListener("click", () => {
  const value = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(value);
  try { localStorage.setItem("periplous-theme", value); } catch {}
});

let documentData = null, receivedAt = 0, sampleAge = 0, failed = false;
const panels = new Map();
const namespace = "http://www.w3.org/2000/svg";
function svgNode(tag, attributes, text) {
  const node = document.createElementNS(namespace, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function bar(element, value) {
  const filled = Number.isFinite(value) ? Math.round(clamp(value) / 5) : 0;
  element.replaceChildren(document.createTextNode(`[${"█".repeat(filled)}`));
  const unused = document.createElement("span"); unused.className = "pp-unused";
  unused.textContent = (Number.isFinite(value) ? "░" : "·").repeat(20 - filled);
  element.append(unused, "]");
}
function field(panel, name, text) { panel.querySelector(`[data-field="${name}"]`).textContent = text; }
function meter(panel, name, value) {
  const span = panel.querySelector(`[data-field="${name}"]`);
  span.style.width = `${Number.isFinite(value) ? clamp(value) : 0}%`;
}
function age() { return sampleAge + (receivedAt ? performance.now() - receivedAt : 0); }
function status() {
  const hasSnapshot = documentData?.snapshot != null;
  const stale = hasSnapshot && (failed || age() > 5000);
  $("periplous-hardware").classList.toggle("pp-stale", stale);
  $("connection").textContent = !hasSnapshot ? (failed ? "UNAVAILABLE · RETRYING" : "CONNECTING") : stale ? "STALE · RETRYING" : documentData.snapshot.issues.length ? "PARTIAL" : "LIVE";
  $("updated").textContent = hasSnapshot ? `sample ${Math.floor(age() / 1000)}s ago${stale ? " / last readings retained" : ""}` : "Awaiting first sample";
}

function historyPoints(index) {
  return (documentData?.history ?? []).map(point => {
    const gpu = point.gpus?.find(g => g.index === index);
    return { elapsed_ms: point.elapsed_ms, utilization_percent: gpu?.utilization_percent ?? null, memory_percent: gpu?.memory_percent ?? null };
  });
}
function drawChart(panel, index) {
  const svg = panel.querySelector("svg"), width = svg.getBoundingClientRect().width;
  if (width < 40 || !documentData) return;
  const points = historyPoints(index);
  const now = (documentData.history.at(-1)?.elapsed_ms ?? 0) + age();
  const windowMs = documentData.history_window_ms;
  const left = 32, right = width - 5, top = 14, bottom = 124;
  svg.setAttribute("viewBox", `0 0 ${width} 153`);
  svg.setAttribute("aria-label", `GPU ${index} utilization and VRAM percentage over the last ten minutes`);
  const nodes = [svgNode("title", {}, `GPU ${index} history`)];
  for (const value of [0, 50, 100]) {
    const y = bottom - value / 100 * (bottom - top);
    nodes.push(svgNode("line", { class: "pp-gridline", x1: left, x2: right, y1: y, y2: y }), svgNode("text", { x: 24, y: y + 4, "text-anchor": "end" }, String(value)));
  }
  for (const [key, css] of [["memory_percent", "pp-vram-line"], ["utilization_percent", "pp-gpu-line"]]) {
    nodes.push(svgNode("path", { class: css, d: stepPath(points, key, now, windowMs, width) }));
  }
  for (const [fraction, label] of [[0, "−10m"], [0.5, "−5m"], [1, "now"]]) {
    nodes.push(svgNode("text", { x: left + fraction * (right - left), y: 146, "text-anchor": fraction === 0 ? "start" : fraction === 1 ? "end" : "middle" }, label));
  }
  const guide = svgNode("line", { class: "pp-guide", x1: 0, x2: 0, y1: top, y2: bottom, visibility: "hidden" });
  const overlay = svgNode("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: "transparent" });
  nodes.push(guide, overlay); svg.replaceChildren(...nodes);
  const readout = panel.querySelector('[data-field="readout"]');
  const latest = () => {
    const point = points.at(-1);
    readout.textContent = point ? `latest / GPU ${unit(point.utilization_percent, "%")} / VRAM ${unit(point.memory_percent, "%", 1)}` : "Awaiting history";
  };
  latest();
  function hover(event) {
    const fraction = Math.max(0, Math.min(1, (event.clientX - svg.getBoundingClientRect().left - left) / (right - left)));
    const time = now - windowMs + fraction * windowMs;
    let nearest = null;
    for (const p of points) if (!nearest || Math.abs(p.elapsed_ms - time) < Math.abs(nearest.elapsed_ms - time)) nearest = p;
    const x = left + fraction * (right - left);
    guide.setAttribute("x1", x); guide.setAttribute("x2", x); guide.setAttribute("visibility", "visible");
    readout.textContent = !nearest || Math.abs(nearest.elapsed_ms - time) > 1500 ? "No sample here" : `−${Math.floor((now - nearest.elapsed_ms) / 1000)}s / GPU ${unit(nearest.utilization_percent, "%")} / VRAM ${unit(nearest.memory_percent, "%", 1)}`;
  }
  overlay.addEventListener("pointermove", hover); overlay.addEventListener("pointerdown", hover);
  svg.onpointerleave = () => { guide.setAttribute("visibility", "hidden"); latest(); };
}

function render() {
  const snapshot = documentData?.snapshot;
  if (!snapshot) return;
  const cpu = snapshot.host.cpu, ram = snapshot.host.memory;
  $("cpu-value").textContent = unit(cpu?.utilization_percent, "%", 1);
  $("cpu-detail").textContent = cpu ? `${cpu.logical_cpu_count} logical CPUs${cpu.utilization_percent === null ? " / warming up" : ""}` : "unavailable";
  bar($("cpu-bar"), cpu?.utilization_percent);
  const ramPercent = percent(ram?.used_bytes, ram?.total_bytes);
  $("ram-value").textContent = ram ? `${number(gib(ram.used_bytes), 1)} / ${number(gib(ram.total_bytes), 1)} GiB` : "—";
  $("ram-detail").textContent = ram ? `${unit(ramPercent, "%", 1)} used` : "unavailable";
  bar($("ram-bar"), ramPercent);
  const gpus = snapshot.gpus;
  $("inventory").textContent = `${gpus ? `${gpus.length} GPUs · ` : ""}read-only`;
  $("gpu-message").hidden = Boolean(gpus?.length);
  $("gpu-message").textContent = gpus === null ? "GPU readings unavailable" : "No NVIDIA GPUs detected";
  const active = new Set((gpus ?? []).map(g => g.index));
  for (const [index, panel] of panels) if (!active.has(index)) { panel.remove(); panels.delete(index); }
  for (const gpu of gpus ?? []) {
    let panel = panels.get(gpu.index);
    if (!panel) {
      panel = $("gpu-template").content.firstElementChild.cloneNode(true);
      panel.setAttribute("aria-label", `GPU ${gpu.index} metrics`);
      panels.set(gpu.index, panel); $("gpus").append(panel);
    }
    field(panel, "index", `GPU ${gpu.index}`); field(panel, "name", gpu.name ?? "unavailable");
    field(panel, "utilization", number(gpu.utilization_percent));
    field(panel, "memory", gpu.memory ? `${number(gib(gpu.memory.used_bytes), 1)} / ${number(gib(gpu.memory.total_bytes), 1)}` : "—");
    meter(panel, "utilization-bar", gpu.utilization_percent); meter(panel, "memory-bar", percent(gpu.memory?.used_bytes, gpu.memory?.total_bytes));
    field(panel, "power", unit(gpu.power_watts, "W")); field(panel, "temperature", unit(gpu.temperature_celsius, "°C"));
    field(panel, "sm-clock", unit(gpu.sm_clock_mhz, "MHz")); field(panel, "memory-clock", unit(gpu.memory_clock_mhz, "MHz"));
    field(panel, "pcie-receive", unit(gpu.pcie_receive_kb_per_second, "KB/s")); field(panel, "pcie-send", unit(gpu.pcie_send_kb_per_second, "KB/s"));
    drawChart(panel, gpu.index);
  }
  const { rows, incomplete } = processRows(gpus);
  $("process-count").textContent = `/ ${rows.length}${incomplete ? "+ unknown" : ""}`;
  $("process-note").hidden = !incomplete;
  const tableRows = rows.map(row => {
    const tr = document.createElement("tr");
    for (const text of [row.gpu, row.pid, row.name, unit(gib(row.used_memory_bytes), "GiB", 2)]) {
      const td = document.createElement("td"); td.textContent = text; tr.append(td);
    }
    return tr;
  });
  if (!rows.length) {
    const tr = document.createElement("tr"), td = document.createElement("td"); td.colSpan = 4;
    td.textContent = incomplete ? "Process readings unavailable" : "No active GPU processes"; tr.append(td); tableRows.push(tr);
  }
  $("processes").replaceChildren(...tableRows);
  $("issues").hidden = !snapshot.issues.length;
  $("issues").textContent = snapshot.issues.map(i => `${i.gpu_index === null ? "host" : `GPU ${i.gpu_index}`} / ${i.metric}: ${i.reason.replaceAll("_", " ")}`).join(" · ");
  status();
}

async function poll() {
  try {
    const response = await fetch("/api/hardware", { cache: "no-store", signal: AbortSignal.timeout(4000) });
    if (!response.ok) throw new Error("unavailable");
    const next = await response.json();
    const headerAge = response.headers.get("x-periplous-sample-age-ms");
    if (!Array.isArray(next.history) || !Number.isFinite(next.history_window_ms) || next.history_window_ms <= 0) throw new Error("invalid snapshot");
    documentData = next; sampleAge = headerAge === null ? 0 : Number(headerAge); receivedAt = performance.now(); failed = response.headers.get("x-periplous-stale") === "true" && next.snapshot !== null;
    render(); status();
  } catch { failed = true; status(); }
  setTimeout(poll, 1000);
}
new ResizeObserver(() => { for (const [index, panel] of panels) drawChart(panel, index); }).observe($("gpus"));
setInterval(status, 1000);
poll();
