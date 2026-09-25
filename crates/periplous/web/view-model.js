export const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
export const gib = bytes => Number.isFinite(bytes) ? bytes / 2 ** 30 : null;
export const percent = (used, total) => Number.isFinite(used) && Number.isFinite(total) && total > 0 ? 100 * used / total : null;
export const clamp = value => Math.max(0, Math.min(100, value));
export const unit = (value, suffix, digits = 0) => Number.isFinite(value) ? `${number(value, digits)} ${suffix}` : "—";

export function processRows(gpus) {
  const rows = [];
  let incomplete = !Array.isArray(gpus);
  for (const gpu of gpus ?? []) {
    const byPid = new Map();
    for (const kind of ["compute", "graphics"]) {
      const group = gpu.processes?.[kind];
      if (!Array.isArray(group)) { incomplete = true; continue; }
      for (const process of group) {
        const previous = byPid.get(process.pid);
        const memory = [previous?.used_memory_bytes, process.used_memory_bytes].filter(Number.isFinite);
        byPid.set(process.pid, {
          gpu: gpu.index, pid: process.pid, name: process.name ?? previous?.name ?? "—",
          // Same PID can have both types of context. Its memory is not additive.
          used_memory_bytes: memory.length ? Math.max(...memory) : null,
        });
      }
    }
    rows.push(...byPid.values());
  }
  rows.sort((a, b) => a.gpu - b.gpu || a.pid - b.pid);
  return { rows, incomplete };
}

// Coordinates use the server's monotonic history clock. Missing samples and
// pauses break the trace rather than drawing a plausible but invented plateau.
export function stepPath(points, key, now, windowMs, width) {
  const left = 32, right = Math.max(left + 1, width - 5), top = 14, bottom = 124;
  const x = time => left + (time - (now - windowMs)) / windowMs * (right - left);
  const y = value => bottom - clamp(value) / 100 * (bottom - top);
  let path = "", previous = null;
  for (const point of points) {
    if (point.elapsed_ms < now - windowMs || point.elapsed_ms > now) continue;
    if (!Number.isFinite(point[key])) { previous = null; continue; }
    const px = x(point.elapsed_ms).toFixed(2), py = y(point[key]).toFixed(2);
    path += previous && point.elapsed_ms - previous.elapsed_ms <= 2500 ? `H${px}V${py}` : `M${px},${py}`;
    previous = point;
  }
  return path;
}
