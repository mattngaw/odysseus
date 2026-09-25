import test from 'node:test';
import assert from 'node:assert/strict';
import { number, gib, percent, unit, processRows, stepPath } from './view-model.js';

test('unavailable metrics stay distinct from a measured zero', () => {
  assert.equal(number(null), '—');
  assert.equal(number(0), '0');
  assert.equal(gib(null), null);
  assert.equal(gib(2 ** 30), 1);
  assert.equal(percent(null, 100), null);
  assert.equal(percent(10, 0), null);
  assert.equal(percent(0, 100), 0);
  assert.equal(unit(null, 'W'), '—');
});

test('GPU process contexts deduplicate memory within a GPU, not across GPUs', () => {
  const p = { pid: 7, name: 'python', used_memory_bytes: 1024 };
  const result = processRows([
    { index: 0, processes: { compute: [p], graphics: [{ ...p, used_memory_bytes: 2048 }] } },
    { index: 1, processes: { compute: [p], graphics: null } },
  ]);
  assert.deepEqual(result.rows.map(r => [r.gpu, r.pid, r.used_memory_bytes]), [[0, 7, 2048], [1, 7, 1024]]);
  assert.equal(result.incomplete, true);
  assert.deepEqual(processRows([]), { rows: [], incomplete: false });
  assert.deepEqual(processRows(null), { rows: [], incomplete: true });
});

test('history traces break on unavailable samples and on missed polling intervals', () => {
  const points = [
    { elapsed_ms: 0, util: 0 }, { elapsed_ms: 1000, util: 100 },
    { elapsed_ms: 2000, util: null }, { elapsed_ms: 3000, util: 50 },
    { elapsed_ms: 7000, util: 50 },
  ];
  const path = stepPath(points, 'util', 10_000, 10_000, 137);
  assert.equal(path, 'M32.00,124.00H42.00V14.00M62.00,69.00M102.00,69.00');
  assert.equal(stepPath(points, 'util', 30_000, 10_000, 137), '');
});
