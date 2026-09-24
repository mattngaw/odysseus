import csv
import io
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'benchmarks/baselines/2026-09-13-cache-associativity'


def section(path, title):
    text = path.read_text().split(f'\n{title}:\n', 1)[1].split('\n\n', 1)[0]
    rows = list(csv.DictReader(io.StringIO(text)))
    assert all(None not in row and None not in row.values() for row in rows)
    return rows


def write(name, rows):
    with (DATA / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(row):
    return tuple(row[k] for k in ['name', 'depth', 'mode', 'legality', 'cache'])


def validate(path):
    summary = section(path, 'Summary')
    raw = section(path, 'Raw samples')
    diagnostics = section(path, 'Untimed diagnostics')
    fixtures = {r['name']: r for r in section(path, 'Fixtures')}
    for row in summary:
        samples = [r for r in raw if label(r) == label(row) and r.get('ways') == row.get('ways')]
        assert samples
        for s in samples:
            assert int(s['total_ns']) == int(s['traverse_ns']) + int(s['clear_ns'])
        assert int(row['leaves']) == int(fixtures[row['name']]['leaves'])
        assert abs(statistics.median(int(s['total_ns']) for s in samples) - int(row['median_total_ns'])) <= 1
        assert abs(statistics.median(int(s['traverse_ns']) for s in samples) - int(row['median_ns'])) <= 1
        stats = [r for r in diagnostics if label(r) == label(row) and r.get('ways') == row.get('ways')]
        by_depth = {int(r['remaining_depth']): r for r in stats}
        for d, r in by_depth.items():
            if d > 0:
                assert int(r['visited']) == int(r['expanded']) + int(r['hits'])
                assert int(by_depth[d - 1]['visited']) == int(r['child_positions'])
        if row['cache'] == 'uncached':
            assert all(int(r['hits']) == 0 for r in stats)
            if row['mode'] == 'apply':
                assert int(by_depth[0]['visited']) == int(row['leaves'])
    return summary, raw, diagnostics


before = validate(DATA / 'before-legacy.txt')
after = validate(DATA / 'after-legacy.txt')
assert before[2] == [{k: v for k, v in row.items() if k != 'ways'} for row in after[2]]
all_summary, all_raw, all_stats, comparison = [], [], [], []
for mib in [1, 16]:
    summary, raw, stats = validate(DATA / f'compare-{mib}mib.txt')
    all_summary.extend(dict(cache_mib=mib, **r) for r in summary)
    all_raw.extend(dict(cache_mib=mib, **r) for r in raw)
    all_stats.extend(dict(cache_mib=mib, **r) for r in stats)
    for name in ['initial', 'kiwipete', 'endgame', 'promotions']:
        for mode in ['bulk', 'apply']:
            variants = {int(r['ways']): r for r in summary if r['name'] == name and r['mode'] == mode}
            assert set(variants) == {0, 1, 2, 4}
            row = dict(cache_mib=mib, name=name, depth=variants[1]['depth'], mode=mode, leaves=variants[1]['leaves'])
            for ways in [0, 1, 2, 4]:
                v = variants[ways]
                total_ns = int(v['median_total_ns'])
                row[f'ways_{ways}_total_ms'] = total_ns / 1e6
                row[f'ways_{ways}_logical_mnps'] = int(v['leaves']) * 1e3 / total_ns
                ds = [r for r in stats if r['name'] == name and r['mode'] == mode and int(r['ways']) == ways]
                row[f'ways_{ways}_expanded'] = sum(int(r['expanded']) for r in ds)
                row[f'ways_{ways}_child_positions'] = sum(int(r['child_positions']) for r in ds)
                row[f'ways_{ways}_hits'] = sum(int(r['hits']) for r in ds)
            for ways in [2, 4]:
                row[f'ways_{ways}_time_reduction_vs_1_pct'] = 100 * (1 - row[f'ways_{ways}_total_ms'] / row['ways_1_total_ms'])
                row[f'ways_{ways}_expansion_reduction_vs_1_pct'] = 100 * (1 - row[f'ways_{ways}_expanded'] / row['ways_1_expanded'])
            comparison.append(row)

write('summary.csv', all_summary)
write('raw-samples.csv', all_raw)
write('diagnostics.csv', all_stats)
write('comparison.csv', comparison)
(DATA / 'analysis-validation.json').write_text(json.dumps(dict(
    original_and_refactored_one_way_diagnostics_identical=True,
    raw_timing_arithmetic_and_medians_verified=True,
    known_leaf_counts_verified=True,
    diagnostic_accounting_verified=True,
    comparison_rows=len(comparison),
    timing_spread='Quartiles are sample spread, not confidence intervals.',
), indent=2) + '\n')
for r in comparison:
    if r['mode'] == 'apply':
        print(f"{r['cache_mib']:2} MiB {r['name']:10} d{r['depth']} "
              + ' '.join(f"{w}-way={r[f'ways_{w}_total_ms']:.3f}ms" for w in [0, 1, 2, 4])
              + f" | 2-way {r['ways_2_time_reduction_vs_1_pct']:+.1f}%, 4-way {r['ways_4_time_reduction_vs_1_pct']:+.1f}%"
              + f" | expansions {r['ways_2_expansion_reduction_vs_1_pct']:+.1f}%, {r['ways_4_expansion_reduction_vs_1_pct']:+.1f}%")
