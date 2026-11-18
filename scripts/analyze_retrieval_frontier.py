"""Apply the preregistered §22.10 decision to published per-family key-pair counts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = ('parent', 'fixed', 'adaptive', 'yoked')
PRIMARY = ('other-writes', 128)
SHORT = (('filler', 0), ('overwrite', 32))


def _read_counts(path, arms, *, data_seeds=None):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    matched_inputs = {}
    for r in rows:
        key = (r['task'], int(r['gap']), r['arm'], int(r['weight_seed']), int(r['family']))
        if data_seeds is not None and int(r['data_seed']) != data_seeds.get(key[0]):
            raise ValueError(f'wrong held-out cohort: {key}')
        hits, total, invalid = (int(r[k]) for k in ('both_correct', 'key_pairs', 'nonfinite'))
        if key in counts or key[2] not in arms or not (0 <= hits <= total and total > 0 and 0 <= invalid <= 2 * total):
            raise ValueError(f'invalid or duplicate result cell: {key}')
        if 'input_sha256' in r:
            labels, predictions = (np.asarray([list(map(int, r[f'world{w}_{col}'].split())) for w in (0, 1)])
                                   for col in ('labels', 'predictions'))
            if (labels.shape != (2, total) or predictions.shape != labels.shape or
                    (predictions == labels).all(axis=0).sum() != hits or (predictions == -1).sum() != invalid):
                raise ValueError(f'raw predictions disagree with the reported counts: {key}')
            family = key[0], key[1], key[4]
            identity = r['input_sha256'], r['world0_labels'], r['world1_labels']
            if matched_inputs.setdefault(family, identity) != identity:
                raise ValueError(f'unpaired histories or labels for family {family}')
        counts[key] = hits, total, invalid
    return counts


def analyze(path):
    counts = _read_counts(path, ARMS)
    seeds = sorted({k[3] for k in counts})
    conditions = sorted({k[:2] for k in counts})
    if len(seeds) != 3 or not {PRIMARY, *SHORT}.issubset(conditions):
        raise ValueError('the registered comparison needs three weight seeds, the primary and both short tasks')
    rates = {}
    invalids = {arm: 0 for arm in ARMS}
    cells = []
    for task, gap in conditions:
        families = sorted({k[4] for k in counts if k[:2] == (task, gap)})
        for arm in ARMS:
            # A missing family/arm is not a zero, nor an excuse to use a smaller denominator.
            cube = np.asarray([[counts[task, gap, arm, s, f] for f in families] for s in seeds], dtype=np.int64)
            if not np.all(cube[..., 1] == cube[0, 0, 1]):
                raise ValueError('different numbers of paired queries within one condition')
            if arm != ARMS[0] and int(cube[0, 0, 1]) != per_family:
                raise ValueError('unmatched number of paired questions across arms')
            per_family = int(cube[0, 0, 1])
            rates[task, gap, arm] = cube[..., 0] / per_family
            invalids[arm] += int(cube[..., 2].sum())
            cells.append(dict(task=task, gap=gap, arm=arm, both_correct=int(cube[..., 0].sum()),
                              key_pairs=int(cube[..., 1].sum()), nonfinite=int(cube[..., 2].sum()),
                              correct_by_seed=cube[..., 0].sum(axis=1).tolist()))
    primary = rates[*PRIMARY, 'adaptive']
    rng = np.random.default_rng(11101)
    seed_draw = rng.integers(len(seeds), size=(4096, len(seeds), 1))
    family_draw = rng.integers(primary.shape[1], size=(4096, 1, primary.shape[1]))
    comparisons = {}
    for other in ('parent', 'fixed', 'yoked'):
        diff = 100 * (primary - rates[*PRIMARY, other])
        boot = diff[seed_draw, family_draw].mean(axis=(1, 2))
        comparisons[other] = dict(mean_pp=float(diff.mean()), by_seed_pp=diff.mean(axis=1).tolist(),
                                  paired_95_pp=np.quantile(boot, [0.025, 0.975]).tolist())
    short = {f'{task}:{gap}': (100 * (rates[task, gap, 'adaptive'] - rates[task, gap, 'parent']))
             .mean(axis=1).tolist() for task, gap in SHORT}
    superiority = all(comparisons[a]['mean_pp'] >= 2 and min(comparisons[a]['by_seed_pp']) > 0
                      for a in ('fixed', 'yoked'))
    retention = all(min(diffs) >= -1 for diffs in short.values())
    # Descriptive only: no interpolation, no skipping a failed point and no extrapolation.
    horizons = {}
    for task in ('filler', 'other-writes'):
        gaps = [gap for name, gap in conditions if name == task]
        if not gaps or gaps[0] != 0:
            continue
        horizons[task] = {}
        for arm in ARMS:
            last = first_fail = None
            for gap in gaps:
                if np.min(rates[task, gap, arm].mean(axis=1)) < 0.90:
                    first_fail = gap
                    break
                last = gap
            horizons[task][arm] = dict(last_passing_tested_gap=last, first_failing_tested_gap=first_fail)
    return dict(weight_seeds=seeds, primary='other-writes:128', primary_adaptive_minus=comparisons,
                short_adaptive_minus_parent_pp=short, nonfinite_by_arm=invalids, cells=cells,
                empirical_90pct_grid_all_seeds=horizons,
                superiority_pass=superiority, retention_pass=retention,
                adopt_adaptive=superiority and retention and invalids['adaptive'] == 0)


def audit_schedules(path):
    """Audit what actually ran, independently of the scheduler that generated it."""
    seeds = (42, 43, 44)
    cells = {(g, w) for g in (32, 64, 128) for w in (0, 4, 8)}
    traces = {(arm, seed): {} for arm in ARMS[1:] for seed in seeds}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            updates = traces[row['arm'], int(row['weight_seed'])]
            step, anchored = int(row['step']), int(row['anchor'])
            cell = int(row['gap']), int(row['writes'])
            if step in updates or anchored not in (0, 1) or cell not in cells or (anchored and cell != (32, 0)):
                raise ValueError(f'invalid or duplicate update: {row}')
            updates[step] = None if anchored else cell
    common_anchor = None
    for key, updates in traces.items():
        if set(updates) != set(range(96)):
            raise ValueError(f'{key} does not have 96 distinct updates')
        anchor = {s for s, cell in updates.items() if cell is None}
        if len(anchor) != 24 or (common_anchor is not None and anchor != common_anchor):
            raise ValueError(f'unmatched anchor budget or positions: {key}')
        common_anchor = anchor
        if key[0] == 'fixed' and (updates != traces['fixed', seeds[0]] or
                                  any(sum(cell == c for cell in updates.values()) != 8 for c in cells)):
            raise ValueError(f'unbalanced fixed schedule: {key}')
    matched = {}
    for i, seed in enumerate(seeds):
        donor = seeds[(i + 1) % len(seeds)]
        if traces['yoked', seed] != traces['adaptive', donor]:
            raise ValueError(f'yoked schedule for {seed} differs from registered donor {donor}')
        matched[seed] = {arm: sum(cell is not None and cell == traces[arm, seed][s]
                                  for s, cell in traces['adaptive', seed].items())
                         for arm in ('fixed', 'yoked')}
    return dict(steps=96, anchor_updates=24, nonanchor_updates=72, adaptive_matching_nonanchor=matched)


def analyze_midpoint(path):
    arms = ('parent', 'fixed', 'midpoint')
    seeds = (42, 43, 44)
    counts = _read_counts(path, arms, data_seeds={'filler': 96175, 'other-writes': 96175, 'overwrite': 96176})
    conditions = [(task, d) for task in ('filler', 'other-writes') for d in (0, 64, 128, 256, 1024)] + [('overwrite', 32)]
    if len(counts) != len(arms) * len(seeds) * len(conditions) * 64:
        raise ValueError('the independent replication requires 64 matched families per arm, seed and condition')
    rates = {}; cells = []; invalid = {arm: 0 for arm in arms}
    for task, gap in conditions:
        for arm in arms:
            cube = np.asarray([[counts[task, gap, arm, seed, f] for f in range(64)] for seed in seeds])
            per_family = 2 if task == 'overwrite' else 8
            if not (cube[..., 1] == per_family).all(): raise ValueError('wrong paired-question denominator')
            hits = cube[..., 0].sum(axis=1)
            rates[task, gap, arm] = 100 * hits / (64 * per_family)
            invalid[arm] += int(cube[..., 2].sum())
            cells.append(dict(task=task, gap=gap, arm=arm, correct_by_seed=hits.tolist(),
                              both_correct=int(hits.sum()), key_pairs=int(cube[..., 1].sum())))
    comparisons = {}
    for arm in ('fixed', 'midpoint'):
        delta = rates[*PRIMARY, arm] - rates[*PRIMARY, 'parent']
        short = {f'{t}:{d}': (rates[t, d, arm] - rates[t, d, 'parent']).tolist() for t, d in SHORT}
        passed = delta.min() > 0 and delta.mean() >= 5 and all(min(v) >= -1 for v in short.values()) and invalid[arm] == 0
        comparisons[arm] = dict(primary_mean_pp=float(delta.mean()), primary_by_seed_pp=delta.tolist(),
                                short_by_seed_pp=short, pass_gate=bool(passed))
    return dict(primary_against_parent=comparisons, nonfinite_by_arm=invalid, cells=cells)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family_csv', type=Path)
    p.add_argument('--midpoint', action='store_true', help='apply the independent §22.12 replication gate')
    p.add_argument('--schedule-csv', type=Path, help='audit the paid runs and the cross-seed donor assignment')
    args = p.parse_args()
    if args.midpoint and args.schedule_csv: p.error('the midpoint has no training schedule')
    result = analyze_midpoint(args.family_csv) if args.midpoint else analyze(args.family_csv)
    if args.schedule_csv:
        result['schedule_audit'] = audit_schedules(args.schedule_csv)
    print(json.dumps(result, indent=2, allow_nan=False))
