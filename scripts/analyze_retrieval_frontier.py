"""Apply the preregistered §22.10 decision to published per-family key-pair counts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = ('parent', 'fixed', 'adaptive', 'yoked')
PRIMARY = ('other-writes', 128)
SHORT = (('filler', 0), ('overwrite', 32))


def analyze(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for r in rows:
        key = (r['task'], int(r['gap']), r['arm'], int(r['weight_seed']), int(r['family']))
        hits, total, invalid = (int(r[k]) for k in ('both_correct', 'key_pairs', 'nonfinite'))
        if key in counts or key[2] not in ARMS or not (0 <= hits <= total and total > 0 and 0 <= invalid <= 2 * total):
            raise ValueError(f'invalid or duplicate result cell: {key}')
        counts[key] = hits, total, invalid
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family_csv', type=Path)
    print(json.dumps(analyze(p.parse_args().family_csv), indent=2, allow_nan=False))
