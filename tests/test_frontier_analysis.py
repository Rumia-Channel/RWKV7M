import csv
from importlib import import_module
from pathlib import Path

import pytest


def records():
    rows = []
    for task, gap in [('other-writes', 128), ('filler', 0), ('overwrite', 32)]:
        for seed in [42, 43, 44]:
            for arm in ['parent', 'fixed', 'adaptive', 'yoked']:
                for family in [0, 1]:
                    hits = 7 if task == 'other-writes' and arm == 'adaptive' else 5
                    rows.append(dict(task=task, gap=gap, weight_seed=seed, arm=arm, family=family,
                                     both_correct=hits, key_pairs=8, nonfinite=0))
    return rows


@pytest.fixture
def score(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    analyze = import_module('analyze_retrieval_frontier').analyze

    def run(rows):
        p = tmp_path / 'counts.csv'
        with p.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        return analyze(p)
    return run


def test_positive_average_cannot_hide_one_regressing_seed(score):
    rows = records()
    assert score(rows)['adopt_adaptive']
    for row in rows:
        if (row['arm'], row['task'], row['weight_seed']) == ('adaptive', 'other-writes', 44):
            row['both_correct'] = 4
    result = score(list(reversed(rows)))
    assert result['primary_adaptive_minus']['fixed']['mean_pp'] == pytest.approx(12.5)
    assert result['primary_adaptive_minus']['fixed']['by_seed_pp'] == [25, 25, -12.5]
    assert not result['adopt_adaptive']


def test_invalid_predictions_in_a_secondary_condition_disqualify_adoption(score):
    rows = records()
    next(r for r in rows if r['arm'] == 'adaptive' and r['task'] == 'overwrite')['nonfinite'] = 1
    result = score(rows)
    assert result['superiority_pass'] and result['retention_pass']
    assert not result['adopt_adaptive']


def test_empirical_horizon_stops_at_first_failed_gap_in_any_seed(score):
    rows = records()
    extra = []
    for r in rows:
        if r['task'] == 'filler':
            r['both_correct'] = 8
            extra.append(r | {'gap': 64, 'both_correct': 7 if r['weight_seed'] == 44 else 8})
            extra.append(r | {'gap': 128})
    grid = score(rows + extra)['empirical_90pct_grid_all_seeds']
    assert grid['filler']['parent'] == {'last_passing_tested_gap': 0, 'first_failing_tested_gap': 64}
    assert 'other-writes' not in grid  # This fixture does not evaluate its zero-distance baseline.


def test_schedule_audit_distinguishes_self_replay_from_cross_seed(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    audit = import_module('analyze_retrieval_frontier').audit_schedules
    path = tmp_path / 'updates.csv'
    cells = [(g, w) for g in (32, 64, 128) for w in (0, 4, 8)]
    def write_replay(self_replay):
        with path.open('w', newline='') as f:
            out = csv.writer(f)
            out.writerow(('arm', 'weight_seed', 'step', 'anchor', 'gap', 'writes'))
            for i, s in enumerate((42, 43, 44)):
                for arm in ('adaptive', 'fixed', 'yoked'):
                    for step in range(96):
                        anchor = step % 4 == 3
                        index = step - step // 4
                        n = index if arm == 'fixed' else i if arm == 'adaptive' or self_replay else (i + 1) % 3
                        cell = (32, 0) if anchor else cells[n % 9]
                        out.writerow((arm, s, step, int(anchor), *cell))
    write_replay(False)
    assert audit(path)['adaptive_matching_nonanchor'][42] == {'fixed': 8, 'yoked': 0}
    write_replay(True)
    with pytest.raises(ValueError, match='registered donor'):
        audit(path)


def test_unmatched_histories_cannot_enter_a_paired_comparison(score):
    rows = records()
    for row in rows:
        hits = row['both_correct']
        row.update(input_sha256='common', world0_labels=' '.join(['32'] * 8), world1_labels=' '.join(['33'] * 8),
                   world0_predictions=' '.join(['32'] * 8), world1_predictions=' '.join(['33'] * hits + ['34'] * (8 - hits)))
    rows[-1]['input_sha256'] = 'different'
    with pytest.raises(ValueError, match='unpaired histories'):
        score(rows)
