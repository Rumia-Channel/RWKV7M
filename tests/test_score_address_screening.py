import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_address_screening import gates, scores


def test_equations_use_live_copy_chain_endpoints_and_both_worlds():
    keys = tuple(range(16, 24))
    focus = (1, 0, 1, 0, 1, 0, 1, 0)
    gold = ((32, 42, 34, 43, 36, 44, 38, 45), (38, 42, 32, 43, 34, 44, 36, 45))
    bad = (gold[0], (-2, *gold[1][1:]))
    poison_wrong = tuple(tuple(x if flag else -2 for x, flag in zip(w, focus)) for w in gold)
    data = {(44, 'isolated', 'version', 24, 0, 0): (keys, focus, gold, poison_wrong, 732013),
            (44, 'isolated', 'version', 24, 1024, 0): (keys, focus, gold, bad, 732013)}
    recall, equations = scores(data)
    assert equations[44, 'isolated', 'version', 24, 0]['both_outputs_both_worlds'] == 1
    assert equations[44, 'isolated', 'version', 24, 1024]['both_outputs_both_worlds'] == 0
    copied = recall[44, 'isolated', 'version', 24, 1024, 'copied']
    assert copied['both_worlds'] == 3
    assert copied['unbound'] == 1


def test_exact_margin_passes_but_one_weak_copy_seed_blocks_advancement():
    recall, equations = {}, {}
    for seed in (42, 43, 44):
        def observe(arm, case, depth, gap, subset, hits):
            recall[seed, arm, case, depth, gap, subset] = dict(both_worlds=hits, key_pairs=100, nonfinite=0)

        observe('isolated', 'mixed', 0, 1024, 'rewritten', 100)
        observe('isolated', 'mixed', 0, 1024, 'old', 90)
        observe('coupled', 'mixed', 0, 1024, 'old', 80)
        observe('isolated', 'version', 24, 1024, 'copied', 100)
        for case, gap in [('near', 0), ('overwrite', 32)]:
            observe('isolated', case, 0, gap, 'all', 98)
            observe('ordinary', case, 0, gap, 'all', 100)
        for arm, hits in [('isolated', 100), ('coupled', 90)]:
            equations[seed, arm, 'version', 24, 1024] = dict(both_outputs_both_worlds=hits, program_pairs=100)
    assert all(row[-1] for row in gates(recall, equations))
    recall[44, 'isolated', 'version', 24, 1024, 'copied']['both_worlds'] = 89
    assert not any(row[-1] for row in gates(recall, equations))
