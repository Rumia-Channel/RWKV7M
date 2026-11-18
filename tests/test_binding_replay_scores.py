from collections import Counter
import csv
from importlib import import_module
from pathlib import Path

import pytest


KEYS = tuple(range(16, 24))
TRUTH = (tuple(range(32, 40)), tuple(range(40, 48)))
MIXED = tuple((world[3], *world[:3], *world[4:]) for world in TRUTH)
REWRITTEN = (1, 1, 1, 1, 0, 0, 0, 0)


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return import_module('score_binding_replay')


def observation(truth=TRUTH, *, keys=KEYS, recent=None, mixed=False, **predictions):
    values = dict(truth=truth, guard=truth, replay=truth, fresh=truth,
                  recent=(1,)*len(keys) if recent is None else recent,
                  hit=(1,)*len(keys))
    if mixed:
        values['tail'] = truth
    values.update(predictions)
    return 901, keys, values


def mixed_data(**far_predictions):
    return {(44, 0, 'near', 0): observation(),
            (44, 0, 'mixed', 0): observation(MIXED, recent=(1,)*8, mixed=True),
            (44, 0, 'mixed', 1024): observation(MIXED, recent=REWRITTEN, mixed=True, **far_predictions)}


def test_pairing_requires_same_key_both_worlds_and_matched_family(scorer):
    data = {}
    for family in (0, 1):
        programs = scorer._programs(901, family, KEYS, 24)

        def partial(group):
            return TRUTH[0], tuple(value if key in group else -1
                                   for key, value in zip(KEYS, TRUTH[1], strict=True))

        data[44, family, 'near', 0] = observation(replay=partial(programs[0].keys))
        data[44, family, 'change', 1024] = observation(replay=partial(programs[1].keys))
        short_truth = tuple(world[:2] for world in TRUTH)
        data[44, family, 'overwrite', 32] = observation(short_truth, keys=KEYS[:2])
    retrieval, arithmetic, _ = scorer.summarize(data)
    far = retrieval[44, 'change', 1024, 'replay', 'all']
    assert (far['key_pairs'], far['both_worlds'], far['near_and_far'], far['nonfinite']) == (16, 8, 0, 8)
    maths = arithmetic[44, 'change', 1024, 24, 'replay']
    assert (maths['program_pairs'], maths['both_outputs_both_worlds'], maths['near_and_far'],
            maths['invalid_worlds']) == (4, 2, 0, 2)
    assert maths['a_near_and_far'] == maths['acd_near_and_far'] == 0
    assert arithmetic[44, 'change', 1024, 24, 'guard']['near_and_far'] == 4
    overwrite = retrieval[44, 'overwrite', 32, 'replay', 'all']
    assert (overwrite['key_pairs'], overwrite['questions'], overwrite['both_worlds']) == (4, 8, 4)
    assert overwrite['paired_key_pairs'] == 0 and 'near_and_far' not in overwrite
    assert all(key[1] != 'overwrite' and key[3] == 24 for key in arithmetic)


def test_mixed_pairing_uses_matched_mixed_near_not_original_or_its_recent_ring(scorer):
    data = mixed_data()
    assert sum(data[44, 0, 'mixed', 0][2]['recent']) > 4
    retrieval, arithmetic, errors = scorer.summarize(data)
    for gap in (0, 1024):
        all_keys = retrieval[44, 'mixed', gap, 'replay', 'all']
        rewritten = retrieval[44, 'mixed', gap, 'replay', 'rewritten']
        old = retrieval[44, 'mixed', gap, 'replay', 'old']
        assert all_keys['both_worlds'] == all_keys['near_and_far'] == 8
        assert rewritten['key_pairs'] == rewritten['both_worlds'] == rewritten['near_and_far'] == 4
        assert old['key_pairs'] == old['both_worlds'] == old['near_and_far'] == 4
        assert errors[44, gap, 'replay']['questions'] == errors[44, gap, 'replay']['latest'] == 8
        assert arithmetic[44, 'mixed', gap, 24, 'replay']['near_and_far'] == 2
        assert retrieval[44, 'mixed', gap, 'tail', 'all']['both_worlds'] == 8
    assert not any(key[1] == 'near' and key[3] == 'tail' for key in retrieval)


@pytest.mark.parametrize('prediction, kind', [
    (35, 'latest'), (32, 'own_obsolete'), (33, 'another_queried'),
    (40, 'distractor_value'), (47, 'distractor_value'),
    (31, 'nonvalue'), (48, 'nonvalue'), (-1, 'nonvalue'),
])
def test_same_key_stale_precedes_other_queried_and_all_legal_values_count(scorer, prediction, kind):
    # 32 is stale for this key but is also another queried key's current value.
    # 40 is queried in the other world, not this one: it is a legal distractor here.
    assert scorer.prediction_kind(prediction, 35, 32, set(MIXED[0])) == kind


def test_mixed_recent_errors_count_each_world_and_keep_nonfinite_misses(scorer):
    replay = ((32, 36, 47, -1, *MIXED[0][4:]), MIXED[1])
    retrieval, _, errors = scorer.summarize(mixed_data(replay=replay), equations=False)
    row = errors[44, 1024, 'replay']
    assert row == dict(families=1, key_pairs=4, questions=8, latest=4, own_obsolete=1,
                       another_queried=1, distractor_value=1, nonvalue=1, nonfinite=1)
    assert sum(row[kind] for kind in scorer.ERROR_KINDS) == row['questions']
    pairs = retrieval[44, 'mixed', 1024, 'replay', 'rewritten']
    assert (pairs['both_worlds'], pairs['correct'], pairs['nonfinite']) == (0, 4, 1)
    # Fresh is perfect, but the supplied blind replay must not be recomputed from labels or hits.
    assert retrieval[44, 'mixed', 1024, 'fresh', 'rewritten']['both_worlds'] == 4


def test_far_recent_flags_must_agree_with_genuinely_rewritten_keys(scorer):
    data = mixed_data()
    data[44, 0, 'mixed', 1024][2]['recent'] = (1,)*8
    with pytest.raises(ValueError, match='recent flags disagree'):
        scorer.summarize(data, equations=False)


def test_missing_prediction_vector_is_not_dropped_from_denominator(scorer, tmp_path):
    row = dict(weight_seed=44, task='near', gap=0, data_seed=901, family=0,
               keys=' '.join(map(str, KEYS)), recent='1 '*8, hit='1 '*8,
               input_sha256='a'*64, tail0='', tail1='')
    for arm in ('truth', *scorer.ARMS):
        for world in (0, 1):
            row[f'{arm}{world}'] = ' '.join(map(str, TRUTH[world]))
    row['replay0'] = ''
    path = tmp_path/'readouts.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scorer.COLUMNS)
        writer.writeheader(); writer.writerow(row)
    with pytest.raises(ValueError, match='expected 8 entries in replay0'):
        scorer.observations(path)


def test_gate_uses_absolute_boundaries_and_unchanged_old_predictions(scorer):
    retrieval = {}
    conditions = (('near', 0, 'all'), ('overwrite', 32, 'all'), ('change', 1024, 'all'),
                  ('mixed', 1024, 'rewritten'), ('mixed', 1024, 'old'))
    for seed in scorer.WEIGHT_SEEDS:
        for task, gap, subset in conditions:
            for arm in ('guard', 'replay'):
                retrieval[seed, task, gap, arm, subset] = Counter(
                    key_pairs=100, paired_key_pairs=100, both_worlds=50, near_and_far=50,
                    different_from_guard=0, nonfinite=0)
            candidate = retrieval[seed, task, gap, 'replay', subset]
            if subset == 'rewritten':
                candidate['both_worlds'] = 60 if seed == 44 else 48
            elif subset == 'all':
                candidate['both_worlds'] = candidate['near_and_far'] = 48
    assert all(gate['passed'] for gate in scorer.gate_results(retrieval).values())
    uniform = scorer.gate_results(retrieval, uniform_gain=True)
    assert not uniform[42]['passed'] and not uniform[43]['passed'] and uniform[44]['passed']
    retrieval[42, 'mixed', 1024, 'replay', 'rewritten']['both_worlds'] = 60
    assert scorer.gate_results(retrieval, uniform_gain=True)[42]['passed']
    retrieval[43, 'mixed', 1024, 'replay', 'old']['different_from_guard'] = 1
    retrieval[44, 'mixed', 1024, 'replay', 'rewritten']['both_worlds'] = 59
    gates = scorer.gate_results(retrieval)
    assert gates[42]['passed'] and not gates[43]['passed'] and not gates[44]['passed']


def test_calibrated_arm_uses_frozen_checkpoint_flag_for_every_heldout_row(scorer):
    evaluation = {}
    bad = ((-1,)*8, (-1,)*8)
    for seed in scorer.WEIGHT_SEEDS:
        for (_, family, task, gap), (draw, keys, values) in mixed_data().items():
            evaluation[seed, family, task, gap] = draw+10, keys, {**values, 'replay': bad}
    flags = {42: False, 43: False, 44: True}
    calibrated = scorer.calibrated_data(evaluation, flags)
    for key, (_, _, values) in calibrated.items():
        original = evaluation[key][2]
        assert values['calibrated'] is original['replay' if flags[key[0]] else 'guard']
        assert 'calibrated' not in original
        assert values['replay'] is original['replay']
    flags[44] = False  # The selected predictions are already frozen, not a live selector.
    retrieval, _, _ = scorer.summarize(calibrated, equations=False)
    assert retrieval[42, 'mixed', 1024, 'calibrated', 'all']['both_worlds'] == 8
    assert retrieval[44, 'mixed', 1024, 'calibrated', 'all']['nonfinite'] == 16


def test_calibration_requires_disjoint_data_seeds_and_matching_checkpoints(scorer):
    key = 42, 0, 'near', 0
    development = {key: observation()}
    _, keys, values = observation()
    evaluation = {key: (902, keys, values)}
    assert scorer.calibration_sources(development, evaluation) == ((901,), (902,))
    with pytest.raises(ValueError, match='data seeds overlap'):
        scorer.calibration_sources(development, development)
    with pytest.raises(ValueError, match='checkpoint sets differ'):
        scorer.calibration_sources(development, {(43, 0, 'near', 0): evaluation[key]})
    with pytest.raises(ValueError, match='flags do not match'):
        scorer.calibrated_data(evaluation, {43: True})
