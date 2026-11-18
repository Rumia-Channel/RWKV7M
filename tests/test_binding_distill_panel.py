"""Contract tests for the late-write distill probe and its scorer.

The tests never read the probe's source text: they exercise the public
functions with synthetic readouts and with the tracked parent checkpoints.
"""
import csv
import json
import struct
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest


KEYS = tuple(range(16, 24))
NEAR = ((32, 33, 34, 35, 36, 37, 38, 39), (40, 41, 42, 43, 44, 45, 46, 47))
MIXED = ((36, 37, 32, 33, 36, 37, 38, 39), (44, 45, 40, 41, 44, 45, 46, 47))
OVERWRITE = ((32, 33), (40, 41))
REWRITTEN = (1, 1, 1, 1, 0, 0, 0, 0)


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return import_module('probe_binding_distill')


def make_rows(probe, *, arm='parent', seed=42, families=1, predictions=None, digest='a'*64):
    """The six fixed cases of one arm, with per-case predictions."""
    predictions = predictions or {}
    rows = []
    for task, gap in probe.CASES:
        if task == 'overwrite':
            truth, keys, flags = OVERWRITE, (16, 17), ((0, 0), (0, 0))
        else:
            truth = MIXED if task == 'mixed' else NEAR
            keys = KEYS
            flags = (tuple(REWRITTEN), tuple(REWRITTEN)) if task == 'mixed' else ((0,)*8, (0,)*8)
        pred = predictions.get((task, gap), truth)
        rows.append(dict(arm=arm, weight_seed=seed, families=families, task=task, gap=gap,
                         data_seed=731968 if task == 'overwrite' else 731967, family=0,
                         checkpoint_sha256=digest, input_sha256=f'{digest}{task}{gap}',
                         keys=' '.join(map(str, keys)),
                         truth0=' '.join(map(str, truth[0])), truth1=' '.join(map(str, truth[1])),
                         pred0=' '.join(map(str, pred[0])), pred1=' '.join(map(str, pred[1])),
                         rewritten0=' '.join(map(str, flags[0])),
                         rewritten1=' '.join(map(str, flags[1]))))
    return rows


def write_rows(probe, path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=probe.READOUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(probe, tmp_path, rows):
    path = tmp_path/'readouts.csv'
    write_rows(probe, path, rows)
    return probe.score(path, tmp_path)


def test_a_stale_world_never_earns_both_world_credit(probe, tmp_path):
    # World 0 answers correctly; world 1 answers with another queried key's value.
    stale = ((32, 33, 34, 35, 36, 37, 38, 39), (41, 41, 42, 43, 44, 45, 46, 47))
    retrieval, _ = score_rows(probe, tmp_path, make_rows(probe, predictions={('near', 0): stale}))
    near = retrieval[42, 'parent', 'near', 0, 'all']
    assert (near['key_pairs'], near['questions'], near['both_worlds'], near['correct']) == (8, 16, 7, 15)
    assert near['near_and_far'] == 7


def test_a_near_miss_in_the_far_case_cannot_borrow_the_near_answer(probe, tmp_path):
    # The near row is perfect, but the far row misses one key in both worlds.
    near_miss = ((33, 33, 34, 35, 36, 37, 38, 39), (41, 41, 42, 43, 44, 45, 46, 47))
    retrieval, _ = score_rows(probe, tmp_path, make_rows(probe, predictions={('change', 1024): near_miss}))
    near = retrieval[42, 'parent', 'near', 0, 'all']
    far = retrieval[42, 'parent', 'change', 1024, 'all']
    assert near['both_worlds'] == near['near_and_far'] == 8
    assert (far['both_worlds'], far['correct']) == (7, 14)
    assert far['near_and_far'] == 7 and far['paired_key_pairs'] == 8


def test_the_rewritten_subset_is_the_changed_truth_not_a_recent_ring(probe, tmp_path):
    # The stale answer keeps the pre-rewrite value of exactly the four changed keys.
    rows = make_rows(probe, predictions={('mixed', 0): NEAR, ('mixed', 1024): NEAR})
    retrieval, _ = score_rows(probe, tmp_path, rows)
    for gap in (0, 1024):
        rewritten = retrieval[42, 'parent', 'mixed', gap, 'rewritten']
        old = retrieval[42, 'parent', 'mixed', gap, 'old']
        every = retrieval[42, 'parent', 'mixed', gap, 'all']
        assert (rewritten['key_pairs'], old['key_pairs'], every['key_pairs']) == (4, 4, 8)
        assert rewritten['both_worlds'] == 0 and old['both_worlds'] == 4
        assert every['both_worlds'] == 4
    perfect, _ = score_rows(probe, tmp_path, make_rows(probe))
    assert perfect[42, 'parent', 'mixed', 1024, 'rewritten']['both_worlds'] == 4
    assert perfect[42, 'parent', 'mixed', 1024, 'old']['both_worlds'] == 4


def test_nonfinite_answers_are_counted_separately(probe, tmp_path):
    broken = ((-1, *NEAR[0][1:]), NEAR[1])
    retrieval, _ = score_rows(probe, tmp_path, make_rows(probe, predictions={('near', 0): broken}))
    near = retrieval[42, 'parent', 'near', 0, 'all']
    assert (near['nonfinite'], near['both_worlds'], near['correct']) == (1, 7, 15)


def test_an_incomplete_panel_is_rejected(probe, tmp_path):
    rows = make_rows(probe)[:-1]
    with pytest.raises(ValueError, match='incomplete panel'):
        score_rows(probe, tmp_path, rows)


def test_arms_must_read_the_same_public_inputs(probe, tmp_path):
    rows = make_rows(probe) + make_rows(probe, arm='ce')
    rows[-1]['input_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='different inputs'):
        score_rows(probe, tmp_path, rows)


def test_duplicate_or_unexpected_rows_are_rejected(probe, tmp_path):
    rows = make_rows(probe)
    with pytest.raises(ValueError, match='duplicate'):
        score_rows(probe, tmp_path, rows + [rows[0]])
    rows[0]['task'] = 'repeat'
    with pytest.raises(ValueError, match='unexpected'):
        score_rows(probe, tmp_path, rows)


def test_the_recorded_rewritten_flags_must_match_the_changed_truth(probe, tmp_path):
    rows = make_rows(probe)
    for row in rows:
        if row['task'] == 'mixed':
            row['rewritten1'] = '0 0 0 0 1 1 1 1'
    with pytest.raises(ValueError, match='rewritten flags disagree'):
        score_rows(probe, tmp_path, rows)


def test_the_overwrite_control_has_two_keys_and_no_equation_group(probe, tmp_path):
    retrieval, arithmetic = score_rows(probe, tmp_path, make_rows(probe))
    overwrite = retrieval[42, 'parent', 'overwrite', 32, 'all']
    assert (overwrite['key_pairs'], overwrite['questions'], overwrite['both_worlds']) == (2, 4, 2)
    assert overwrite['paired_key_pairs'] == 0 and overwrite['near_and_far'] == 0
    assert not any(key[2] == 'overwrite' for key in arithmetic)


def test_equations_score_both_worlds_with_the_shared_exact_solver(probe, tmp_path):
    retrieval, arithmetic = score_rows(probe, tmp_path, make_rows(probe))
    near = arithmetic[42, 'parent', 'near', 0, 24]
    assert near['program_pairs'] == 2
    assert near['both_outputs_both_worlds'] == near['a_both_worlds'] == near['acd_both_worlds'] == 2
    assert near['near_and_far'] == near['a_near_and_far'] == near['acd_near_and_far'] == 2
    assert near['invalid_worlds'] == 0 and near['both_targets_changed'] == 2
    assert retrieval[42, 'parent', 'near', 0, 'all']['both_worlds'] == 8


def test_an_invalid_operand_invalidates_the_equation_instead_of_scoring_a_hit(probe, tmp_path):
    invalid = ((31,)*8, (31,)*8)
    rows = make_rows(probe, predictions={('mixed', 1024): invalid})
    retrieval, arithmetic = score_rows(probe, tmp_path, rows)
    mixed = arithmetic[42, 'parent', 'mixed', 1024, 24]
    assert mixed['program_pairs'] == 2
    assert mixed['invalid_worlds'] == 4
    assert mixed['both_outputs_both_worlds'] == 0 and mixed['a_both_worlds'] == 0
    assert mixed['near_and_far'] == 0 and mixed['a_near_and_far'] == 0
    assert arithmetic[42, 'parent', 'mixed', 0, 24]['both_outputs_both_worlds'] == 2
    assert retrieval[42, 'parent', 'mixed', 1024, 'rewritten']['nonfinite'] == 0


def _write_fake_safetensors(path, metadata):
    """A header-only stand-in with the public metadata and parameter count."""
    header = {'__metadata__': {str(key): str(value) for key, value in metadata.items()},
              'weights': {'dtype': 'F32', 'shape': [504960], 'data_offsets': [0, 2019840]}}
    raw = json.dumps(header).encode('utf-8')
    Path(path).write_bytes(struct.pack('<Q', len(raw)) + raw)


def test_trained_checkpoints_must_declare_their_arm_seed_and_parent(probe, tmp_path):
    parents = tmp_path/'parents'
    checkpoints = tmp_path/'trained'
    parents.mkdir()
    checkpoints.mkdir()
    parent = parents/'long-s42.safetensors'
    _write_fake_safetensors(parent, {'use_screening': 'false', 'vocab_size': '128'})
    digest = sha256(parent.read_bytes()).hexdigest()
    published = {42: digest}
    trained = checkpoints/'ce-s42.safetensors'
    metadata = {'use_screening': 'false', 'vocab_size': '128', 'binding_arm': 'ce',
                'weight_seed': '42', 'parent_weights_sha256': digest}
    _write_fake_safetensors(trained, metadata)
    path, found, read, parameters = probe.resolve_checkpoint('ce', 42, checkpoints, parents, published)
    assert path == trained and found == sha256(trained.read_bytes()).hexdigest()
    assert parameters == 504960 and read['binding_arm'] == 'ce'
    assert probe.resolve_checkpoint('parent', 42, checkpoints, parents, published)[0] == parent

    for field, wrong in (('binding_arm', 'distill'), ('weight_seed', '43'),
                         ('parent_weights_sha256', '0'*64)):
        _write_fake_safetensors(trained, {**metadata, field: wrong})
        with pytest.raises(ValueError):
            probe.resolve_checkpoint('ce', 42, checkpoints, parents, published)
    with pytest.raises(ValueError):
        probe.resolve_checkpoint('parent', 42, checkpoints, parents, {42: '0'*64})
    with pytest.raises(FileNotFoundError):
        probe.resolve_checkpoint('distill', 42, checkpoints, parents, published)


def test_the_published_parent_digests_match_the_tracked_parents(probe):
    published = probe._published_parents()
    assert set(published) == set(probe.WEIGHT_SEEDS)
    for seed, digest in published.items():
        path = probe.DEFAULT_PARENTS/f'long-s{seed}.safetensors'
        assert sha256(path.read_bytes()).hexdigest() == digest


def test_the_public_mixed_subset_is_the_rewritten_keys_not_the_recent_ring(probe):
    pytest.importorskip('jax')
    from benchmark_binding_state import panel

    def writes(tokens):
        events = []
        for position in range(len(tokens)-2):
            marker, key, value = map(int, tokens[position:position+3])
            if marker == 2 and 16 <= key < 32 and 32 <= value < 48:
                events.append((key, value))
        return events

    cases = {(task, gap): (history, queries, labels)
             for task, gap, _, history, queries, labels
             in panel(8, probe.DIRECT_SEED, probe.OVERWRITE_SEED, guard_study=True)}
    _, queries, near_labels = cases['near', 0]
    mixed_history, _, mixed_labels = cases['mixed', 0]
    for family in range(8):
        keys = list(map(int, queries[family, 0, :, 1]))
        changed = [index for index in range(len(keys))
                   if mixed_labels[family, 0, index] != near_labels[family, 0, index]]
        assert len(changed) == 4
        assert [index for index in range(len(keys))
                if mixed_labels[family, 1, index] != near_labels[family, 1, index]] == changed
        flags = probe.rewritten_flags(mixed_labels[family], near_labels[family])
        assert flags[0] == flags[1] == tuple(int(index in changed) for index in range(len(keys)))
        rewritten = {keys[index] for index in changed}
        events = writes(mixed_history[family, 0])
        assert rewritten == {key for key, _ in events[-4:]}
        recent = {key for key, _ in events[-16:]} & set(keys)
        assert rewritten <= recent and len(recent) > len(rewritten)
        probe._check_mixed(mixed_history[family], keys, flags, 2)


def test_second_batch_keeps_its_own_counterfactual_truth(probe, tmp_path):
    pytest.importorskip('jax')
    # An on-device read of two real families, not two copies of the first one's labels.
    output = tmp_path/'two-batches'
    probe.measure(None, output, families=2, batch=1, seeds=(42,), arms=('parent',))
    observations = probe.observations(output/'readouts.csv')
    for (task,gap), (_,_,_,truth) in probe.public_cases(2).items():
        for family in (0,1):
            assert observations['parent',42,family,task,gap]['truth'] == tuple(tuple(w) for w in truth[family])
