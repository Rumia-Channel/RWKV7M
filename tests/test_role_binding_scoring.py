"""Reference boundaries: public programs, hard banks, and paired/quartet scoring."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_role_binding import (
    Policy, Readout, evaluate_cell, gate_rows, load_registered_policy,
    paired_effect_rows, parse_public, predict_data, prepare, quartet_flags,
    run_program, select_followup,
)
from role_binding_data import NAME_TOKENS, OBJECT_TOKENS, question, statement
from rwkv7m.io.role_binding import save_policy
from rwkv7m.model.role_binding import Router


def marked_policy(*, decode=None):
    """A real marked-count head with an explicitly controlled local payload decoder."""
    decode = tuple(range(8)) if decode is None else decode
    input_weights = np.zeros((128, 129), np.float32)
    output_weights = np.zeros((129, 9), np.float32)
    for value, token in enumerate(OBJECT_TOKENS):
        input_weights[token, value] = 8.
        output_weights[value, 1 + decode[value]] = 4.
    output_bias = np.zeros(9, np.float32)
    output_bias[0] = 1.
    params = dict(input=input_weights, bias=np.zeros(129, np.float32),
                  out=output_weights, out_bias=output_bias)
    router = Router(np.asarray((8, 9), np.int32), np.asarray(sorted(NAME_TOKENS), np.int32))
    meta = dict(arm='marked-count', seed='42', weight_sha='none', updates='960', curriculum='mixed')
    return Policy(params, np.zeros(128, np.float32), np.ones(128, np.float32), meta, router, 'a' * 64)


def base(policy):
    return dict(arm=policy.arm, seed=42, curriculum=policy.meta['curriculum'],
                weight_sha='none', head_sha=policy.head_sha,
                form='midterm', surface='midterm', live=3, gap=0)


def public_packet(*, reads=2):
    return dict(blocks=[list(statement(0, 1, 2, (0, 1, 2)))],
                queries=[list(question(0, 1, (0, 1, 2))), list(question(1, 0, (0, 1, 2)))],
                reads=reads,
                cards=[dict(on=token, ask=list(question((i + 2) % 8, (i + 3) % 8, (0, 1, 2))))
                       for i, token in enumerate(OBJECT_TOKENS)])


def test_followup_abstention_is_not_negative_indexing():
    cards = np.arange(16, dtype=np.int32).reshape(2, 8)
    selected, active = select_followup(cards, np.asarray([[-1, 7], [0, -1]], np.int32))
    np.testing.assert_array_equal(selected, [[-1, 7], [8, -1]])
    np.testing.assert_array_equal(active, [[False, True], [True, False]])
    with pytest.raises(ValueError, match='object class'):
        select_followup(cards, np.asarray([[8, 0], [0, 0]], np.int32))
    with pytest.raises(ValueError, match='cards'):
        select_followup(-np.ones((2, 8), np.int32), np.zeros((2, 2), np.int32))


def test_public_packet_has_no_label_or_hidden_id_boundary():
    packet = public_packet()
    expected = parse_public(packet)
    packet['cards'] = packet['cards'][::-1]
    actual = parse_public(packet)
    for left, right in zip(expected[:3], actual[:3], strict=True):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match='only public'):
        parse_public(dict(packet, answers=[1, 2]))
    packet['cards'][0]['on'] = 0  # A label ID is not an actual public object token.
    with pytest.raises(ValueError, match='actual public object'):
        parse_public(packet)


@pytest.mark.parametrize('reads', [True, 0, -1, 2.0])
def test_invalid_read_counts_are_rejected(reads):
    packet = public_packet()
    packet['reads'] = reads
    with pytest.raises(ValueError, match='positive integer'):
        parse_public(packet)


def test_one_read_allows_empty_cards_but_multiread_does_not():
    packet = public_packet(reads=1)
    packet['cards'] = []
    assert parse_public(packet)[2].shape == (0, 8)
    packet['reads'] = 2
    with pytest.raises(ValueError, match='eight'):
        parse_public(packet)


def test_duplicate_cards_and_invalid_token_indices_are_rejected():
    packet = public_packet()
    packet['cards'][1] = packet['cards'][0].copy()
    with pytest.raises(ValueError, match='exactly once'):
        parse_public(packet)
    for token in (-1, 128, 1.5):
        packet = public_packet()
        packet['queries'][0][0] = token
        with pytest.raises(ValueError, match='queries'):
            parse_public(packet)


def test_streaming_marked_bank_rejects_bad_route_and_tracks_overwrite_outside_state():
    policy = marked_policy()
    tokens = np.asarray([statement(0, 1, 2, (0, 1, 2)), statement(0, 1, 4, (1, 2, 0)),
                         question(0, 1, (2, 0, 1)), [16] * 8], np.int32)
    reader = prepare(policy, tokens, np.asarray([[0, 1]], np.int32))
    trace = run_program(reader, np.asarray([[2, 3]], np.int32), np.empty((1, 0), np.int32), 1)
    np.testing.assert_array_equal(trace.final, [[4, -1]])
    np.testing.assert_array_equal(trace.sources[..., 0], [[1, -1]])
    assert trace.addresses[0, 1, 0] == -1
    assert reader.state._fields == ('bank', 'occupied')
    with pytest.raises(ValueError, match='out of bounds'):
        reader.read(np.asarray([[-1, 2]], np.int32), np.ones((1, 2), bool))
    with pytest.raises(ValueError, match='blocks'):
        prepare(policy, tokens, np.asarray([[-1]], np.int32))


def test_public_program_never_teaches_wrong_intermediate_answer():
    class RecordingReader:
        def __init__(self):
            self.calls = []

        def read(self, query_ids, active):
            self.calls.append((query_ids.copy(), active.copy()))
            answers = np.full(query_ids.shape, -1, np.int32)
            answers[active & (query_ids == 10)] = 2
            answers[active & (query_ids == 22)] = 7
            return Readout(answers, np.full_like(answers, -1), np.full_like(answers, -1))

    reader = RecordingReader()
    trace = run_program(reader, np.asarray([[10, 11]], np.int32),
                        np.arange(20, 28, dtype=np.int32)[None], 2)
    # Even if an external scorer's gold intermediate were 1, the actual read is 22, not 21.
    np.testing.assert_array_equal(trace.predictions, [[[2, 7], [-1, -1]]])
    np.testing.assert_array_equal(reader.calls[1][0], [[22, -1]])
    np.testing.assert_array_equal(reader.calls[1][1], [[True, False]])
    assert len(reader.calls) == 2
    reader = RecordingReader()
    trace = run_program(reader, np.asarray([[11, 11]], np.int32),
                        np.arange(20, 28, dtype=np.int32)[None], 3)
    assert len(reader.calls) == 1
    assert not trace.active[..., 1:].any()
    assert (trace.final == -1).all()


def test_actual_eval_and_oracle_rerun_are_separate_with_real_bank():
    decode = list(range(8))
    decode[1] = 2  # Root's correct stored object is decoded incorrectly, selecting card 2.
    policy = marked_policy(decode=decode)
    raw = np.asarray([statement(0, 1, 1, (0, 1, 2)),
                      statement(2, 3, 7, (1, 2, 0)),
                      statement(4, 5, 6, (2, 0, 1)),
                      question(0, 1, (0, 1, 2)),
                      question(2, 3, (0, 1, 2)),
                      question(4, 5, (0, 1, 2)),
                      question(1, 0, (0, 1, 2))], np.int32)
    cards = np.full((1, 8), 6, np.int32)
    cards[0, 1], cards[0, 2] = 5, 4
    data = dict(feature_tokens=raw, blocks=np.tile(np.arange(3, dtype=np.int32), (1, 4, 1)),
                queries=np.asarray([[3, 3]], np.int32), cards=cards, reads=2,
                answers=np.full((1, 4, 2), 6, np.int32),
                values=np.tile(np.asarray([1, 6], np.int32), (1, 4, 2, 1)),
                support=np.tile(np.asarray([0, 2], np.int32), (1, 4, 2, 1)),
                draw=np.asarray([766101], np.int32), family=np.asarray([0], np.int32),
                input_sha=np.asarray(['b' * 64]))
    rows, families, summary, steps = evaluate_cell(policy, data, base(policy))
    assert {row['first_hard'] for row in rows} == {2}
    assert {row['final_hard'] for row in rows} == {7}
    assert {row['oracle_terminal'] for row in rows} == {6}
    assert {row['source'] for row in rows if row['step'] == 1} == {1}
    assert summary['answer_count'] == summary['joint_count'] == 0
    assert summary['first_fail'] == 8 and summary['later_fail'] == 0
    assert summary['oracle_terminal_correct'] == 8
    assert summary['bank_reads'] == 16 and summary['one_step_bank_reads'] == 8
    assert steps[0]['source_correct'] == 8 and steps[1]['source_correct'] == 0
    assert all(row['joint_success'] == 0 for row in families)


def test_ordinary_selects_full_continuous_branch_encoding_not_local_query():
    policy = Policy(dict(input=np.eye(256, 1, dtype=np.float32), bias=np.zeros(1, np.float32),
                         out=np.asarray([[0., 0., 0., 0., 1., 0., 0., 0.]], np.float32),
                         out_bias=np.asarray([.5, 0., 0., 0., 0., 0., 0., 0.], np.float32)),
                    np.zeros(256, np.float32), np.ones(256, np.float32),
                    dict(arm='ordinary', seed='42', weight_sha='c' * 64, updates='960', curriculum='mixed'),
                    None, 'd' * 64)
    tokens = np.arange(88, dtype=np.int32).reshape(11, 8)
    continuous = np.zeros((1, 10, 256), np.float32)
    continuous[0, [0, 2], 0] = 1.
    reader = prepare(policy, tokens, np.asarray([[10]], np.int32), ordinary=continuous,
                     ordinary_queries=np.arange(10, dtype=np.int32)[None])
    trace = run_program(reader, np.asarray([[0, 1]], np.int32),
                        np.arange(2, 10, dtype=np.int32)[None], 2)
    np.testing.assert_array_equal(trace.predictions, [[[4, 0], [0, 4]]])
    np.testing.assert_array_equal(trace.queries, [[[0, 6], [1, 2]]])
    assert (trace.sources == -1).all()


def test_raw_count_prediction_needs_neither_neural_cache_nor_encoder():
    packet = public_packet(reads=1)
    packet['cards'] = []
    result = predict_data(marked_policy(), packet)
    assert result['answers'] == [2, -1]
    assert result['bank_reads'] == [1, 1]
    assert result['paths'][0][0]['ask'] == packet['queries'][0]


def test_quartet_not_pooled_success_and_joint_not_marginal_success():
    gold = np.zeros((2, 4, 2), np.int32)
    prediction = gold.copy()
    prediction[0, 0, 0] = 1
    prediction[1, 1, 1] = 1
    support = np.zeros((2, 4, 2, 2), np.int32)
    sources = support.copy()
    sources[0, 2, 1, 1] = 1
    answer, source, joint = quartet_flags(prediction, gold, sources, support)
    assert (prediction == gold).mean() == .875
    np.testing.assert_array_equal(answer, [[False, True], [True, False]])
    np.testing.assert_array_equal(source, [[True, False], [True, True]])
    np.testing.assert_array_equal(joint, [[False, False], [True, False]])
    assert answer.sum() == 2 and source.sum() == 3 and joint.sum() == 1


def test_elementary_pair_uses_world_specific_support():
    gold = np.asarray([[[2, 5], [5, 2]]], np.int32)
    support = np.asarray([[[[0], [1]], [[1], [0]]]], np.int32)
    answer, source, joint = quartet_flags(gold.copy(), gold, support.copy(), support)
    assert answer.all() and source.all() and joint.all()
    wrong = np.repeat(support[:, :1], 2, axis=1)
    _, source, joint = quartet_flags(gold.copy(), gold, wrong, support)
    assert not source.any() and not joint.any()


def test_joint_gate_uses_actual_quartet_joint_for_every_seed_and_gap():
    summaries = [dict(arm='marked-rwkv', form='midterm', surface='midterm', live=12,
                      seed=seed, gap=gap, family_questions=80,
                      answer_count=76, source_count=76, joint_count=71)
                 for seed in (42, 43, 44) for gap in (0, 128)]
    rows = list(gate_rows(summaries))
    assert all(row['answer_gate'] == row['source_gate'] == 1 for row in rows)
    assert all(row['joint_gate'] == 0 for row in rows)
    summaries[-1]['answer_count'] = 71
    rows = list(gate_rows(summaries))
    assert next(row for row in rows if row['seed'] == 'all')['answer_gate'] == 0


def test_duplicate_gaps_do_not_pass_every_seed_and_gap_gate():
    summaries = [dict(arm='marked-rwkv', form='midterm', surface='midterm', live=12,
                      seed=seed, gap=0, family_questions=80,
                      answer_count=80, source_count=80, joint_count=80)
                 for seed in (42, 43, 44) for _ in range(2)]
    assert all(row['joint_gate'] == 0 for row in gate_rows(summaries))


def test_paired_bootstrap_preserves_seed_group_and_family_questions():
    rows = []
    for seed in (42, 43, 44):
        for draw in (766101, 766103, 766107, 766109, 766113):
            for family in range(8):
                for question_id in range(2):
                    for arm, value in (('marked-rwkv', 1), ('learned', 0), ('marked-count', 0)):
                        rows.append(dict(form='midterm', surface='midterm', live=12, gap=0,
                                         arm=arm, seed=seed, draw=draw, family=family, question=question_id,
                                         answer_success=value, source_success=value, joint_success=value))
    effects = list(paired_effect_rows(rows, resamples=100))
    assert len(effects) == 6
    assert all(row['delta_pp'] == row['lower95_pp'] == row['upper95_pp'] == 100. for row in effects)
    assert all(row['seeds'] == 3 and row['families_per_draw'] == 8 for row in effects)
    controls = list(paired_effect_rows([row for row in rows if row['arm'] != 'learned'], resamples=100))
    assert len(controls) == 3
    assert all(row['reference'] == 'marked-count' for row in controls)


def test_registered_curriculum_and_fixed_update_count_cannot_be_reused(tmp_path):
    policy = marked_policy()
    basic = tmp_path / 'basic.safetensors'
    save_policy(basic, policy.params, policy.mean, policy.scale, arm=policy.arm,
                seed=42, weight_sha='none', updates=960, router=policy.router,
                curriculum='basic-only')
    with pytest.raises(ValueError, match='wrong registered'):
        load_registered_policy(basic, curriculum='mixed')
    assert load_registered_policy(basic, curriculum='basic-only').meta['curriculum'] == 'basic-only'
    old_updates = tmp_path / 'old-updates.safetensors'
    save_policy(old_updates, policy.params, policy.mean, policy.scale, arm=policy.arm,
                seed=42, weight_sha='none', updates=480, router=policy.router,
                curriculum='mixed')
    with pytest.raises(ValueError, match='wrong registered'):
        load_registered_policy(old_updates, curriculum='mixed')
