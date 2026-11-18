import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from episodic_scene_data import (ACTOR_TOKENS, DEV_DRAWS, OBJECT_TOKENS, OCCASION_TOKENS,
                                 SPLITS, TEST_CELLS, TEST_DRAWS, TOKEN_TO_WORD, TRAIN_CELLS,
                                 TRAIN_DRAWS, Panel, gloss, make_panel)


def _domain_id(clause, domain):
    mentions = [int(token) for token in clause if int(token) in domain]
    assert len(mentions) == 1
    return domain.index(mentions[0])


def _cue(clause):
    return _domain_id(clause, ACTOR_TOKENS), _domain_id(clause, OCCASION_TOKENS)


def _fact(clause):
    return (*_cue(clause), _domain_id(clause, OBJECT_TOKENS))


def _words(clause):
    return {TOKEN_TO_WORD[int(token)] for token in clause}


def test_published_draws_and_cells_are_explicit():
    assert SPLITS == ('train', 'dev', 'seen', 'held')
    assert TRAIN_DRAWS == (752101, 752103, 752107, 752109)
    assert DEV_DRAWS == (753101, 753103)
    assert TEST_DRAWS == (754101, 754103, 754107, 754109, 754113)
    assert TRAIN_CELLS == ((4, 0), (4, 8), (8, 0), (8, 8))
    assert TEST_CELLS == ((4, 0), (4, 128), (16, 0), (16, 128))
    assert not set(TRAIN_DRAWS) & set(DEV_DRAWS)
    assert not set(TRAIN_DRAWS + DEV_DRAWS) & set(TEST_DRAWS)


@pytest.mark.parametrize('split', SPLITS)
def test_panel_shapes_dtypes_and_visible_vocabulary(split):
    cells = TRAIN_CELLS if split in ('train', 'dev') else TEST_CELLS
    for live, gap in cells:
        panel = make_panel(split, TEST_DRAWS[0], 3, live, gap)
        assert isinstance(panel, Panel)
        assert panel.blocks.shape == (3, 2, live + gap, 8)
        assert panel.queries.shape == (3, 2, 8)
        assert panel.support.shape == (3, 2)
        assert panel.answers.shape == (3, 2, 2)
        assert all(array.dtype == np.int32 for array in panel)
        for tokens in (panel.blocks, panel.queries):
            assert np.all((tokens >= 0) & (tokens < 128))
            assert set(tokens.reshape(-1)) <= set(TOKEN_TO_WORD)
        assert np.all((panel.support >= 0) & (panel.support < live))
        assert np.all((panel.answers >= 0) & (panel.answers < 8))


@pytest.mark.parametrize('overrides', (
    {'split': 'final'}, {'split': 'development'}, {'draw': -1}, {'families': 0},
    {'live': 3}, {'live': 33}, {'gap': -1},
    {'split': 'train', 'shift': True}, {'split': 'dev', 'shift': True},
))
def test_invalid_panels_are_rejected(overrides):
    arguments = dict(split='seen', draw=TEST_DRAWS[0], families=2, live=4, gap=0)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        make_panel(**arguments)


def test_draw_and_family_seeds_are_reproducible_without_batch_size_dependence():
    small = make_panel('seen', TEST_DRAWS[1], 2, 16, 17)
    repeat = make_panel('seen', TEST_DRAWS[1], 2, 16, 17)
    larger = make_panel('seen', TEST_DRAWS[1], 5, 16, 17)
    for first, again, extended in zip(small, repeat, larger, strict=True):
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(first, extended[:2])


@pytest.mark.parametrize('split', ('seen', 'held'))
def test_worlds_are_an_actual_two_fact_object_swap(split):
    live = 16
    panel = make_panel(split, TEST_DRAWS[2], 9, live, 19)
    np.testing.assert_array_equal(panel.answers[:, 1], panel.answers[:, 0, ::-1])
    assert np.all(panel.answers[:, 0] != panel.answers[:, 1])
    for family, (before, after) in enumerate(panel.blocks):
        support = panel.support[family]
        assert len(set(support)) == 2
        changed = before != after
        assert np.count_nonzero(changed) == 2
        np.testing.assert_array_equal(np.flatnonzero(changed.any(axis=-1)), np.sort(support))
        np.testing.assert_array_equal(np.bincount(before.reshape(-1), minlength=128),
                                      np.bincount(after.reshape(-1), minlength=128))
        facts = [_fact(clause) for clause in before[:live]]
        pairs = {(actor, occasion) for actor, occasion, _ in facts}
        assert len(pairs) == live
        target_pairs = [_cue(query) for query in panel.queries[family]]
        (a0, o0), (a1, o1) = target_pairs
        assert a0 != a1 and o0 != o1
        assert (a0, o1) in pairs and (a1, o0) in pairs
        for query, position in enumerate(support):
            assert _cue(before[position]) == target_pairs[query]
            assert _cue(after[position]) == target_pairs[query]
            assert _fact(before[position])[2] == panel.answers[family, 0, query]
            assert _fact(after[position])[2] == panel.answers[family, 1, query]
            column, = np.flatnonzero(changed[position])
            assert before[position, column] in OBJECT_TOKENS
            assert after[position, column] in OBJECT_TOKENS


@pytest.mark.parametrize('split', ('train', 'dev', 'seen'))
def test_semantic_holdout_excludes_facts_mentions_and_queries_in_nonheld_splits(split):
    panel = make_panel(split, TRAIN_DRAWS[0], 12, 32, 32)
    for family in panel.blocks:
        for world in family:
            pairs = [_cue(clause) for clause in world]
            assert len(set(pairs[:32])) == 32
            assert all((actor + occasion) % 4 != 0 for actor, occasion in pairs)
    assert all(sum(_cue(q)) % 4 != 0 for q in panel.queries.reshape(-1,8))


def test_held_targets_are_unseen_compositions_and_keep_both_rectangle_lures():
    panel = make_panel('held', TEST_DRAWS[3], 12, 32, 3)
    for family, questions in enumerate(panel.queries):
        (a0, o0), (a1, o1) = [_cue(question) for question in questions]
        assert (a0 + o0) % 4 == 0 and (a1 + o1) % 4 == 0
        assert a0 != a1 and o0 != o1
        pairs = {_cue(clause) for clause in panel.blocks[family, 0, :32]}
        assert len(pairs) == 32
        assert {(a0, o0), (a1, o1), (a0, o1), (a1, o0)} <= pairs


def test_no_label_raw_copy_store_opcode_query_marker_or_target_only_history_form():
    panel = make_panel('train', TRAIN_DRAWS[1], 8, 8, 32)
    # All visible IDs are disjoint from output classes, including old opcodes 2/3/4.
    assert np.all(panel.blocks >= 8) and np.all(panel.queries >= 8)
    assert not np.isin(panel.queries, OBJECT_TOKENS).any()
    cues = set(ACTOR_TOKENS + OCCASION_TOKENS)
    history_grammar = {'kept', 'the', 'at', 'with', 'care', 'discussed', 'interest', 'noticed'}
    query_grammar = {'which', 'object', 'did', 'keep', 'during', '?'}
    assert history_grammar.isdisjoint(query_grammar)
    for clause in panel.blocks.reshape(-1, 8):
        _fact(clause)  # Every clause mentions exactly one actor, occasion, and object.
        words = {TOKEN_TO_WORD[int(token)] for token in clause
                 if int(token) not in cues | set(OBJECT_TOKENS)}
        assert words <= history_grammar
    for question in panel.queries.reshape(-1, 8):
        _cue(question)
        words = {TOKEN_TO_WORD[int(token)] for token in question if int(token) not in cues}
        assert words == query_grammar
    for family in panel.blocks:
        for world in family:
            for clause in world[:8]:
                assert _words(clause) & history_grammar == {'kept', 'the', 'at', 'with', 'care'}
            for clause in world[8:]:
                words = _words(clause)
                assert 'kept' not in words and 'care' not in words
                assert len(words & {'discussed', 'noticed'}) == 1
                assert {'the', 'at', 'with', 'interest'} <= words
    # A query-only or token-histogram-only prediction cannot satisfy both worlds.
    assert np.all(panel.answers[:, 0] != panel.answers[:, 1])


@pytest.mark.parametrize('split', ('seen', 'held'))
def test_gap_is_a_prefix_extension_and_shift_changes_question_phrasing_only(split):
    live = 16
    base = make_panel(split, TEST_DRAWS[4], 4, live, 0)
    middle = make_panel(split, TEST_DRAWS[4], 4, live, 3)
    long = make_panel(split, TEST_DRAWS[4], 4, live, 128)
    shifted = make_panel(split, TEST_DRAWS[4], 4, live, 128, shift=True)
    np.testing.assert_array_equal(base.blocks, long.blocks[:, :, :live])
    np.testing.assert_array_equal(middle.blocks, long.blocks[:, :, :live + 3])
    for panel in (middle, long):
        np.testing.assert_array_equal(base.queries, panel.queries)
        np.testing.assert_array_equal(base.support, panel.support)
        np.testing.assert_array_equal(base.answers, panel.answers)
    np.testing.assert_array_equal(shifted.blocks, long.blocks)
    np.testing.assert_array_equal(shifted.support, long.support)
    np.testing.assert_array_equal(shifted.answers, long.answers)
    assert np.all(np.any(shifted.queries != long.queries, axis=-1))
    # Exactly the same lexemes, including opaque cues; only their order changes.
    np.testing.assert_array_equal(np.sort(shifted.queries, axis=-1),
                                  np.sort(long.queries, axis=-1))
    for original, changed in zip(long.queries.reshape(-1, 8), shifted.queries.reshape(-1, 8)):
        assert _cue(original) == _cue(changed)
        assert TOKEN_TO_WORD[int(changed[2])] == 'during'
        assert TOKEN_TO_WORD[int(changed[4])] == 'did'


def test_live_size_does_not_drive_queries_values_or_background_surfaces():
    short = make_panel('seen', TEST_DRAWS[0], 4, 4, 32)
    long = make_panel('seen', TEST_DRAWS[0], 4, 16, 32)
    np.testing.assert_array_equal(short.queries, long.queries)
    np.testing.assert_array_equal(short.answers, long.answers)
    np.testing.assert_array_equal(short.blocks[:, :, 4:], long.blocks[:, :, 16:])
    for short_family, long_family in zip(short.blocks, long.blocks, strict=True):
        for short_world, long_world in zip(short_family, long_family, strict=True):
            assert {_fact(clause) for clause in short_world[:4]} <= {
                _fact(clause) for clause in long_world[:16]}


def test_training_has_both_statement_and_question_orders_and_an_accessible_gloss():
    statement_starts, query_starts = set(), set()
    for draw in TRAIN_DRAWS:
        panel = make_panel('train', draw, 4, 8, 8)
        for clause in panel.blocks[:, 0, :8].reshape(-1, 8):
            statement_starts.add('actor' if clause[0] in ACTOR_TOKENS else TOKEN_TO_WORD[int(clause[0])])
        for query in panel.queries.reshape(-1, 8):
            query_starts.add(TOKEN_TO_WORD[int(query[0])])
    assert statement_starts == {'actor', 'at'}
    assert query_starts == {'which', 'during'}
    example = make_panel('seen', TEST_DRAWS[0], 1, 4, 0)
    words = gloss(example.blocks[0, 0, 0]).split()
    assert len(words) == 8
    assert 'kept' in words and 'care' in words
