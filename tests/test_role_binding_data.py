import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from role_binding_data import (ASK, CONSIDER, DEV_DRAWS, GA, HAND, NAME_TOKENS, NI,
                               OBJECT_TOKENS, PERIOD, SEEN_ORDERS, SHIFT_ORDERS,
                               SPLITS, TEST_CELLS, TEST_DRAWS, TOKEN_TO_WORD,
                               TRAIN_CELLS, TRAIN_DRAWS, WHAT, WO, Panel, gloss,
                               make_panel, question, statement)


# Parse only the public raw grammar, never generator metadata or private helpers.
def _read(row):
    row = tuple(map(int, row))
    assert len(row) == 8
    markers = tuple(row[column] for column in (1, 3, 5))
    assert set(markers) == {GA, NI, WO}
    roles = {row[column + 1]: row[column] for column in (0, 2, 4)}
    assert roles[GA] in NAME_TOKENS and roles[NI] in NAME_TOKENS
    giver, recipient = NAME_TOKENS.index(roles[GA]), NAME_TOKENS.index(roles[NI])
    assert giver != recipient
    assert row[6] in (HAND, CONSIDER)
    assert row[7] in (PERIOD, ASK)
    if row[7] == ASK:
        assert row[6] == HAND and roles[WO] == WHAT
        obj = None
    else:
        assert roles[WO] in OBJECT_TOKENS
        obj = OBJECT_TOKENS.index(roles[WO])
    order = tuple((GA, NI, WO).index(marker) for marker in markers)
    return (giver, recipient, obj), order, row[6]


def _distance(pair):
    a, b = pair
    return min((a - b) % 8, (b - a) % 8)


def _facts(rows):
    result = {}
    for position, row in enumerate(rows):
        (giver, recipient, obj), _, predicate = _read(row)
        assert predicate == HAND and obj is not None
        assert (giver, recipient) not in result
        result[giver, recipient] = (obj, position)
    return result


def test_published_draws_cells_vocabulary_and_parity_partition():
    assert SPLITS == ('train', 'dev', 'seen', 'held')
    assert TRAIN_DRAWS == (761101, 761103, 761107, 761109)
    assert DEV_DRAWS == (762101, 762103)
    assert TEST_DRAWS == (763101, 763103, 763107, 763109, 763113)
    assert TRAIN_CELLS == ((4, 0), (4, 8), (8, 0), (8, 8))
    assert TEST_CELLS == ((4, 0), (4, 128), (16, 0), (16, 128))
    assert not set(TRAIN_DRAWS) & set(DEV_DRAWS)
    assert not set(TRAIN_DRAWS + DEV_DRAWS) & set(TEST_DRAWS)
    assert len(NAME_TOKENS) == len(OBJECT_TOKENS) == 8
    lexemes = (*NAME_TOKENS, *OBJECT_TOKENS, GA, NI, WO, HAND, CONSIDER, PERIOD, WHAT, ASK)
    assert len(set(lexemes)) == 24
    assert set(lexemes) == set(TOKEN_TO_WORD)
    assert all(8 <= token < 125 for token in lexemes)
    assert len(set(TOKEN_TO_WORD.values())) == 24
    assert SEEN_ORDERS == ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    assert set(SHIFT_ORDERS) == {(0, 2, 1), (2, 1, 0), (1, 0, 2)}
    assert not set(SEEN_ORDERS) & set(SHIFT_ORDERS)
    withheld = {(a, b) for a in range(8) for b in range(a + 1, 8) if _distance((a, b)) == 2}
    assert len(withheld) == 8


@pytest.mark.parametrize('order', SEEN_ORDERS + SHIFT_ORDERS)
def test_public_renderers_and_readable_japanese_gloss(order):
    fact, surface, predicate = _read(statement(7, 0, 4, order))
    assert (fact, surface, predicate) == ((7, 0, 4), order, HAND)
    assert _read(statement(7, 0, 4, order, binding=False)) == ((7, 0, 4), order, CONSIDER)
    assert _read(question(7, 0, order)) == ((7, 0, None), order, HAND)
    assert 'handed' in gloss(statement(0, 1, 2, order))
    assert 'アロ' in gloss(statement(0, 1, 2, order), japanese=True)
    assert 'リボン' in gloss(statement(0, 1, 2, order), japanese=True)
    assert '渡そうと考えた' in gloss(statement(0, 1, 2, order, binding=False), japanese=True)
    assert '何' in gloss(question(0, 1, order), japanese=True)


@pytest.mark.parametrize('kind,shift', [(kind, False) for kind in SPLITS] + [('held', True)])
def test_shapes_dtypes_and_reserved_ids(kind, shift):
    cells = TRAIN_CELLS if kind in ('train', 'dev') else TEST_CELLS
    for live, gap in cells:
        panel = make_panel(kind, TEST_DRAWS[0], 3, live, gap, shift=shift)
        assert isinstance(panel, Panel)
        assert panel.blocks.shape == (3, 2, live + gap, 8)
        assert panel.queries.shape == (3, 2, 8)
        assert panel.support.shape == panel.answers.shape == (3, 2, 2)
        assert all(array.dtype == np.int32 for array in panel)
        for tokens in (panel.blocks, panel.queries):
            assert set(tokens.reshape(-1)) <= set(TOKEN_TO_WORD)
            assert np.all((tokens >= 8) & (tokens < 125))
        assert np.all((panel.support >= 0) & (panel.support < live))
        assert np.all((panel.answers >= 0) & (panel.answers < 8))


@pytest.mark.parametrize('overrides', (
    {'kind': 'shift'}, {'kind': 'unknown'}, {'draw': -1}, {'families': 0},
    {'live': 2}, {'live': 5}, {'live': 42}, {'kind': 'held', 'live': 58}, {'gap': -1},
    {'kind': 'train', 'shift': True}, {'kind': 'dev', 'shift': True}, {'shift': True},
))
def test_invalid_panels_are_rejected(overrides):
    arguments = dict(kind='seen', draw=TEST_DRAWS[0], families=2, live=4, gap=0)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        make_panel(**arguments)


def test_public_renderers_reject_invalid_ids_and_orders():
    for arguments in ((8, 0, 0, (0, 1, 2)), (0, 1, -1, (0, 1, 2)), (0, 1, 0, (0, 1, 1))):
        with pytest.raises(ValueError):
            statement(*arguments)
    with pytest.raises(ValueError):
        question(0, 8, (0, 1, 2))
    with pytest.raises(TypeError):
        make_panel('seen', 1.5, 2, 4, 0)


def test_reproducibility_is_independent_of_family_batch_size():
    first = make_panel('held', TEST_DRAWS[1], 2, 16, 17, shift=True)
    again = make_panel('held', TEST_DRAWS[1], 2, 16, 17, shift=True)
    extended = make_panel('held', TEST_DRAWS[1], 5, 16, 17, shift=True)
    for a, b, c in zip(first, again, extended, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c[:2])


@pytest.mark.parametrize('kind,shift', [(kind, False) for kind in SPLITS] + [('held', True)])
@pytest.mark.parametrize('live', (4, 8, 16))
def test_raw_worlds_preserve_each_window_bag_but_reverse_directed_truth(kind, shift, live):
    panel = make_panel(kind, TEST_DRAWS[2], 8, live, 7, shift=shift)
    for family, worlds in enumerate(panel.blocks):
        queries = [_read(row)[0][:2] for row in panel.queries[family]]
        assert queries[0] == queries[1][::-1]
        assert (_distance(queries[0]) == 2) == (kind == 'held')
        target_names = {NAME_TOKENS[name] for name in queries[0]}
        changed_rows = set(np.flatnonzero(np.any(worlds[0] != worlds[1], axis=-1)))
        assert changed_rows == set(panel.support[family, 0])
        for position, (before, after) in enumerate(zip(*worlds, strict=True)):
            assert Counter(before) == Counter(after)
            changed_columns = set(np.flatnonzero(before != after))
            if position in changed_rows:
                assert len(changed_columns) == 2 and changed_columns <= {0, 2, 4}
                for column in changed_columns:
                    assert {before[column], after[column]} == target_names
            else:
                assert not changed_columns
        for world in range(2):
            facts = _facts(worlds[world, :live])
            assert len(facts) == live
            assert {key[::-1] for key in facts} == set(facts)
            assert {giver for giver, _ in facts} == {recipient for _, recipient in facts}
            unordered = {frozenset(pair) for pair in facts}
            assert len(unordered) == live // 2
            target = frozenset(queries[0])
            assert any(pair != target and len(pair & target) == 1 for pair in unordered)
            for query, key in enumerate(queries):
                obj, position = facts[key]
                assert panel.answers[family, world, query] == obj
                assert panel.support[family, world, query] == position
            for row in worlds[world, live:]:
                assert _read(row)[2] == CONSIDER
        np.testing.assert_array_equal(panel.support[family, 1], panel.support[family, 0, ::-1])
        np.testing.assert_array_equal(panel.answers[family, 1], panel.answers[family, 0, ::-1])
        assert panel.answers[family, 0, 0] != panel.answers[family, 0, 1]


@pytest.mark.parametrize('kind,draws,families,cells', (
    ('train', TRAIN_DRAWS, 24, TRAIN_CELLS),
    ('dev', DEV_DRAWS, 8, TRAIN_CELLS),
    ('seen', TEST_DRAWS, 8, TEST_CELLS),
))
def test_nonmember_pairs_never_leak_into_any_nonheld_utterance(kind, draws, families, cells):
    givers, recipients = set(), set()
    for draw in draws:
        for live, gap in cells:
            panel = make_panel(kind, draw, families, live, gap)
            for rows in (panel.blocks.reshape(-1, 8), panel.queries.reshape(-1, 8)):
                for row in rows:
                    (giver, recipient, _), order, _ = _read(row)
                    assert _distance((giver, recipient)) != 2
                    assert order in SEEN_ORDERS
                    givers.add(giver)
                    recipients.add(recipient)
    assert givers == recipients == set(range(8))


@pytest.mark.parametrize('live', (4, 16))
def test_held_and_shift_share_all_semantics_but_have_disjoint_clause_and_question_orders(live):
    held = make_panel('held', TEST_DRAWS[3], 8, live, 128)
    shifted = make_panel('held', TEST_DRAWS[3], 8, live, 128, shift=True)
    for before, after in ((held.blocks, shifted.blocks), (held.queries, shifted.queries)):
        for original, changed in zip(before.reshape(-1, 8), after.reshape(-1, 8), strict=True):
            fact, order, predicate = _read(original)
            other_fact, other_order, other_predicate = _read(changed)
            assert fact == other_fact and predicate == other_predicate
            assert order in SEEN_ORDERS and other_order in SHIFT_ORDERS
            assert Counter(original) == Counter(changed)
            assert not np.array_equal(original, changed)
    np.testing.assert_array_equal(held.answers, shifted.answers)
    np.testing.assert_array_equal(held.support, shifted.support)


@pytest.mark.parametrize('kind,shift', [(kind, False) for kind in SPLITS] + [('held', True)])
def test_noise_prefixes_and_pair_object_identities_survive_gap_and_live_changes(kind, shift):
    panels = [make_panel(kind, TEST_DRAWS[0], 4, live, 9, shift=shift) for live in (4, 8, 16)]
    near = make_panel(kind, TEST_DRAWS[0], 4, 16, 0, shift=shift)
    far = make_panel(kind, TEST_DRAWS[0], 4, 16, 128, shift=shift)
    np.testing.assert_array_equal(near.blocks, far.blocks[:, :, :16])
    np.testing.assert_array_equal(panels[-1].blocks, far.blocks[:, :, :25])
    for panel in (*panels, far):
        np.testing.assert_array_equal(panel.queries, near.queries)
        np.testing.assert_array_equal(panel.answers, near.answers)
    for panel in (panels[-1], far):
        np.testing.assert_array_equal(panel.support, near.support)
    for live, panel in zip((4, 8, 16), panels, strict=True):
        np.testing.assert_array_equal(panel.blocks[:, :, live:], far.blocks[:, :, 16:25])
        for family in range(4):
            for world in range(2):
                facts = _facts(panel.blocks[family, world, :live])
                full = _facts(far.blocks[family, world, :16])
                assert set(facts) <= set(full)
                assert all(full[key][0] == value[0] for key, value in facts.items())


def test_orientations_positions_objects_and_clause_question_orders_are_not_fixed_cues():
    panel = make_panel('train', TRAIN_DRAWS[1], 128, 8, 0)
    orientations, orders = set(), set()
    for family, queries in enumerate(panel.queries):
        giver, recipient, _ = _read(queries[0])[0]
        orientations.add(giver < recipient)
        for query, row in enumerate(queries):
            fact = panel.blocks[family, 0, panel.support[family, 0, query]]
            orders.add((_read(fact)[1], _read(row)[1]))
    assert orientations == {False, True}
    assert set(panel.support[:, 0, 0]) == set(range(8))
    assert set(panel.answers[:, 0, 0]) == set(range(8))
    assert orders == {(clause, query) for clause in SEEN_ORDERS for query in SEEN_ORDERS}


@pytest.mark.parametrize('kind,live', (('seen', 40), ('held', 56)))
def test_all_admissible_live_pairs_can_be_used_without_duplicate_ordered_keys(kind, live):
    panel = make_panel(kind, TEST_DRAWS[-1], 2, live, 1)
    for rows in panel.blocks.reshape(-1, live + 1, 8):
        assert len(_facts(rows[:live])) == live
