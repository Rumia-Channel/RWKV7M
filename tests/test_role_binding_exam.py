import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from role_binding_data import (ASK, CONSIDER, DEV_DRAWS, GA, HAND, NAME_TOKENS,
                               NI, OBJECT_TOKENS, PERIOD, SEEN_ORDERS,
                               SHIFT_ORDERS, TEST_DRAWS, TOKEN_TO_JAPANESE,
                               TRAIN_CELLS, TRAIN_DRAWS, WHAT, WO, make_panel)
from role_binding_exam import (ADV_DRAWS, FORMS, MID_DRAWS, WORLD_ORDER, ExamPanel,
                               make_exam, public_document, render_exam)


def _decode(row, *, query=False):
    """Independent reader of the public marked phrases, not generator metadata."""
    row = tuple(map(int, row))
    assert len(row) == 8
    phrases = {row[column + 1]: row[column] for column in (0, 2, 4)}
    assert set(phrases) == {GA, NI, WO}
    giver = NAME_TOKENS.index(phrases[GA])
    recipient = NAME_TOKENS.index(phrases[NI])
    assert giver != recipient
    if query:
        assert phrases[WO] == WHAT
        assert row[6:] == (HAND, ASK)
        obj = None
    else:
        obj = OBJECT_TOKENS.index(phrases[WO])
        assert row[6] in (HAND, CONSIDER)
        assert row[7] == PERIOD
    return giver, recipient, obj, row[6]


def _cue(row):
    return _decode(row, query=True)[:2]


def _held(direction):
    a, b = sorted(direction)
    return b - a in (2, 6)


def _order(row):
    return tuple({GA: 0, NI: 1, WO: 2}[int(row[column])] for column in (1, 3, 5))


def _effective(blocks):
    result = {}
    for position, row in enumerate(blocks):
        giver, recipient, obj, predicate = _decode(row)
        if predicate == HAND:
            result[giver, recipient] = (obj, position)
    return result


def _oracle(document, *, rule='latest-binding'):
    """Solve only printed raw records and labeled public rows, retaining traces."""
    records = [_decode(row) for row in document['blocks']]
    cards = {int(row['on']): row['ask'] for row in document['cards']}
    assert len(cards) == 8
    assert set(cards) == set(OBJECT_TOKENS)
    traces = []
    sources = []
    for initial in document['queries']:
        current = initial
        values, support = [], []
        for hop in range(document['reads']):
            direction = _cue(current)
            matching = [(position, row[2]) for position, row in enumerate(records)
                        if row[:2] == direction and
                        (rule == 'latest-any' or row[3] == HAND)]
            assert matching, 'every displayed question must have a real binding record'
            position, obj = matching[0] if rule == 'first-binding' else matching[-1]
            values.append(obj)
            support.append(position)
            if hop + 1 < document['reads']:
                current = cards[OBJECT_TOKENS[obj]]
        traces.append(values)
        sources.append(support)
    values = np.asarray(traces, dtype=np.int32)
    return values[:, -1], np.asarray(sources, dtype=np.int32), values


def _jp(tokens):
    return ''.join(TOKEN_TO_JAPANESE[int(token)] for token in tokens)


def test_registered_forms_have_separate_fixed_draws_and_true_quartet_order():
    assert FORMS == ('advanced', 'midterm')
    assert ADV_DRAWS == (764101, 764103, 764107, 764109)
    assert MID_DRAWS == (766101, 766103, 766107, 766109, 766113)
    assert WORLD_ORDER == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert not set(ADV_DRAWS) & set(MID_DRAWS)
    assert not set(ADV_DRAWS + MID_DRAWS) & set(TRAIN_DRAWS + DEV_DRAWS + TEST_DRAWS)


@pytest.mark.parametrize('form,draw,reads,live,core', (
    ('advanced', ADV_DRAWS[0], 3, 12, 16),
    ('midterm', MID_DRAWS[0], 2, 8, 12),
))
def test_shapes_live_bidirectional_facts_and_public_vocabulary(form, draw, reads, live, core):
    panel = make_exam(form, draw, 5, 17)
    assert isinstance(panel, ExamPanel)
    assert panel.reads == reads
    assert panel.blocks.shape == (5, 4, core + 17, 8)
    assert panel.queries.shape == (5, 2, 8)
    assert panel.cards.shape == (5, 8, 8)
    assert panel.answers.shape == (5, 4, 2)
    assert panel.support.shape == panel.values.shape == (5, 4, 2, reads)
    assert panel.card_order.shape == (5, 8)
    for name in ('blocks', 'queries', 'cards', 'answers', 'support', 'values', 'card_order'):
        assert getattr(panel, name).dtype == np.int32
    for tokens in (panel.blocks, panel.queries, panel.cards):
        assert set(tokens.reshape(-1)) <= set(TOKEN_TO_JAPANESE)
    assert np.all((panel.values >= 0) & (panel.values < 8))
    assert np.all((panel.support >= 0) & (panel.support < core - 2))
    np.testing.assert_array_equal(panel.answers, panel.values[..., -1])

    for family in range(5):
        directions = set(_effective(panel.blocks[family, 0]))
        assert len(directions) == live
        assert len({tuple(sorted(direction)) for direction in directions}) == live // 2
        assert all(direction[::-1] in directions for direction in directions)
        assert sum(_decode(row)[3] == HAND for row in panel.blocks[family, 0]) == live + 2
        roots = [_cue(row) for row in panel.queries[family]]
        assert roots[0] == roots[1][::-1]
        card_directions = [_cue(row) for row in panel.cards[family]]
        assert len(set(card_directions)) == 8
        assert set(roots + card_directions) <= directions
        assert sorted(panel.card_order[family]) == list(range(8))
        for row in np.concatenate((panel.queries[family], panel.cards[family])):
            assert WHAT in row
            assert not set(row) & set(OBJECT_TOKENS)


@pytest.mark.parametrize('form,draw',
                         [('advanced', draw) for draw in ADV_DRAWS] +
                         [('midterm', draw) for draw in MID_DRAWS])
def test_independent_public_oracle_recovers_every_hop_and_final_answer(form, draw):
    panel = make_exam(form, draw, 7, 29)
    for family in range(7):
        for world in range(4):
            document = public_document(panel, family, world)
            answer, support, values = _oracle(document)
            np.testing.assert_array_equal(answer, panel.answers[family, world])
            np.testing.assert_array_equal(support, panel.support[family, world])
            np.testing.assert_array_equal(values, panel.values[family, world])
            effective = _effective(document['blocks'])
            for query in range(2):
                assert len(set(support[query])) == panel.reads
                for hop, position in enumerate(support[query]):
                    direction = _decode(document['blocks'][position])[:2]
                    assert effective[direction] == (values[query, hop], position)


@pytest.mark.parametrize('form,draw', (('advanced', ADV_DRAWS[1]), ('midterm', MID_DRAWS[1])))
def test_quartet_changes_only_role_bindings_at_two_regions_and_implements_xor(form, draw):
    panel = make_exam(form, draw, 11, 19)
    for family in range(11):
        root_positions = set(panel.support[family, :, :, 0].reshape(-1))
        terminal_positions = set(panel.support[family, :, :, -1].reshape(-1))
        assert len(root_positions) == len(terminal_positions) == 2
        assert not root_positions & terminal_positions
        before = panel.blocks[family, 0]
        base_public = public_document(panel, family, 0)
        for world, (rootflip, terminalflip) in enumerate(WORLD_ORDER):
            after = panel.blocks[family, world]
            expected = (root_positions if rootflip else set()) | (
                terminal_positions if terminalflip else set())
            changed = np.flatnonzero(np.any(before != after, axis=-1))
            assert set(changed) == expected
            np.testing.assert_array_equal(np.sort(before, axis=-1), np.sort(after, axis=-1))
            np.testing.assert_array_equal(before[:2], after[:2])
            for position in changed:
                original, altered = _decode(before[position]), _decode(after[position])
                assert original[:2] == altered[:2][::-1]
                assert original[2:] == altered[2:]
                assert np.count_nonzero(before[position] != after[position]) == 2
            public = public_document(panel, family, world)
            for key in ('queries', 'cards', 'reads'):
                assert public[key] == base_public[key]
        np.testing.assert_array_equal(panel.answers[family, 0], panel.answers[family, 3])
        np.testing.assert_array_equal(panel.answers[family, 1], panel.answers[family, 2])
        assert np.all(panel.answers[family, 0] != panel.answers[family, 1])
        assert np.all(panel.answers[family, 0] != panel.answers[family, 2])
        assert np.all(panel.answers[family, :, 0] != panel.answers[family, :, 1])
        # A root-only observer cannot distinguish 00/01; a terminal-only
        # observer cannot distinguish 00/10. Both pairs require different gold.
        np.testing.assert_array_equal(panel.blocks[family, 0, sorted(root_positions)],
                                      panel.blocks[family, 1, sorted(root_positions)])
        np.testing.assert_array_equal(panel.blocks[family, 0, sorted(terminal_positions)],
                                      panel.blocks[family, 2, sorted(terminal_positions)])
        np.testing.assert_array_equal(panel.values[family, 0, :, :-1],
                                      panel.values[family, 1, :, :-1])
        for world in range(4):
            layers = [set(panel.values[family, world, :, hop]) for hop in range(panel.reads)]
            assert all(len(layer) == 2 for layer in layers)
            assert all(not layer & other for hop, layer in enumerate(layers)
                       for other in layers[hop + 1:])


@pytest.mark.parametrize('form,draw',
                         [('advanced', draw) for draw in ADV_DRAWS] +
                         [('midterm', draw) for draw in MID_DRAWS])
def test_first_latest_and_binding_are_genuinely_distinct(form, draw):
    panel = make_exam(form, draw, 5, 0)
    core = panel.blocks.shape[2]
    for family in range(5):
        final_possibilities = set(panel.answers[family].reshape(-1))
        for world in range(4):
            document = public_document(panel, family, world)
            first, _, _ = _oracle(document, rule='first-binding')
            last_any, _, _ = _oracle(document, rule='latest-any')
            correct = panel.answers[family, world]
            assert np.all(first != correct)
            assert np.all(last_any != correct)
            assert np.all(first != last_any)
            for position in set(panel.support[family, world, :, -1]):
                direction = _decode(document['blocks'][position])[:2]
                matches = [(i, _decode(row)) for i, row in enumerate(document['blocks'])
                           if _decode(row)[:2] == direction]
                binding = [(i, row) for i, row in matches if row[3] == HAND]
                plans = [(i, row) for i, row in matches if row[3] == CONSIDER]
                assert len(binding) == 2
                assert binding[0][0] in (0, 1)
                assert binding[0][0] < binding[1][0] == position
                assert binding[0][1][2] not in final_possibilities
                assert len(plans) == 1
                assert plans[0][0] in (core - 2, core - 1)
                assert plans[0][0] > position
                assert plans[0][1][2] not in final_possibilities
                assert plans[0][1][2] != binding[0][1][2]


def test_no_training_occurrence_including_cards_or_background_uses_any_held_dyad():
    held_pairs = {pair for a in range(8) for b in range(a + 1, 8)
                  if _held(pair := (a, b))}
    assert len(held_pairs) == 8
    for draw in ADV_DRAWS:
        panel = make_exam('advanced', draw, 17, 128)
        for row in panel.blocks.reshape(-1, 8):
            assert tuple(sorted(_decode(row)[:2])) not in held_pairs
        for row in np.concatenate((panel.queries.reshape(-1, 8), panel.cards.reshape(-1, 8))):
            assert tuple(sorted(_cue(row))) not in held_pairs
    # Elementary training and its separate basic development source obey the
    # same exclusion; advanced augmentation must not quietly undo the split.
    for kind, draws in (('train', TRAIN_DRAWS), ('dev', DEV_DRAWS)):
        for draw in draws:
            for live, gap in TRAIN_CELLS:
                panel = make_panel(kind, draw, 3, live, gap)
                assert all(not _held(_decode(row)[:2]) for row in panel.blocks.reshape(-1, 8))
                assert all(not _held(_cue(row)) for row in panel.queries.reshape(-1, 8))


@pytest.mark.parametrize('draw', MID_DRAWS)
def test_midterm_holds_both_root_and_terminal_dyads_and_has_shared_name_foils(draw):
    panel = make_exam('midterm', draw, 13, 9)
    for family in range(13):
        root = tuple(sorted(_cue(panel.queries[family, 0])))
        terminal = tuple(sorted(_decode(panel.blocks[family, 0, panel.support[family, 0, 0, -1]])[:2]))
        assert root != terminal
        assert _held(root) and _held(terminal)
        pairs = {tuple(sorted(direction)) for direction in _effective(panel.blocks[family, 0])}
        foils = pairs - {root, terminal}
        assert len(foils) == 2
        assert any(set(pair) & set(root) for pair in foils)
        assert any(set(pair) & set(terminal) for pair in foils)


@pytest.mark.parametrize('draw', MID_DRAWS)
def test_familiar_is_exactly_a_seen_order_rerender_of_the_new_classroom_semantics(draw):
    shifted = make_exam('midterm', draw, 7, 31)
    familiar = make_exam('midterm', draw, 7, 31, familiar=True)
    for name in ('answers', 'support', 'values', 'card_order'):
        np.testing.assert_array_equal(getattr(shifted, name), getattr(familiar, name))
    assert shifted.reads == familiar.reads == 2
    for name in ('blocks', 'queries', 'cards'):
        new_rows = getattr(shifted, name).reshape(-1, 8)
        familiar_rows = getattr(familiar, name).reshape(-1, 8)
        for new, seen in zip(new_rows, familiar_rows, strict=True):
            assert _order(new) in SHIFT_ORDERS
            assert _order(seen) in SEEN_ORDERS
            assert _order(new) != _order(seen)
            assert _decode(new, query=name != 'blocks') == _decode(seen, query=name != 'blocks')
    for family in range(7):
        for world in range(4):
            for panel in (shifted, familiar):
                answer, support, values = _oracle(public_document(panel, family, world))
                np.testing.assert_array_equal(answer, panel.answers[family, world])
                np.testing.assert_array_equal(support, panel.support[family, world])
                np.testing.assert_array_equal(values, panel.values[family, world])


def test_advanced_statements_and_all_questions_use_seen_orders():
    panel = make_exam('advanced', ADV_DRAWS[-1], 9, 33)
    for name in ('blocks', 'queries', 'cards'):
        assert all(_order(row) in SEEN_ORDERS for row in getattr(panel, name).reshape(-1, 8))


@pytest.mark.parametrize('form,draw,familiar', (
    ('advanced', ADV_DRAWS[2], False),
    ('midterm', MID_DRAWS[2], False),
    ('midterm', MID_DRAWS[2], True),
))
def test_reproducible_family_and_gap_prefixes_do_not_change_the_problem(form, draw, familiar):
    short = make_exam(form, draw, 3, 7, familiar=familiar)
    repeat = make_exam(form, draw, 3, 7, familiar=familiar)
    long = make_exam(form, draw, 8, 93, familiar=familiar)
    for name in ('blocks', 'queries', 'cards', 'answers', 'support', 'values', 'card_order'):
        np.testing.assert_array_equal(getattr(short, name), getattr(repeat, name))
        extended = getattr(long, name)[:3]
        if name == 'blocks':
            extended = extended[:, :, :short.blocks.shape[2]]
        np.testing.assert_array_equal(getattr(short, name), extended)
    assert short.reads == repeat.reads == long.reads


@pytest.mark.parametrize('form,draw', (('advanced', ADV_DRAWS[3]), ('midterm', MID_DRAWS[3])))
def test_public_boundary_has_only_raw_inputs_and_actual_object_token_labels(form, draw):
    panel = make_exam(form, draw, 8, 3)
    for family in range(8):
        document = public_document(panel, family, 2)
        assert set(document) == {'blocks', 'queries', 'cards', 'reads'}
        assert np.shape(document['blocks']) == panel.blocks.shape[2:]
        assert np.shape(document['queries']) == (2, 8)
        assert type(document['reads']) is int
        assert len(document['cards']) == 8
        assert [row['on'] for row in document['cards']] == [
            OBJECT_TOKENS[int(obj)] for obj in panel.card_order[family]]
        assert all(set(row) == {'on', 'ask'} for row in document['cards'])
        for row in document['cards']:
            assert type(row['on']) is int
            assert all(type(token) is int for token in row['ask'])
            obj = OBJECT_TOKENS.index(row['on'])
            assert row['ask'] == panel.cards[family, obj].tolist()
        # Changing every annotation cannot change either the machine-facing
        # public document or the human-facing story.
        scrambled = panel._replace(answers=np.full_like(panel.answers, -700),
                                   support=np.full_like(panel.support, -800),
                                   values=np.full_like(panel.values, -900))
        assert public_document(scrambled, family, 2) == document
        assert render_exam(scrambled, family, 2, form) == render_exam(panel, family, 2, form)
    orders = {tuple(order) for order in panel.card_order}
    assert len(orders) > 1
    assert any(order != tuple(range(8)) for order in orders)


@pytest.mark.parametrize('form,draw,reads', (
    ('advanced', ADV_DRAWS[0], 3), ('midterm', MID_DRAWS[0], 2),
))
def test_printed_story_contains_complete_records_two_starts_and_eight_labeled_questions(form, draw, reads):
    panel = make_exam(form, draw, 4, 2)
    for family in range(4):
        document = public_document(panel, family, 1)
        text = render_exam(panel, family, 1, form)
        assert f'ちょうど{reads}回照会' in text
        for row in document['blocks'] + document['queries']:
            assert _jp(row) in text
        printed_cards = [line for line in text.splitlines() if ' → ' in line]
        assert printed_cards == [f'{_jp((row["on"],))} → {_jp(row["ask"])}'
                                 for row in document['cards']]
        assert text.count('解答欄（　　　　　　　）') == 2


@pytest.mark.parametrize('overrides', (
    {'form': 'held'}, {'form': 'train'}, {'draw': -1}, {'families': 0},
    {'gap': -1}, {'draw': 1.5}, {'families': 2.5}, {'gap': 1.5},
    {'form': 'advanced', 'familiar': True},
))
def test_invalid_requests_do_not_silently_change_the_experiment(overrides):
    arguments = dict(form='midterm', draw=MID_DRAWS[0], families=2, gap=0)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        make_exam(**arguments)


def test_rendering_rejects_a_mismatched_story_form():
    panel = make_exam('midterm', MID_DRAWS[0], 1, 0)
    with pytest.raises(ValueError):
        render_exam(panel, 0, 0, 'advanced')
