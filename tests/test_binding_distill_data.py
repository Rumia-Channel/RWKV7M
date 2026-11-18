"""Behavioural tests for the §22.16 counterfactual late-write data.

The tests parse the emitted token rows instead of trusting the generator: they
recover the public writes, the prefix slots, the tail rewrites, the world
structure and the teacher window from the tokens alone and check them against
the returned labels and eligibility mask.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def distill(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("binding_distill_data")


def writes_of(row, module):
    """Return the ``(key, value)`` records of a row in stream order."""
    records = []
    for position in np.flatnonzero(np.asarray(row) == module.STORE):
        key = int(row[position + 1])
        value = int(row[position + 2])
        if 16 <= key < 32 and 32 <= value < 48:
            records.append((key, value))
    return records


def latest_of(row, module):
    """Latest public value of every key written in a row."""
    latest = {}
    for key, value in writes_of(row, module):
        latest[key] = value
    return latest


def teacher_hits(row, key, module):
    """True when a complete legal write of ``key`` lies in the last 64 tokens."""
    window = np.asarray(row)[-module.TEACHER_TOKENS:]
    for position in np.flatnonzero(window == module.STORE):
        if int(window[position + 1]) == key and 32 <= int(window[position + 2]) < 48:
            return True
    return False


def prefix_records(row, module):
    """The sixteen ``(key, value)`` records of the stride-four prefix."""
    records = writes_of(row, module)
    return records[:16]


def tail_records(row, module):
    """The four rewrite records of the trailing 24-token tail."""
    records = writes_of(row, module)
    return records[-4:]


def gap_records(row, module):
    """The records of the 128-token gap region."""
    records = writes_of(row, module)
    return records[16:-4]


# -- shapes and conventions ------------------------------------------------
def test_shapes_dtypes_and_history_length(distill):
    for step, tokens in ((0, 215), (1, 215), (2, 215), (3, 63), (7, 253)):
        history, queries, labels, eligible = distill.training_batch(11, step, 2)
        assert history.shape == (2, 4, tokens)
        assert queries.shape == (2, 4, 2, 3)
        assert labels.shape == (2, 4, 2)
        assert eligible.shape == (2, 4, 2)
        assert history.dtype == np.int32
        assert queries.dtype == np.int32
        assert labels.dtype == np.int32
        assert eligible.dtype == bool
        assert distill.history_tokens(step) == tokens
        # Only public tokens: no query or answer marker inside a history.
        assert not np.any(history == distill.QUERY)
        assert not np.any(history == distill.ANSWER_MARKER)
        assert np.all((history >= 0) & (history < 128))


def test_query_block_carries_the_key_and_the_row_ends_at_the_answer_marker(distill):
    history, queries, labels, eligible = distill.training_batch(11, 0, 2)
    for family in range(history.shape[0]):
        for world in range(4):
            for question in range(2):
                row = np.concatenate((history[family, world], queries[family, world, question]))
                assert int(row[-3]) == distill.QUERY
                assert 16 <= int(row[-2]) < 32
                assert int(row[-1]) == distill.ANSWER_MARKER
                # The answer is hidden: nothing follows the marker.
                assert int(np.count_nonzero(row == distill.ANSWER_MARKER)) == 1
                assert int(row[-2]) == int(queries[family, world, question, 1])


def test_labels_are_the_exact_latest_public_writes(distill):
    for step in (0, 1, 2, 3, 7):
        history, queries, labels, eligible = distill.training_batch(11, step, 2)
        for family in range(history.shape[0]):
            for world in range(4):
                latest = latest_of(history[family, world], distill)
                for question in range(2):
                    key = int(queries[family, world, question, 1])
                    assert int(labels[family, world, question]) == latest[key]


def test_eligible_marks_only_questions_written_in_the_teacher_window(distill):
    for step in (0, 1, 2):
        history, queries, labels, eligible = distill.training_batch(11, step, 2)
        for family in range(history.shape[0]):
            for world in range(4):
                for question in range(2):
                    key = int(queries[family, world, question, 1])
                    assert bool(eligible[family, world, question]) == teacher_hits(
                        history[family, world], key, distill
                    )


def test_anchor_questions_are_never_teacher_eligible(distill):
    for step in (3, 7, 11, 15):
        history, queries, labels, eligible = distill.training_batch(11, step, 2)
        assert not eligible.any()


# -- the distillation worlds ------------------------------------------------
def test_worlds_form_a_nuisance_by_version_factorial(distill):
    history, queries, labels, eligible = distill.training_batch(11, 0, 3)
    for family in range(history.shape[0]):
        views = history[family]
        # Same version: the frozen teacher window is byte-identical across the
        # nuisance variants.
        assert np.array_equal(views[0, -64:], views[2, -64:])
        assert np.array_equal(views[1, -64:], views[3, -64:])
        # The two versions differ inside that window.
        assert not np.array_equal(views[0, -64:], views[1, -64:])
        # Same nuisance: the earlier history is shared by the two versions.
        assert np.array_equal(views[0, :-64], views[1, :-64])
        assert np.array_equal(views[2, :-64], views[3, :-64])
        # The nuisance variants differ before the window.
        assert not np.array_equal(views[0, :-64], views[2, :-64])


def test_nuisance_difference_is_confined_to_the_first_88_gap_tokens(distill):
    history, queries, labels, eligible = distill.training_batch(23, 4, 2)
    gap = slice(distill.GAP_START, distill.GAP_START + distill.GAP_TOKENS)
    shared = slice(
        distill.GAP_START + distill.NUISANCE_GAP_TOKENS, distill.GAP_START + distill.GAP_TOKENS
    )
    for family in range(history.shape[0]):
        for version in (0, 1):
            # Worlds 2 * nuisance + version: compare the two nuisance
            # variants of one version.
            low = history[family, version]
            high = history[family, version + 2]
            assert np.array_equal(low[shared], high[shared])
            assert not np.array_equal(low[gap][:-distill.NUISANCE_GAP_TOKENS],
                                      high[gap][:-distill.NUISANCE_GAP_TOKENS])


def test_gap_writes_only_touch_the_other_keys_and_always_change_the_value(distill):
    for seed, step in ((23, 0), (23, 1), (29, 2)):
        history, queries, labels, eligible = distill.training_batch(seed, step, 2)
        for family in range(history.shape[0]):
            for world in range(4):
                row = history[family, world]
                prefix = prefix_records(row, distill)
                tail = tail_records(row, distill)
                gap = gap_records(row, distill)
                assert len(gap) == distill.GAP_TOKENS // 4
                gap_keys = {key for key, _ in gap}
                assert len(gap_keys) == 8
                # The gap never touches a scored key.
                assert gap_keys.isdisjoint({key for key, _ in tail})
                assert gap_keys.isdisjoint({int(queries[family, world, q, 1]) for q in range(2)})
                # Every gap write changes the value its key currently holds.
                current = {key: value for key, value in prefix}
                last_write = {}
                for key, value in gap:
                    assert value != current[key]
                    current[key] = value
                    last_write[key] = value
                # The last write of every interfered key lies in the teacher
                # window, so the shared suffix fixes the recent state.
                window = row[-distill.TEACHER_TOKENS:]
                for key, value in last_write.items():
                    assert (key, value) in writes_of(window, distill)
                # No gap value is a label of this family.
                assert {value for _, value in gap}.isdisjoint(
                    {int(labels[family, world, q]) for q in range(2)}
                )


def test_prefix_and_gap_use_the_established_slots(distill):
    history, queries, labels, eligible = distill.training_batch(23, 0, 2)
    for family in range(history.shape[0]):
        row = history[family, 0]
        positions = np.flatnonzero(row == distill.STORE)
        assert len(positions) == 16 + distill.GAP_TOKENS // 4 + distill.TAIL_WRITES
        assert np.all(np.diff(positions[:16]) == 4)
        prefix = prefix_records(row, distill)
        slot_of = {key: index for index, (key, _) in enumerate(prefix)}
        tail_keys = {key for key, _ in tail_records(row, distill)}
        gap_keys = {key for key, _ in gap_records(row, distill)}
        # The scored slots are the eight odd ones; the interfered slots the
        # eight even ones.
        for key in tail_keys | {int(queries[family, 0, q, 1]) for q in range(2)}:
            assert slot_of[key] % 2 == 1
        for key in gap_keys:
            assert slot_of[key] % 2 == 0


def test_tail_rewrites_four_targets_with_a_cyclic_derangement(distill):
    history, queries, labels, eligible = distill.training_batch(23, 0, 2)
    for family in range(history.shape[0]):
        originals = dict(prefix_records(history[family, 0], distill))
        rotations = []
        for version in (0, 1):
            written = tail_records(history[family, version], distill)
            assert len({key for key, _ in written}) == distill.TAIL_WRITES
            values = [value for _, value in written]
            # The tail permutes the four original target values.
            assert set(values) == {originals[key] for key, _ in written}
            # Every rewritten key genuinely changes its value.
            for key, value in written:
                assert value != originals[key]
            # The permutation is a cyclic rotation of the ordered targets.
            base = [originals[key] for key, _ in written]
            shift = next(
                s for s in range(1, distill.TAIL_WRITES)
                if [base[(index + s) % distill.TAIL_WRITES] for index in range(distill.TAIL_WRITES)] == values
            )
            rotations.append(shift)
        assert rotations[0] != rotations[1]


def test_questions_cover_one_rewritten_and_one_untouched_target(distill):
    history, queries, labels, eligible = distill.training_batch(23, 0, 2)
    for family in range(history.shape[0]):
        keys = [int(queries[family, world, question, 1]) for world in range(4) for question in range(2)]
        # The questions are shared by the four views of a family.
        assert len(set(keys)) == 2
        rewritten = {key for key, _ in tail_records(history[family, 0], distill)}
        for world in range(4):
            asked = [int(queries[family, world, question, 1]) for question in range(2)]
            assert len(set(asked)) == 2
            assert len(set(asked) & rewritten) == 1
            # Exactly the rewritten question is teacher eligible.
            assert [bool(eligible[family, world, question]) for question in range(2)] == [
                key in rewritten for key in asked
            ]
            # The untouched answer is the original prefix value and never
            # changes between the versions.
            untouched = next(key for key in asked if key not in rewritten)
            assert int(labels[family, world, asked.index(untouched)]) == int(
                labels[family, (world // 2) * 2 + 1, asked.index(untouched)]
            )


# -- anchor steps -----------------------------------------------------------
def test_anchor_steps_hold_two_distinct_counterfactual_pairs(distill):
    for step in (3, 7, 11, 15):
        history, queries, labels, eligible = distill.training_batch(31, step, 2)
        assert not eligible.any()
        for family in range(history.shape[0]):
            for pair in (0, 1):
                low, high = 2 * pair, 2 * pair + 1
                # Both worlds of a pair are meaningful: their answers differ.
                assert list(labels[family, low]) != list(labels[family, high])
                # The two pairs are distinct counterfactual pairs.
                assert not np.array_equal(history[family, low], history[family, 2 - pair])
                assert [int(queries[family, low, q, 1]) for q in range(2)] != [
                    int(queries[family, 2 - pair, q, 1]) for q in range(2)
                ]


def test_anchor_alternates_between_near_binding_and_legacy_overwrite(distill):
    near, queries, labels, eligible = distill.training_batch(31, 3, 2)
    assert near.shape[2] == distill.NEAR_TOKENS
    assert int((near[0, 0] == distill.STORE).sum()) == 16
    legacy, queries, labels, eligible = distill.training_batch(31, 7, 2)
    assert legacy.shape[2] == distill.LEGACY_TOKENS
    assert int((legacy[0, 0] == distill.STORE).sum()) == 8
    # The legacy anchor asks for the latest of two writes per key.
    for family in range(legacy.shape[0]):
        records = writes_of(legacy[family, 0], distill)
        counts = {}
        for key, _ in records:
            counts[key] = counts.get(key, 0) + 1
        assert set(counts.values()) == {2}
        latest = latest_of(legacy[family, 0], distill)
        for question in range(2):
            key = int(queries[family, 0, question, 1])
            assert int(labels[family, 0, question]) == latest[key]


# -- sampling hygiene -------------------------------------------------------
def test_family_count_extension_leaves_earlier_families_untouched(distill):
    for step in (0, 3, 7):
        one = distill.training_batch(11, step, 1)
        three = distill.training_batch(11, step, 3)
        two = distill.training_batch(11, step, 2)
        for small, large in ((one, three), (two, three)):
            for index in range(small[0].shape[0]):
                assert np.array_equal(small[0][index], large[0][index])
                assert np.array_equal(small[1][index], large[1][index])
                assert np.array_equal(small[2][index], large[2][index])
                assert np.array_equal(small[3][index], large[3][index])


def test_steps_seeds_and_conditions_use_disjoint_data(distill):
    first = distill.training_batch(11, 0, 2)
    second = distill.training_batch(11, 1, 2)
    other_seed = distill.training_batch(12, 0, 2)
    anchor = distill.training_batch(11, 3, 2)
    for other in (second, other_seed, anchor):
        assert not np.array_equal(first[0], other[0])
        assert not np.array_equal(first[2], other[2])
    # The two anchor conditions differ from each other as well.
    legacy = distill.training_batch(11, 7, 2)
    assert not np.array_equal(anchor[0], legacy[0])


def test_repeated_calls_return_the_same_data(distill):
    first = distill.training_batch(41, 2, 2)
    second = distill.training_batch(41, 2, 2)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_invalid_arguments_are_rejected(distill):
    with pytest.raises(ValueError):
        distill.training_batch(-1, 0, 1)
    with pytest.raises(ValueError):
        distill.training_batch(1, -1, 1)
    with pytest.raises(ValueError):
        distill.training_batch(1, 0, 0)
    with pytest.raises(ValueError):
        distill.training_batch(1.5, 0, 1)
