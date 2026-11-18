"""Behavioural tests for the retrieval-frontier dataset and controller.

The dataset tests parse the actual token rows instead of trusting the
generator: they recover the store records, the refresh counts before and after
the prefix, the literal distance from the stored value to the query, and the
counterfactual worlds, and then check them against the labels.
"""

import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def frontier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("retrieval_frontier")


def stores_of(row, module):
    """Return the ``(key, value)`` records of a row in stream order."""
    positions = np.flatnonzero(np.asarray(row) == module.STORE)
    return [(int(row[p + 1]), int(row[p + 2])) for p in positions]


def store_positions(row, module):
    return np.flatnonzero(np.asarray(row) == module.STORE)


def split_positions(row, module, gap):
    """Split the store positions into the region before, inside and after the
    shared 16-key prefix."""
    pre_len = module.ROW_TOKENS - module.PREFIX_TOKENS - gap - module.QUERY_TOKENS
    positions = store_positions(row, module)
    before = positions[positions < pre_len]
    prefix = positions[(positions >= pre_len) & (positions < pre_len + module.PREFIX_TOKENS)]
    after = positions[positions >= pre_len + module.PREFIX_TOKENS]
    return before, prefix, after


def distance_to_query(value_position, module):
    """Tokens strictly between a stored value and the query token."""
    return (module.ROW_TOKENS - module.QUERY_TOKENS) - 1 - int(value_position)


# -- dataset --------------------------------------------------------------
def test_batch_shapes_dtypes_and_budget(frontier):
    ids, labels = frontier.make_examples(42, 0, 3, gap=64, writes=8)
    assert ids.shape == (3, 2, 2, frontier.ROW_TOKENS)
    assert labels.shape == (3, 2, 2)
    assert ids.dtype == np.int32 and labels.dtype == np.int32
    assert frontier.ROW_TOKENS == 256
    # The answer is hidden: the answer marker ends every row exactly once and
    # nothing follows it.
    for row in ids.reshape(-1, frontier.ROW_TOKENS):
        assert int(row[-3]) == frontier.QUERY
        assert int(row[-1]) == frontier.ANSWER_MARKER
        assert int(np.count_nonzero(row == frontier.ANSWER_MARKER)) == 1


def test_query_block_carries_the_queried_key_and_the_label_is_its_value(frontier):
    ids, labels = frontier.make_examples(7, 3, 2, gap=32, writes=4)
    for family in range(ids.shape[0]):
        for world in range(2):
            for question in range(2):
                row = ids[family, world, question]
                key = int(row[-2])
                records = stores_of(row, frontier)
                matching = [value for stored_key, value in records if stored_key == key]
                # A queried key is written exactly once, so its stored value is
                # the answer.
                assert len(matching) == 1
                assert matching[0] == int(labels[family, world, question])


def test_prefix_uses_a_stride_of_four_and_gap_geometry_is_literal(frontier):
    for gap in frontier.GAPS:
        for writes in frontier.WRITES:
            ids, labels = frontier.make_examples(11, 1, 1, gap=gap, writes=writes)
            row = ids[0, 0, 0]
            before, prefix_positions, after = split_positions(row, frontier, gap)

            # 16 prefix stores on a stride of four, 8 refresh stores in total.
            assert len(prefix_positions) == 16
            assert np.all(np.diff(prefix_positions) == 4)
            assert len(before) + len(after) == frontier.REFRESH_TOTAL
            # The gap region holds exactly ``writes`` refresh stores.
            assert len(after) == writes
            assert len(before) == frontier.REFRESH_TOTAL - writes
            # The stated gap is the literal distance from the last prefix value
            # to the query token.
            last_prefix_value = int(prefix_positions[-1]) + 2
            assert distance_to_query(last_prefix_value, frontier) == gap


def test_refresh_stores_never_touch_a_queried_key_or_a_label(frontier):
    for gap in frontier.GAPS:
        for writes in frontier.WRITES:
            ids, labels = frontier.make_examples(13, 2, 2, gap=gap, writes=writes)
            for family in range(ids.shape[0]):
                for world in range(2):
                    for question in range(2):
                        row = ids[family, world, question]
                        before, prefix_positions, after = split_positions(row, frontier, gap)
                        refresh = [
                            (int(row[p + 1]), int(row[p + 2]))
                            for p in np.concatenate([before, after])
                        ]
                        prefix = [
                            (int(row[p + 1]), int(row[p + 2])) for p in prefix_positions
                        ]
                        queried_keys = {int(row[-2])}
                        assert queried_keys.isdisjoint({key for key, _ in refresh})
                        assert {int(labels[family, world, question])}.isdisjoint(
                            {value for _, value in refresh}
                        )
                        # A refresh repeats the pair the prefix already stores.
                        assert set(refresh).issubset(set(prefix))


def test_queried_values_are_at_most_56_tokens_older_than_the_stated_gap(frontier):
    extras = []
    for gap in frontier.GAPS:
        ids, _ = frontier.make_examples(17, 5, 8, gap=gap, writes=4)
        for row in ids.reshape(-1, frontier.ROW_TOKENS):
            key = int(row[-2])
            positions = store_positions(row, frontier)
            value_position = next(
                int(p) + 2 for p in positions if int(row[p + 1]) == key
            )
            extras.append(distance_to_query(value_position, frontier) - gap)
    assert extras
    assert min(extras) >= 0
    assert max(extras) <= 56


def test_facts_do_not_depend_on_the_cell(frontier):
    """The same (seed, index) yields the same facts in every cell."""
    reference = None
    for gap in frontier.GAPS:
        for writes in frontier.WRITES:
            ids, labels = frontier.make_examples(23, 4, 2, gap=gap, writes=writes)
            facts = []
            for family in range(ids.shape[0]):
                for world in range(2):
                    row = ids[family, world, 0]
                    _, prefix_positions, _ = split_positions(row, frontier, gap)
                    prefix = [
                        (int(row[p + 1]), int(row[p + 2])) for p in prefix_positions
                    ]
                    facts.append((tuple(prefix), int(row[-2])))
            if reference is None:
                reference = facts
            else:
                assert facts == reference


def test_worlds_differ_in_the_answer_but_share_every_token(frontier):
    ids, labels = frontier.make_examples(29, 6, 4, gap=64, writes=4)
    for family in range(ids.shape[0]):
        for question in range(2):
            assert int(labels[family, 0, question]) != int(labels[family, 1, question])
            world0 = np.sort(ids[family, 0, question])
            world1 = np.sort(ids[family, 1, question])
            assert np.array_equal(world0, world1)
        # The two questions of a family ask about different keys.
        assert int(ids[family, 0, 0, -2]) != int(ids[family, 0, 1, -2])


def test_anchor_rows_are_genuine_overwrites_with_the_stale_value_present(frontier):
    ids, labels = frontier.make_examples(31, 7, 2, gap=32, writes=0, anchor=True)
    for family in range(ids.shape[0]):
        for world in range(2):
            for question in range(2):
                row = ids[family, world, question]
                positions = store_positions(row, frontier)
                assert len(positions) == 8
                key = int(row[-2])
                values = [int(row[p + 2]) for p in positions if int(row[p + 1]) == key]
                assert len(values) == 2
                stale, latest = values
                assert stale != latest
                # The answer is the *latest* write of the key.
                assert latest == int(labels[family, world, question])


def test_anchor_latest_values_sit_exactly_gap_to_gap_plus_12_tokens_before_the_query(frontier):
    ids, _ = frontier.make_examples(33, 8, 2, gap=32, writes=0, anchor=True)
    for row in ids.reshape(-1, frontier.ROW_TOKENS):
        positions = store_positions(row, frontier)
        distances = sorted(distance_to_query(int(p) + 2, frontier) for p in positions)
        # Four stale writes and four latest writes.  The latest writes are the
        # ones closest to the query, and the last prefix value sits exactly
        # ``gap`` tokens before it.
        assert distances[:4] == [32, 36, 40, 44]
        assert distances[-4:] == [48, 52, 56, 60]
        assert distances[0] == 32 == frontier.GAPS[0]


def test_anchor_rejects_other_cells_and_regular_rows_keep_a_unique_stored_value(frontier):
    with pytest.raises(ValueError):
        frontier.make_examples(1, 1, 1, gap=64, writes=0, anchor=True)
    with pytest.raises(ValueError):
        frontier.make_examples(1, 1, 1, gap=32, writes=4, anchor=True)
    ids, labels = frontier.make_examples(37, 9, 2, gap=32, writes=4)
    for family in range(ids.shape[0]):
        for world in range(2):
            for question in range(2):
                key = int(ids[family, world, question, -2])
                matching = [
                    value
                    for stored_key, value in stores_of(ids[family, world, question], frontier)
                    if stored_key == key
                ]
                assert len(matching) == 1


def test_sampling_is_deterministic_and_index_selects_the_facts(frontier):
    first, labels_a = frontier.make_examples(41, 10, 2, gap=32, writes=0)
    second, labels_b = frontier.make_examples(41, 10, 2, gap=32, writes=0)
    assert np.array_equal(first, second)
    assert np.array_equal(labels_a, labels_b)
    other, _ = frontier.make_examples(41, 11, 2, gap=32, writes=0)
    assert not np.array_equal(first, other)
    # A different seed changes the facts as well.
    other_seed, _ = frontier.make_examples(42, 10, 2, gap=32, writes=0)
    assert not np.array_equal(first, other_seed)


def test_next_token_batch_keeps_the_answer_out_of_the_input(frontier):
    ids, labels = frontier.make_examples(43, 12, 2, gap=128, writes=8)
    batch = frontier.to_next_token_batch(ids, labels)
    assert batch["input_ids"].shape == ids.shape
    assert np.array_equal(batch["input_ids"], ids)
    assert np.array_equal(batch["target_ids"][..., -1], labels)
    assert np.array_equal(batch["mask"], (ids == frontier.ANSWER_MARKER).astype(np.float32))
    # Exactly one supervised position per row, at the end.
    assert np.array_equal(batch["mask"].sum(axis=-1), np.ones(ids.shape[:-1]))


# -- schedule -------------------------------------------------------------
def test_fixed_schedule_visits_every_cell_eight_times_in_96_steps(frontier):
    visits = {}
    anchors = 0
    for step in range(96):
        cell = frontier.fixed_cell(step)
        if cell is None:
            anchors += 1
            assert step % 4 == 3
        else:
            visits[cell] = visits.get(cell, 0) + 1
    assert anchors == 24
    assert sum(visits.values()) == 72
    assert set(visits) == set(frontier.CELLS)
    assert set(visits.values()) == {8}


def test_fixed_and_adaptive_arms_share_the_anchor_positions(frontier):
    controller = frontier.Frontier()
    for step in range(96):
        fixed = frontier.fixed_cell(step)
        adaptive = controller.cell(step)
        assert (fixed is None) == (adaptive is None)
        assert (fixed is None) == frontier.is_anchor_step(step)


def test_cell_update_index_counts_only_cell_updates(frontier):
    indices = [
        frontier.cell_update_index(step)
        for step in range(96)
        if not frontier.is_anchor_step(step)
    ]
    assert indices == list(range(72))


# -- controller -----------------------------------------------------------
def _perfect(window=32):
    return (window, window)


def test_promotion_needs_two_successive_qualifying_calibrations(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    assert (32, 0) not in controller.mastered
    assert controller.streak[(32, 0)] == 1
    controller.observe(counts, 1.0)
    assert (32, 0) in controller.mastered
    assert controller.promotions == 3


def test_a_failing_calibration_breaks_the_streak(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    degraded = dict(counts)
    degraded[(32, 0)] = (16, 32)
    controller.observe(degraded, 1.0)
    assert controller.streak[(32, 0)] == 0
    assert (32, 0) not in controller.mastered
    controller.observe(counts, 1.0)
    assert (32, 0) not in controller.mastered


def test_missing_counts_break_the_streak_of_a_separated_measurement(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    partial = {cell: value for cell, value in counts.items() if cell != (32, 0)}
    controller.observe(partial, 1.0)
    assert controller.streak[(32, 0)] == 0
    controller.observe(counts, 1.0)
    assert (32, 0) not in controller.mastered


def test_a_frozen_calibration_blocks_promotion_and_clears_streaks(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    assert controller.streak[(32, 0)] == 1
    # One percentage point of anchor regression freezes the policy.
    controller.observe(counts, 0.99)
    assert controller.frozen is True
    assert controller.streak[(32, 0)] == 0
    assert (32, 0) not in controller.mastered
    # Recovery needs two fresh qualifying calibrations again.
    controller.observe(counts, 1.0)
    assert (32, 0) not in controller.mastered
    controller.observe(counts, 1.0)
    assert (32, 0) in controller.mastered


def test_a_farther_gap_is_not_promoted_out_of_order(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    controller.observe(counts, 1.0)
    assert (32, 0) in controller.mastered
    assert (64, 0) not in controller.mastered
    assert controller.active_gap(0) == 64
    controller.observe(counts, 1.0)
    controller.observe(counts, 1.0)
    assert (64, 0) in controller.mastered
    assert controller.active_gap(0) == 128


def test_a_promoted_cell_stays_promoted(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    controller.observe(counts, 1.0)
    assert (32, 0) in controller.mastered
    counts[(32, 0)] = (8, 32)
    controller.observe(counts, 1.0)
    assert (32, 0) in controller.mastered
    assert controller.active_gap(0) == 64


def test_the_weakest_candidate_is_chosen(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    counts[(32, 4)] = (10, 32)
    controller.observe(counts, 1.0)
    assert controller.cell(0) == (32, 4)


def test_all_three_writes_regimes_receive_positive_coverage(frontier):
    controller = frontier.Frontier()
    seen = set()
    for step in range(0, 96, 4):
        cell = controller.cell(step)
        if cell is not None:
            seen.add(cell[1])
    assert seen == {0, 4, 8}


def test_state_dict_roundtrip_continues_identically(frontier):
    controller = frontier.Frontier()
    counts = {cell: _perfect() for cell in frontier.CELLS}
    controller.observe(counts, 1.0)
    for step in range(0, 40, 4):
        controller.cell(step)
    payload = controller.state_dict()
    # The state is checkpointable as plain JSON.
    assert json.loads(json.dumps(payload)) == payload

    restored = frontier.Frontier.from_state_dict(payload)
    assert restored.summary() == controller.summary()
    for step in range(40, 96, 4):
        assert restored.cell(step) == controller.cell(step)


def test_from_state_dict_rejects_a_corrupt_state(frontier):
    controller = frontier.Frontier()
    payload = controller.state_dict()
    with pytest.raises(ValueError):
        frontier.Frontier.from_state_dict(dict(payload, version=99))
    with pytest.raises(ValueError):
        frontier.Frontier.from_state_dict(dict(payload, mastered=[[7, 7]]))
    with pytest.raises(ValueError):
        frontier.Frontier.from_state_dict(dict(payload, streak={"32,0": -1}))


def test_feedback_never_asks_for_labels(frontier):
    parameters = list(inspect.signature(frontier.Frontier.observe).parameters)
    assert parameters == ["self", "counts", "anchor_accuracy"]


def test_replay_cell_consumes_a_recorded_schedule(frontier):
    controller = frontier.Frontier()
    for step in range(0, 20, 4):
        controller.cell(step)
    schedule = dict(controller.schedule)
    assert schedule
    for step, cell in schedule.items():
        assert frontier.replay_cell(schedule, step) == cell
    assert frontier.replay_cell(schedule, 3) is None
    # A step the schedule does not cover is rejected, not silently padded.
    with pytest.raises(ValueError):
        frontier.replay_cell(schedule, 20)


def test_yoked_arm_replays_the_next_seed_cyclically(frontier):
    schedules = {
        42: {0: (32, 0)},
        43: {0: (64, 4)},
        44: {0: (128, 8)},
    }
    assert frontier.yoked_schedule(schedules, 42, (42, 43, 44)) == schedules[43]
    assert frontier.yoked_schedule(schedules, 43, (42, 43, 44)) == schedules[44]
    assert frontier.yoked_schedule(schedules, 44, (42, 43, 44)) == schedules[42]
    # A seed outside the study, or a donor without a schedule, is rejected.
    with pytest.raises(ValueError):
        frontier.yoked_schedule(schedules, 45, (42, 43, 44))
    with pytest.raises(ValueError):
        frontier.yoked_schedule(schedules, 44, (42, 43, 44, 45))
