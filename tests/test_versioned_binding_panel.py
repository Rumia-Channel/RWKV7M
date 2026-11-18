import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from versioned_binding_panel import (  # noqa: E402
    CHAINS, COPY, FILLER_TOKEN, PREFIX_TOKENS, QUERY_TOKEN, STAGE_TOKENS, STORE_TOKEN,
    SUFFIX_TOKENS, literal_reference, versioned_cases,
)


def naive(history, keys):
    """A slow, obvious reading of the public grammar for one row."""
    bindings = {}
    for i in range(len(history) - 2):
        marker, target, argument = int(history[i]), int(history[i + 1]), int(history[i + 2])
        if marker == STORE_TOKEN and 16 <= target < 32 and 32 <= argument < 48:
            bindings[target] = argument
        elif marker == COPY and 16 <= target < 32 and 16 <= argument < 32:
            bindings[target] = bindings.get(argument, -2)
    return [bindings.get(int(key), -2) for key in keys]


def original_values(history):
    """The value the prefix stored under each key of one row."""
    prefix = history[:PREFIX_TOKENS]
    keys = prefix[1::4]
    values = prefix[2::4]
    return dict(zip((int(k) for k in keys), (int(v) for v in values), strict=True))


def gap_start(depth):
    return PREFIX_TOKENS + (depth // 2) * CHAINS * STAGE_TOKENS


@pytest.fixture(scope='module')
def panel():
    cases = list(versioned_cases(families=4, depths=(2, 24), gaps=(0, 1024)))
    return cases


def test_labels_follow_the_lineage_and_match_the_literal_reference(panel):
    for depth, gap, history, queries, truth, active in panel:
        gold = literal_reference(history.reshape(-1, history.shape[-1]), queries[..., 1].reshape(-1, queries.shape[2]))
        assert np.array_equal(gold.reshape(truth.shape), truth)
        for f in range(history.shape[0]):
            for w in range(2):
                assert naive(history[f, w], queries[f, w, :, 1]) == truth[f, w].tolist()
                assert active[f].sum() == 4
                stored = original_values(history[f, w])
                protected = {stored[int(key)] for key in queries[f, w, :, 1]}
                for i, key in enumerate(queries[f, w, :, 1]):
                    assert 32 <= truth[f, w, i] < 48
                    if active[f, i]:
                        # A live wire carries the original value of another key.
                        assert truth[f, w, i] in protected
                        assert truth[f, w, i] != stored[int(key)]
                    else:
                        # A poisoned wire holds a literal that never was a target value.
                        assert truth[f, w, i] not in protected


def test_live_values_differ_between_worlds_and_poison_does_not(panel):
    for depth, gap, history, queries, truth, active in panel:
        for f in range(history.shape[0]):
            assert not np.array_equal(truth[f, 0][active[f]], truth[f, 1][active[f]])
            assert np.array_equal(truth[f, 0][~active[f]], truth[f, 1][~active[f]])
            # Only the original target assignments differ between the worlds.
            differing = np.flatnonzero(history[f, 0] != history[f, 1])
            assert set(differing) <= set(int(p) for p in np.arange(6, 63, 8))


def test_original_key_lookup_cannot_solve_the_panel(panel):
    for depth, gap, history, queries, truth, active in panel:
        for f in range(history.shape[0]):
            for w in range(2):
                stored = original_values(history[f, w])
                lookup = [stored[int(key)] for key in queries[f, w, :, 1]]
                assert lookup != truth[f, w].tolist()
                assert all(lookup[i] != truth[f, w, i] for i in range(len(lookup)))


def test_dropping_the_copies_destroys_the_endpoints(panel):
    for depth, gap, history, queries, truth, active in panel:
        for f in range(history.shape[0]):
            row = history[f, 0]
            stripped = []
            i = 0
            while i < len(row):
                if row[i] == COPY:
                    i += 3
                    continue
                stripped.append(int(row[i]))
                i += 1
            without = literal_reference(np.array([stripped], np.int32), queries[f, 0][None, :, 1])
            assert not np.array_equal(without[0], truth[f, 0])
            assert all(without[0][i] != truth[f, 0][i] for i in np.flatnonzero(active[f]))


def test_the_gap_sits_halfway_through_the_stages_and_touches_no_target(panel):
    for depth, gap, history, queries, truth, active in panel:
        if not gap:
            continue
        start = gap_start(depth)
        for f in range(history.shape[0]):
            for w in range(2):
                region = history[f, w, start:start + gap]
                assert set(region[1::8]) <= set(range(16, 32))
                assert not set(region[1::8]) & set(int(k) for k in queries[f, w, :, 1])
                assert history[f, w, start - 3] in (COPY, STORE_TOKEN)
                assert history[f, w, start + gap] == COPY
                # The distractor values really change from block to block.
                first = region[2:64:8]
                second = region[64 + 2:128:8]
                assert not np.array_equal(first, second)
                assert set(first) == set(second)


def test_each_public_copy_is_immediately_followed_by_a_write_to_its_source(panel):
    for depth, gap, history, _, truth, active in panel:
        split = gap_start(depth)
        stages = np.concatenate((history[:, :, PREFIX_TOKENS:split],
                                 history[:, :, split+gap:-SUFFIX_TOKENS]), axis=-1)
        pairs = stages.reshape(*stages.shape[:2], depth*CHAINS, STAGE_TOKENS)
        assert np.all(pairs[..., 0] == COPY)
        assert np.all(pairs[..., 3] == STORE_TOKEN)
        np.testing.assert_array_equal(pairs[..., 2], pairs[..., 4])
        for f in range(len(history)):
            assert not np.isin(pairs[f, :, :, 5], truth[f, :, active[f]]).any()


def test_near_and_far_differ_only_by_the_gap(panel):
    by_depth = {}
    for depth, gap, history, queries, truth, active in panel:
        by_depth.setdefault(depth, {})[gap] = (history, queries, truth, active)
    for depth, cases in by_depth.items():
        near, far = cases[0], cases[1024]
        start = gap_start(depth)
        for f in range(near[0].shape[0]):
            for w in range(2):
                trimmed = np.concatenate((far[0][f, w, :start], far[0][f, w, start + 1024:]))
                np.testing.assert_array_equal(trimmed, near[0][f, w])
        np.testing.assert_array_equal(near[1], far[1])
        np.testing.assert_array_equal(near[2], far[2])
        np.testing.assert_array_equal(near[3], far[3])


def test_the_suffix_carries_no_answer(panel):
    for depth, gap, history, queries, truth, active in panel:
        suffix = history[:, :, -SUFFIX_TOKENS:]
        assert np.all(suffix == FILLER_TOKEN)
        assert not np.any(history == QUERY_TOKEN)
        for f in range(history.shape[0]):
            for w in range(2):
                assert not set(truth[f, w].tolist()) & set(history[f, w, -SUFFIX_TOKENS:].tolist())


def test_seed_and_family_stability():
    first = list(versioned_cases(seed=732013, families=2, depths=(2,), gaps=(0,)))
    again = list(versioned_cases(seed=732013, families=2, depths=(2,), gaps=(0,)))
    other = list(versioned_cases(seed=732014, families=2, depths=(2,), gaps=(0,)))
    for a, b in zip(first, again, strict=True):
        for x, y in zip(a[2:], b[2:], strict=True):
            np.testing.assert_array_equal(x, y)
    assert not np.array_equal(first[0][4], other[0][4])
    wider = list(versioned_cases(seed=732013, families=8, depths=(2,), gaps=(0,)))
    np.testing.assert_array_equal(wider[0][2][0], first[0][2][0])
    np.testing.assert_array_equal(wider[0][3][0], first[0][3][0])


def test_routes_are_prefix_stable_across_depths():
    shallow = list(versioned_cases(families=2, depths=(2,), gaps=(0,)))[0]
    deep = list(versioned_cases(families=2, depths=(24,), gaps=(0,)))[0]
    start = gap_start(2)
    stage = PREFIX_TOKENS
    for f in range(shallow[2].shape[0]):
        np.testing.assert_array_equal(shallow[2][f, :, stage:start], deep[2][f, :, stage:start])


def test_a_copy_survives_a_later_overwrite_of_its_source():
    history = np.array([[STORE_TOKEN, 16, 33, COPY, 17, 16, STORE_TOKEN, 16, 40]], np.int32)
    assert literal_reference(history, np.array([[16, 17, 18]], np.int32)).tolist() == [[40, 33, -2]]


def test_repeated_copies_keep_the_original_value():
    history = np.array([[
        STORE_TOKEN, 16, 33, COPY, 17, 16, STORE_TOKEN, 16, 40,
        COPY, 18, 17, STORE_TOKEN, 17, 41, COPY, 19, 18, STORE_TOKEN, 18, 42,
    ]], np.int32)
    # The last copy still carries the original value of key 16.
    assert literal_reference(history, np.array([[16, 17, 18, 19]], np.int32)).tolist() == [[40, 41, 42, 33]]


def test_a_copy_of_an_unbound_source_leaves_the_destination_unbound():
    history = np.array([[STORE_TOKEN, 16, 33, COPY, 17, 18]], np.int32)
    assert literal_reference(history, np.array([[16, 17, 18]], np.int32)).tolist() == [[33, -2, -2]]


def test_malformed_events_and_missing_keys():
    history = np.array([[
        STORE_TOKEN, 16, 33,          # legal
        STORE_TOKEN, 17, 50,          # value outside the literal range
        STORE_TOKEN, 40, 34,          # key outside the key range
        COPY, 18, 50,                 # source is not a key
        STORE_TOKEN, 16,              # truncated event
        COPY, 19, 16,                 # legal copy of the first write
    ]], np.int32)
    assert literal_reference(history, np.array([[16, 17, 18, 19, 20]], np.int32)).tolist() == [[33, -2, -2, 33, -2]]


def test_the_reference_only_accepts_matching_batches():
    with pytest.raises(ValueError):
        literal_reference(np.zeros((2, 4), np.int32), np.zeros((3, 2), np.int32))


def test_only_midchain_destinations_explain_the_counterfactual_answers():
    from probe_middle_copy_swap import measure, middle_swap_case

    case, true_reads = middle_swap_case(families=4)
    _, depth, _, _, history, questions, _, _ = case
    np.testing.assert_array_equal(questions[:, 0], questions[:, 1])
    for w0, w1 in history:
        starts = np.flatnonzero(w0 == COPY)
        pair = starts[(depth//2 - 1)*CHAINS:][:2] + 1
        np.testing.assert_array_equal(np.flatnonzero(w0 != w1), pair)
        np.testing.assert_array_equal(w0[pair], w1[pair[::-1]])
        assert np.unique(w0[starts + 5]).size == 1  # No poison token identifies a lineage.
    changed, n, _, _, correct, programs, diff = measure(case, np.repeat(true_reads[:, :1], 2, axis=1))
    assert (changed, n) == (0, 8)
    assert (correct, programs, diff) == (0, 4, 4)
