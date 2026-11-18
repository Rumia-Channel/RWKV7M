from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return import_module('benchmark_binding_state')


@pytest.fixture
def guard_cases(bench):
    return {(task, gap): (seed, history, queries, labels)
            for task, gap, seed, history, queries, labels in bench.panel(16, guard_study=True)}


def _writes(tokens):
    # Independent public-token oracle: do not use the benchmark's event parser.
    result = []
    for position in range(len(tokens)-2):
        marker, key, value = map(int, tokens[position:position+3])
        if marker == 2 and 16 <= key < 32 and 32 <= value < 48:
            result.append((key, value))
    return result


def test_guard_defaults_and_explicit_seeds_preserve_legacy_cases(bench):
    legacy = list(bench.panel(2))
    cases = [('near', 0), ('repeat', 128), ('change', 128), ('repeat', 1024),
             ('change', 1024), ('filler', 1024), ('overwrite', 32)]
    assert [(task, gap) for task, gap, *_ in legacy] == cases
    assert [seed for _, _, seed, *_ in legacy] == [731907]*6 + [731908]
    guarded = list(bench.panel(2, guard_study=True))
    assert [(task, gap) for task, gap, *_ in guarded] == cases + [('mixed', 0), ('mixed', 1024)]
    assert [seed for _, _, seed, *_ in guarded] == [731927]*6 + [731928, 731927, 731927]
    same_seeds = list(bench.panel(2, 731927, 731928))
    for old, new in zip(same_seeds, guarded[:7], strict=True):
        assert old[:3] == new[:3]
        for left, right in zip(old[3:], new[3:], strict=True):
            np.testing.assert_array_equal(left, right)
    assert bench._panel_seeds(19, None, True) == (19, 731928)
    assert bench._panel_seeds(None, 23, True) == (731927, 23)
    assert bench._panel_seeds(19, 23, True) == (19, 23)
    explicit = list(bench.panel(2, 19, 23, guard_study=True))
    assert [seed for _, _, seed, *_ in explicit] == [19]*6 + [23, 19, 19]


def test_mixed_near_and_far_share_queries_and_latest_answers(bench, guard_cases):
    _, near, queries, original = guard_cases['near', 0]
    _, far, far_queries, far_labels = guard_cases['change', 1024]
    _, mixed_near, near_queries, near_labels = guard_cases['mixed', 0]
    _, mixed_far, mixed_queries, mixed_labels = guard_cases['mixed', 1024]
    np.testing.assert_array_equal(queries, far_queries)
    np.testing.assert_array_equal(queries, near_queries)
    np.testing.assert_array_equal(queries, mixed_queries)
    np.testing.assert_array_equal(original, far_labels)
    np.testing.assert_array_equal(near_labels, mixed_labels)
    np.testing.assert_array_equal(mixed_near[..., :-24], near)
    np.testing.assert_array_equal(mixed_far[..., :-24], far)
    np.testing.assert_array_equal(mixed_near[..., -24:], mixed_far[..., -24:])
    assert mixed_far.shape[-1] - mixed_near.shape[-1] == 1024
    assert np.all(mixed_near[..., -8:] == bench.FILLER)
    assert np.all(mixed_far[..., -8:] == bench.FILLER)
    np.testing.assert_array_equal(queries[:, 0], queries[:, 1])
    assert np.all(near_labels[:, 0] != near_labels[:, 1])
    assert np.all(mixed_labels[:, 0] != mixed_labels[:, 1])


@pytest.mark.parametrize('gap,base', [(0, ('near', 0)), (1024, ('change', 1024))])
def test_mixed_updates_are_true_latest_not_obsolete_and_preserve_value_sets(guard_cases, gap, base):
    _, before, queries, obsolete = guard_cases[base]
    _, history, mixed_queries, labels = guard_cases['mixed', gap]
    np.testing.assert_array_equal(queries, mixed_queries)
    for family, worlds in enumerate(history):
        updates_by_world = []
        for world, tokens in enumerate(worlds):
            previous_events = _writes(before[family, world])
            events = _writes(tokens)
            assert events[:-4] == previous_events
            assert len(events) == len(previous_events) + 4
            updates = events[-4:]
            updated_keys = {key for key, _ in updates}
            assert len(updated_keys) == 4
            assert updated_keys < set(map(int, queries[family, world, :, 1]))
            previous, latest = dict(previous_events), dict(events)
            assert all(value != previous[key] for key, value in updates)
            assert sorted(value for _, value in updates) == sorted(previous[key] for key in updated_keys)
            assert sorted(latest.values()) == sorted(previous.values())
            np.testing.assert_array_equal(labels[family, world],
                                          [latest[int(key)] for key in queries[family, world, :, 1]])
            changed = labels[family, world] != obsolete[family, world]
            np.testing.assert_array_equal(changed,
                                          [int(key) in updated_keys for key in queries[family, world, :, 1]])
            assert changed.sum() == 4
            updates_by_world.append(tuple(key for key, _ in updates))
        assert updates_by_world[0] == updates_by_world[1]


def test_recent_flags_follow_last_capacity_legal_events_for_every_guard_row(bench, guard_cases):
    for _, history, queries, _ in guard_cases.values():
        flags = bench._recent_flags(history, queries, 16)
        assert flags.shape == (16, queries.shape[2])
        assert set(np.unique(flags)) <= {0, 1}
        for family, worlds in enumerate(history):
            chronology = [tuple(key for key, _ in _writes(tokens)) for tokens in worlds]
            assert chronology[0] == chronology[1]
            recent = set(chronology[0][-16:])
            np.testing.assert_array_equal(flags[family],
                                          [int(key) in recent for key in queries[family, 0, :, 1]])
    _, far, queries, _ = guard_cases['mixed', 1024]
    flags = bench._recent_flags(far, queries, 16)
    np.testing.assert_array_equal(flags.sum(axis=1), np.full(16, 4))
    assert len({tuple(row) for row in flags}) > 1
    _, filler, filler_queries, _ = guard_cases['filler', 1024]
    assert np.all(bench._recent_flags(filler, filler_queries, 16) == 1)
    _, changing, changing_queries, _ = guard_cases['change', 1024]
    assert np.all(bench._recent_flags(changing, changing_queries, 16) == 0)


def test_mixed_subsets_are_deterministic_and_vary_across_families(bench, guard_cases):
    smaller = {(task, gap): (history, queries, labels)
               for task, gap, _, history, queries, labels in bench.panel(8, guard_study=True)}
    subsets = set()
    for gap in (0, 1024):
        _, history, queries, labels = guard_cases['mixed', gap]
        for actual, expected in zip(smaller['mixed', gap], (history[:8], queries[:8], labels[:8]), strict=True):
            np.testing.assert_array_equal(actual, expected)
        for family, worlds in enumerate(history):
            query_order = list(map(int, queries[family, 0, :, 1]))
            updated = {key for key, _ in _writes(worlds[0])[-4:]}
            subsets.add(tuple(key in updated for key in query_order))
    assert len(subsets) > 1
    assert any(subset != (True, True, True, True, False, False, False, False) for subset in subsets)
    assert any(subset != (False, False, False, False, True, True, True, True) for subset in subsets)


def test_recency_is_per_key_membership_not_global_input_length(bench, guard_cases):
    _, history, queries, _ = guard_cases['mixed', 1024]
    history, queries = history[:1], queries[:1]
    flags = bench._recent_flags(history, queries, 16)
    padded = np.pad(history, ((0, 0), (0, 0), (0, 257)), constant_values=bench.FILLER)
    np.testing.assert_array_equal(bench._recent_flags(padded, queries, 16), flags)
    changed = history.copy()
    stale_key = int(queries[0, 0, np.flatnonzero(flags[0] == 0)[0], 1])
    last_store = np.flatnonzero(changed[0, 0] == bench.STORE)[-1]
    changed[0, :, last_store+1] = stale_key
    updated_flags = bench._recent_flags(changed, queries, 16)
    assert changed.shape == history.shape
    assert not np.array_equal(updated_flags, flags)
    assert flags.sum() == updated_flags.sum() == 4


def test_recent_flags_ignore_incomplete_and_illegal_events(bench, guard_cases):
    _, history, queries, _ = guard_cases['mixed', 1024]
    history, queries = history[:1], queries[:1]
    flags = bench._recent_flags(history, queries, 1)
    missing_key = int(queries[0, 0, np.flatnonzero(flags[0] == 0)[0], 1])
    invalid = np.asarray([bench.STORE, missing_key, bench.FILLER,
                          bench.STORE, 15, 32, bench.STORE, missing_key], np.int32)
    appended = np.concatenate((history, np.broadcast_to(invalid, (1, 2, len(invalid)))), axis=-1)
    np.testing.assert_array_equal(bench._recent_flags(appended, queries, 1), flags)
    completed = np.concatenate((appended, np.full((1, 2, 1), 32, np.int32)), axis=-1)
    np.testing.assert_array_equal(bench._recent_flags(completed, queries, 1),
                                  (queries[:, 0, :, 1] == missing_key).astype(np.int32))


def test_recent_flags_verify_world_key_chronology(bench, guard_cases):
    _, history, queries, _ = guard_cases['mixed', 1024]
    changed = history[:1].copy()
    last_store = np.flatnonzero(changed[0, 1] == bench.STORE)[-1]
    old_key = int(changed[0, 1, last_store+1])
    changed[0, 1, last_store+1] = 16 + (old_key-15) % 16
    with pytest.raises(ValueError, match='chronology'):
        bench._recent_flags(changed, queries[:1], 16)
    changed_queries = queries[:1].copy()
    changed_queries[0, 1, :, 1] = np.roll(changed_queries[0, 1, :, 1], 1)
    with pytest.raises(ValueError, match='query keys'):
        bench._recent_flags(history[:1], changed_queries, 16)
