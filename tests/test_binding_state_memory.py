from collections import OrderedDict, deque
from importlib import import_module
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rwkv7m import ModelConfig, save_model_safetensors
from rwkv7m.model.nnx_conversion import nnx_params_to_linen
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return import_module('binding_state_memory')


def state(markers):
    # Distinct shapes/dtypes represent complete state leaves, not cached answers.
    return {'rwkv': jnp.asarray(markers, jnp.float32)[:, None],
            'screen': jnp.asarray(markers, jnp.bfloat16)[:, None, None]}


def read_markers(memory, live, fifo, keyed, keys):
    keys = jnp.asarray(keys, jnp.int32)
    return np.asarray(memory._fork_states(live, fifo, keyed, keys)['rwkv']).reshape(3, *keys.shape)


def commits(memory, keys, capacity=2):
    fifo, keyed = memory._empty_banks(state([0]), capacity)
    for i, key in enumerate(keys, 1):
        fifo, keyed = memory._commit(state([i * 100]), fifo, keyed, jnp.asarray([key], jnp.int32))
    return fifo, keyed


def guarded_commits(memory, keys, capacity=2):
    recent, keyed = memory._empty_guarded_banks(state([0]), capacity)
    for i, key in enumerate(keys, 1):
        recent, keyed = memory._commit_guarded(state([i * 100]), recent, keyed,
                                              jnp.asarray([key], jnp.int32))
    return recent, keyed


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-5)


def test_empty_banks_fall_back_to_each_lanes_live_state(memory):
    live = state([123, 456])
    fifo, keyed = memory._empty_banks(live, 2)
    result = read_markers(memory, live, fifo, keyed, [[16, 17], [18, 19]])
    np.testing.assert_array_equal(result, np.broadcast_to([[123, 123], [456, 456]], (3, 2, 2)))
    assert memory.ARMS == ('live', 'fifo', 'keyed')


def test_fifo_counts_duplicate_commits_but_keyed_upserts(memory):
    fifo, keyed = commits(memory, [16, 17, 17])
    result = read_markers(memory, state([999]), fifo, keyed, [[16, 17, 18]])
    np.testing.assert_array_equal(result[:, 0], [[999, 999, 999], [999, 300, 999], [100, 300, 999]])
    np.testing.assert_array_equal(fifo.keys, [[17, 17]])
    assert sorted(np.asarray(keyed.keys)[0]) == [16, 17]


def test_reassign_refreshes_keyed_write_order_and_reads_latest_snapshot(memory):
    fifo, keyed = commits(memory, [16, 17, 16, 18])
    result = read_markers(memory, state([999]), fifo, keyed, [[16, 17, 18]])
    np.testing.assert_array_equal(result[1:, 0], [[300, 999, 400], [300, 999, 400]])
    # FIFO must choose the newest match even while both duplicate commits remain.
    fifo, keyed = commits(memory, [16, 16, 17], capacity=3)
    np.testing.assert_array_equal(
        read_markers(memory, state([999]), fifo, keyed, [[16]])[:, 0, 0], [999, 200, 200],
    )


def test_keyed_write_ranks_do_not_lose_order_after_many_reassignments(memory):
    fifo, keyed = commits(memory, [16, 17, 18] + [18] * 9 + [19], capacity=3)
    result = read_markers(memory, state([9999]), fifo, keyed, [[16, 17, 18, 19]])
    np.testing.assert_array_equal(result[2, 0], [9999, 200, 1200, 1300])
    np.testing.assert_array_equal(np.sort(np.asarray(keyed.ranks)[0]), [0, 1, 2])


@pytest.mark.parametrize('capacity', [1, 2, 16])
def test_banks_remain_bounded_and_copy_all_state_leaves(memory, capacity):
    fifo, keyed = commits(memory, list(range(16, 32)) * 2, capacity)
    for bank in (fifo, keyed):
        assert bank.keys.shape == (1, capacity)
        assert bank.states['rwkv'].shape == (1, capacity, 1)
        assert bank.states['screen'].shape == (1, capacity, 1, 1)
        assert bank.states['screen'].dtype == jnp.bfloat16
    fork = memory._fork_states(state([9999]), fifo, keyed, jnp.asarray([[31]], jnp.int32))
    np.testing.assert_array_equal(fork['rwkv'][:, 0], [9999, 3200, 3200])
    np.testing.assert_array_equal(fork['screen'][:, 0, 0], jnp.asarray([9999, 3200, 3200], jnp.bfloat16))


def test_nonsynchronous_commits_do_not_change_other_lanes(memory):
    live = state([999, 888])
    fifo, keyed = memory._empty_banks(live, 2)
    for keys, values in [([16, 17], [100, 200]), ([16, -1], [300, 400]), ([18, 17], [500, 600])]:
        fifo, keyed = memory._commit(state(values), fifo, keyed, jnp.asarray(keys, jnp.int32))
    result = read_markers(memory, live, fifo, keyed, [[16, 17, 18], [16, 17, 18]])
    np.testing.assert_array_equal(result[1:], np.broadcast_to([[300, 999, 500], [888, 600, 888]], (2, 2, 3)))


def test_queries_are_independent_and_do_not_refresh_replacement_order(memory):
    fifo, keyed = commits(memory, [16, 17])
    before = [np.asarray(x).copy() for x in jax.tree.leaves((fifo, keyed))]
    first = read_markers(memory, state([999]), fifo, keyed, [[16, 17, 16]])
    reversed_queries = read_markers(memory, state([999]), fifo, keyed, [[16, 17, 16][::-1]])
    np.testing.assert_array_equal(first[:, :, ::-1], reversed_queries)
    for old, current in zip(before, jax.tree.leaves((fifo, keyed)), strict=True):
        np.testing.assert_array_equal(current, old)
    fifo, keyed = memory._commit(state([300]), fifo, keyed, jnp.asarray([18], jnp.int32))
    assert read_markers(memory, state([999]), fifo, keyed, [[16]])[2, 0, 0] == 999


@pytest.mark.parametrize('keys,capacity,questions,expected', [
    ([], 2, [16, 17, 31, -1, 15, 32], [999] * 6),
    ([16, 17, 17], 2, [16, 17, 18, 31], [100, 999, 999, 999]),
    ([16, 17, 16, 18, 18], 2, [16, 17, 18, 31], [300, 999, 999, 999]),
    ([16, 17, 17, -1, 15, 32], 2, [16, 17, 18], [100, 999, 999]),
    ([16, 17, 16], 1, [16, 17, 31], [999, 999, 999]),
])
def test_guard_routes_each_query_by_recent_writes_with_live_fallback(
        memory, keys, capacity, questions, expected):
    recent, keyed = guarded_commits(memory, keys, capacity)
    before = [np.asarray(x).copy() for x in jax.tree.leaves((recent, keyed))]
    query_keys = jnp.asarray([questions], jnp.int32)
    routed = memory._fork_guarded_states(state([999]), recent, keyed, query_keys)
    np.testing.assert_array_equal(routed['rwkv'][:, 0], expected)
    np.testing.assert_array_equal(routed['screen'][:, 0, 0], jnp.asarray(expected, jnp.bfloat16))
    reversed_states = memory._fork_guarded_states(state([999]), recent, keyed, query_keys[:, ::-1])
    assert_tree_close(reversed_states, jax.tree.map(lambda x: x[::-1], routed))
    for old, current in zip(before, jax.tree.leaves((recent, keyed)), strict=True):
        np.testing.assert_array_equal(current, old)


def test_guarded_commits_match_comparative_tags_and_keyed_states_per_lane(memory):
    live = state([999, 888])
    fifo, keyed = memory._empty_banks(live, 2)
    recent, guarded_keyed = memory._empty_guarded_banks(live, 2)
    events = [([16, 17], [100, 200]), ([16, -1], [300, 400]),
              ([18, 17], [500, 600]), ([18, 16], [700, 800]),
              ([15, 32], [900, 1000]), ([18, 16], [1100, 1200])]
    for keys, markers in events:
        keys = jnp.asarray(keys, jnp.int32)
        fifo, keyed = memory._commit(state(markers), fifo, keyed, keys)
        recent, guarded_keyed = memory._commit_guarded(state(markers), recent, guarded_keyed, keys)
        np.testing.assert_array_equal(recent.keys, fifo.keys)
        np.testing.assert_array_equal(recent.cursor, fifo.cursor)
        assert_tree_close(guarded_keyed, keyed)


@pytest.mark.parametrize('capacity', [1, 2, 16])
def test_guarded_resident_payload_is_one_state_bank_plus_bounded_tags(memory, capacity):
    live = state([0, 0])
    recent, keyed = memory._empty_guarded_banks(live, capacity)
    for i, key in enumerate(list(range(16, 32)) * 2, 1):
        recent, keyed = memory._commit_guarded(
            state([i * 100, i * 200]), recent, keyed, jnp.asarray([key, 47 - key], jnp.int32),
        )
    assert recent._fields == ('keys', 'cursor')
    assert recent.keys.shape == keyed.keys.shape == keyed.ranks.shape == (2, capacity)
    assert recent.cursor.shape == (2,)
    assert keyed.states['rwkv'].shape == (2, capacity, 1)
    assert keyed.states['screen'].shape == (2, capacity, 1, 1)
    assert keyed.states['screen'].dtype == jnp.bfloat16
    assert memory._state_bytes(keyed.states) == capacity * memory._state_bytes(live)
    assert memory._state_bytes((recent, keyed)) == (
        capacity * memory._state_bytes(live) + 2 * (3 * capacity + 1) * 4
    )
    assert np.all((np.asarray(keyed.ranks) >= 0) & (np.asarray(keyed.ranks) < capacity))


@pytest.mark.parametrize('stride', [4, 8])
def test_aligned_writes_use_event_blocks(memory, stride):
    history = np.ones((4, stride * 15 + 3), np.int32)
    history[:, 0::stride] = 2
    history[:, 1::stride] = 16
    history[:, 2::stride] = 32
    assert memory._commit_grid(history) == (3, history.shape[1], stride)


def test_grid_supports_nonsynchronous_events_and_ignores_incomplete_or_invalid_writes(memory):
    history = np.ones((2, 15), np.int32)
    history[0, 0:3] = [2, 16, 32]
    history[1, 1:4] = [2, 17, 33]
    history[0, 5:8] = [2, 15, 34]
    history[1, 7:10] = [2, 18, 48]
    history[0, -2:] = [2, 19]
    assert memory._commit_grid(history) == (3, 4, 1)
    assert memory._commit_grid(np.ones((2, 130), np.int32)) == (None, None, 64)


@pytest.mark.parametrize('length', [0, 1, 2, 24, 63, 64])
def test_replay_mask_never_resets_short_histories(memory, length):
    history = np.ones((2, 2, length), np.int32)
    if length >= 3:
        history[..., -3:] = [2, 16, 32]
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 31, 4]], np.int32), (2, 2, 2, 3))
    result = memory.replay_mask(history, queries, capacity=16)
    assert memory.REPLAY_TOKENS == 64
    assert result.shape == (2, 2, 2) and result.dtype == np.bool_
    assert not result.any()


@pytest.mark.parametrize('capacity', [1, 2, 3, 4, 16])
def test_replay_mask_counts_ragged_complete_events_per_lane_not_tokens_or_distinct_keys(
        memory, capacity):
    history = np.ones((2, 2, 98), np.int32)
    events = [
        (0, 0, [(33, 16, 32), (37, 17, 33), (45, 18, 34), (56, 18, 35),
                (62, 15, 32), (68, 31, 48), (79, 19, 36), (86, 32, 47), (95, 20, 37)]),
        (0, 1, [(34, 16, 32), (46, 17, 33), (60, 16, 34), (78, 19, 31), (90, 18, 35)]),
        (1, 0, [(33, 20, 36)]),
        (1, 1, [(0, 31, 47), (34, 31, 47), (95, 30, 46)]),
    ]
    for family, world, lane_events in events:
        for start, key, value in lane_events:
            history[family, world, start:start + 3] = [2, key, value]
    history[0, 1, -2:] = [2, 31]
    history[1, 0, -2:] = [2, 31]
    queries = np.broadcast_to(np.array([[3, key, 4] for key in range(16, 32)], np.int32),
                              (2, 2, 16, 3)).copy()
    expected = np.zeros((2, 2, 16), np.bool_)
    for family in range(2):
        for world in range(2):
            suffix = history[family, world, -64:]
            completed = []
            for start in range(len(suffix) - 2):
                store, key, value = suffix[start:start + 3]
                if store == 2 and 16 <= key < 32 and 32 <= value < 48:
                    completed.append(int(key))
            expected[family, world] = np.isin(queries[family, world, :, 1], completed[-capacity:])
    actual = memory.replay_mask(history, queries, capacity)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.bool_
    assert not actual[0, 0, 0]  # The key-16 STORE is just outside the suffix.
    assert not actual[1, 0].any()  # Neither partial boundary is a completed write.
    if capacity == 2:
        for family, world, keys in [(0, 0, [19, 20]), (0, 1, [16, 18]), (1, 1, [30, 31])]:
            np.testing.assert_array_equal(np.flatnonzero(actual[family, world]) + 16, keys)
    np.testing.assert_array_equal(
        memory.replay_mask(history, queries[..., ::-1, :], capacity), actual[..., ::-1],
    )
    np.testing.assert_array_equal(
        memory.replay_mask(history[::-1, ::-1], queries[::-1, ::-1], capacity), actual[::-1, ::-1],
    )


def test_only_nonfinite_vectors_are_invalid(memory):
    logits = np.full((5, 128), -5, np.float32)
    logits[:, 33] = 10
    logits[1, 90] = np.nan
    logits[2, 90] = np.inf
    logits[3, 90] = -np.inf
    logits[4, 2] = 20
    np.testing.assert_array_equal(memory._predictions(jnp.asarray(logits)), [33, -1, -1, -1, 2])


@pytest.mark.parametrize('token', [0, 15, 16, 31, 32, 47, 48, 127])
def test_finite_argmax_is_preserved_across_value_boundaries(memory, token):
    logits = np.full((1, 128), -5, np.float32)
    logits[0, token] = 10
    prediction = np.asarray(memory._predictions(jnp.asarray(logits)))
    assert prediction.dtype == np.int32
    np.testing.assert_array_equal(prediction, [token])


def test_state_bytes_counts_all_actual_shapes_and_dtypes(memory):
    leaves = (jnp.zeros((1, 2, 3), jnp.float32),
              {'screen': jnp.zeros((1, 7), jnp.bfloat16), 'occupancy': jnp.zeros((1, 5), jnp.bool_)})
    assert memory._state_bytes(leaves) == 6 * 4 + 7 * 2 + 5


@pytest.mark.parametrize('mode', ['guarded', 'fresh', 'replayed'])
def test_query_paths_evaluate_only_one_selected_batch_and_head(memory, mode):
    class ReadProbe(nnx.Module):
        def __init__(self):
            self.read_counts = nnx.Variable(jnp.zeros(2, jnp.int32))
            self.head_counts = nnx.Variable(jnp.zeros(2, jnp.int32))

        def compute_recurrent_hidden(self, ids, rwkv, screen, *, phase, deterministic):
            assert phase == 'read_write' and deterministic
            self.read_counts[...] += jnp.asarray([1, ids.shape[0]], jnp.int32)
            hidden = jnp.broadcast_to(rwkv[:, None, :], (ids.shape[0], ids.shape[1], 1))
            return hidden, rwkv, screen, None

        def compute_logits(self, hidden):
            self.head_counts[...] += jnp.asarray([1, hidden.shape[0]], jnp.int32)
            return jax.nn.one_hot(hidden[..., 0].astype(jnp.int32), 128)

    live = (jnp.array([[90]], jnp.float32), jnp.array([[[90]]], jnp.bfloat16))
    recent, keyed = memory._empty_guarded_banks(live, 2)
    for marker, key in [(20, 16), (30, 17), (40, 17)]:
        checkpoint = jax.tree.map(lambda x: jnp.full_like(x, marker), live)
        recent, keyed = memory._commit_guarded(checkpoint, recent, keyed, jnp.array([key], jnp.int32))
    queries = jnp.array([[[3, 16, 4], [3, 17, 4], [3, 31, 4]]], jnp.int32)
    model = ReadProbe()
    fresh = jax.tree.map(lambda x: jnp.full_like(x, 60), live)
    if mode == 'guarded':
        result = memory._read_guarded_queries(model, queries, live, recent, keyed)
        expected = [[20, 90, 90]]
    elif mode == 'fresh':
        result = memory._read_fresh_queries(model, queries, fresh)
        expected = [[60, 60, 60]]
    else:
        result = memory._read_replayed_queries(
            model, queries, live, recent, keyed, fresh, jnp.array([[True, False, True]]),
        )
        expected = [[60, 90, 60]]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(model.read_counts[...], [1, 3])
    np.testing.assert_array_equal(model.head_counts[...], [1, 3])


@pytest.fixture(scope='module')
def native_weights(tmp_path_factory):
    config = ModelConfig(d_model=16, d_ffn=32, n_layers=2, n_heads=4, head_size=4,
                         vocab_size=128, max_seq_len=64, dtype='float32', use_screening=False)
    model = initialize_nnx_model(jax.random.key(481), config)
    rng = np.random.default_rng(982)
    # Native initializers zero residual outputs; make recurrence observable.
    for i in range(config.n_layers):
        block = getattr(getattr(model, f'layer_{i}'), f'rwkv_block_{i}')
        for projection in (block.att.output, block.ffn.value):
            projection.kernel[...] = jnp.asarray(rng.normal(0, 0.08, projection.kernel[...].shape), jnp.float32)
    # Paired +/- columns ensure a native unrestricted argmax is a value token.
    head = np.zeros(model.lm_head.kernel[...].shape, np.float32)
    head[:, 32:40] = rng.normal(0, 0.2, (config.d_model, 8))
    head[:, 40:48] = -head[:, 32:40]
    model.lm_head.kernel[...] = jnp.asarray(head)
    path = tmp_path_factory.mktemp('binding-state-native') / 'model.safetensors'
    save_model_safetensors(path, nnx_params_to_linen(model), config,
                           tokenizer_metadata={'tokenizer_format': 'synthetic_delayed_kv'})
    return path


@nnx.jit
def native_forward(model, ids, initial):
    hidden, rwkv, screen, _ = model.compute_recurrent_hidden(
        ids, *initial, phase='read_write', deterministic=True,
    )
    logits = model.compute_logits(hidden[:, -1:, :])[:, 0, :]
    return logits, (rwkv, screen)


def initial(reader, batch=1):
    config = reader._config
    return init_rwkv_state(batch, config), init_screen_state(batch, config.screening)


def reference_endpoints(tokens, capacity):
    fifo = deque(maxlen=capacity)
    keyed = OrderedDict()
    for end in range(3, len(tokens) + 1):
        store, key, value = tokens[end - 3:end]
        if store == 2 and 16 <= key < 32 and 32 <= value < 48:
            fifo.append((int(key), end))
            keyed.pop(int(key), None)
            keyed[int(key)] = end
            if len(keyed) > capacity:
                keyed.popitem(last=False)
    return dict(fifo), dict(keyed)


def paired_history():
    history = np.ones((1, 2, 22), np.int32)
    for world, events in enumerate([
        [(0, 16, 32), (4, 17, 33), (8, 16, 34), (12, 18, 35)],
        [(1, 16, 36), (5, 17, 37), (10, 16, 38), (15, 18, 39)],
    ]):
        for position, key, value in events:
            history[0, world, position:position + 3] = [2, key, value]
    return history


def replay_history():
    history = np.ones((1, 2, 103), np.int32)
    for world, events in enumerate([
        [(1, 16, 32), (9, 18, 33), (38, 16, 34), (43, 17, 35),
         (56, 17, 36), (73, 17, 37), (83, 19, 48), (96, 17, 38)],
        [(2, 17, 39), (11, 18, 40), (39, 16, 41),
         (51, 16, 42), (65, 16, 43), (79, 31, 31), (92, 16, 44)],
    ]):
        for position, key, value in events:
            history[0, world, position:position + 3] = [2, key, value]
    history[..., -2:] = [2, 20]
    return history


def test_native_snapshots_match_complete_causal_prefixes_and_genuine_neural_reads(memory, native_weights):
    reader = memory.SnapshotReader(native_weights, capacity=2)
    history = paired_history()
    live, fifo, keyed = reader._prefill(history.reshape(2, -1))
    assert memory._commit_grid(history.reshape(2, -1))[2] == 1
    assert reader.state_bytes == sum(np.asarray(x).nbytes for x in jax.tree.leaves(initial(reader)))
    endpoints = [reference_endpoints(tokens, 2) for tokens in history[0]]
    for world, tokens in enumerate(history[0]):
        _, expected_live = native_forward(reader._model, jnp.asarray(tokens[None]), initial(reader))
        assert_tree_close(jax.tree.map(lambda x: x[world:world + 1], live), expected_live)
        for policy, bank in enumerate((fifo, keyed)):
            for slot, key in enumerate(np.asarray(bank.keys)[world]):
                end = endpoints[world][policy][int(key)]
                _, expected = native_forward(reader._model, jnp.asarray(tokens[None, :end]), initial(reader))
                actual = jax.tree.map(lambda x: x[world:world + 1, slot], bank.states)
                assert_tree_close(actual, expected)

    # Read the chosen keyed snapshot and compare full vocabulary logits against
    # a separate native rollout of exactly prefix + raw query, not a label.
    queries = np.array([[[3, 16, 4]], [[3, 16, 4]]], np.int32)
    forked = memory._fork_states(live, fifo, keyed, jnp.asarray(queries[:, :, 1]))
    selected = jax.tree.map(lambda x: x[4:6], forked)  # keyed arm, one question per world
    actual_logits, _ = native_forward(reader._model, jnp.asarray(queries[:, 0]), selected)
    prediction = reader.predict(history, queries[None])['keyed'][0, :, 0]
    np.testing.assert_array_equal(prediction, np.asarray(actual_logits).argmax(axis=-1))
    for world, tokens in enumerate(history[0]):
        end = endpoints[world][1][16]
        ids = np.concatenate([tokens[:end], queries[world, 0]])[None]
        expected_logits, _ = native_forward(reader._model, jnp.asarray(ids), initial(reader))
        np.testing.assert_allclose(actual_logits[world], expected_logits[0], rtol=3e-5, atol=3e-5)
        for wrong_end in (end - 1, end + 1, len(tokens)):
            wrong_ids = np.concatenate([tokens[:wrong_end], queries[world, 0]])[None]
            wrong_logits, _ = native_forward(reader._model, jnp.asarray(wrong_ids), initial(reader))
            assert not np.allclose(actual_logits[world], wrong_logits[0], rtol=1e-6, atol=1e-7)

    # Changing future tokens cannot alter an already-committed snapshot.
    changed = history.copy()
    changed[..., -3:] = [3, 31, 4]
    _, future_fifo, future_keyed = reader._prefill(changed.reshape(2, -1))
    assert_tree_close(future_fifo, fifo)
    assert_tree_close(future_keyed, keyed)


def test_native_api_is_independent_across_families_worlds_queries_and_calls(memory, native_weights):
    reader = memory.SnapshotReader(native_weights, capacity=2)
    history = np.concatenate([paired_history(), paired_history()[:, ::-1]], axis=0)
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 17, 4], [3, 18, 4], [3, 31, 4]], np.int32),
                              (2, 2, 4, 3)).copy()
    actual = reader.predict(history, queries)
    assert tuple(actual) == memory.ARMS
    independent = [reader.predict(history[i:i + 1], queries[i:i + 1]) for i in range(2)]
    reordered = reader.predict(history, queries[..., ::-1, :].copy())
    repeated = reader.predict(history, queries)
    for arm in memory.ARMS:
        assert actual[arm].shape == (2, 2, 4)
        assert actual[arm].dtype == np.int32
        np.testing.assert_array_equal(actual[arm], np.concatenate([row[arm] for row in independent]))
        np.testing.assert_array_equal(actual[arm], reordered[arm][..., ::-1])
        np.testing.assert_array_equal(actual[arm], repeated[arm])
        np.testing.assert_array_equal(actual[arm][1], actual[arm][0, ::-1])
        assert np.all((actual[arm] >= 32) & (actual[arm] < 48))
    # A missing key uses each lane's live state in both cache controls.
    for arm in ('fifo', 'keyed'):
        np.testing.assert_array_equal(actual[arm][..., -1], actual['live'][..., -1])


def test_native_guarded_matches_four_arm_and_independent_visible_prefix_reads(memory, native_weights):
    reader = memory.SnapshotReader(native_weights, capacity=2)
    history = np.ones((1, 2, 31), np.int32)
    for world, keys in enumerate(([16, 18, 16, 17, 17], [17, 18, 17, 16, 16])):
        for i, key in enumerate(keys):
            start = world + 4 * i
            history[0, world, start:start + 3] = [2, key, 32 + world * 5 + i]
    # Neither a malformed event nor an incomplete STORE advances recency.
    history[..., 22:25] = [2, 19, 48]
    history[..., -2:] = [2, 16]
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 17, 4], [3, 18, 4], [3, 31, 4]], np.int32),
                              (1, 2, 4, 3)).copy()
    legacy = reader.predict(history, queries)
    comparative = reader.predict(history, queries, include_guard=True)
    standalone = reader.predict_guarded(history, queries)
    assert tuple(legacy) == memory.ARMS == ('live', 'fifo', 'keyed')
    assert tuple(comparative) == (*memory.ARMS, 'guarded')
    assert standalone.shape == (1, 2, 4) and standalone.dtype == np.int32
    for arm in memory.ARMS:
        np.testing.assert_array_equal(comparative[arm], legacy[arm])
    np.testing.assert_array_equal(standalone, comparative['guarded'])
    np.testing.assert_array_equal(
        reader.predict_guarded(history, queries[..., ::-1, :].copy()), standalone[..., ::-1],
    )
    combined = reader.predict_guarded(
        np.concatenate([history, history[:, ::-1]], axis=0),
        np.concatenate([queries, queries[:, ::-1]], axis=0),
    )
    np.testing.assert_array_equal(combined, np.concatenate([standalone, standalone[:, ::-1]], axis=0))

    live, recent, keyed = reader._prefill(history.reshape(2, -1), guarded=True)
    recent_hits = np.asarray(memory._recent_hits(recent, jnp.asarray(queries.reshape(2, 4, 3)[:, :, 1])))
    np.testing.assert_array_equal(recent_hits, [[False, True, False, False], [True, False, False, False]])
    np.testing.assert_array_equal(
        comparative['guarded'], np.where(recent_hits[None], legacy['live'], legacy['keyed']),
    )
    routed = memory._fork_guarded_states(live, recent, keyed, jnp.asarray(queries[0, :, :, 1]))
    for world, tokens in enumerate(history[0]):
        visible_recent, visible_keyed = reference_endpoints(tokens, reader.capacity)
        for question, query in enumerate(queries[0, world]):
            key = int(query[1])
            end = len(tokens) if key in visible_recent else visible_keyed.get(key, len(tokens))
            _, expected_state = native_forward(
                reader._model, jnp.asarray(tokens[None, :end]), initial(reader),
            )
            row = world * queries.shape[2] + question
            selected = jax.tree.map(lambda x: x[row:row + 1], routed)
            assert_tree_close(selected, expected_state)
            selected_logits, _ = native_forward(reader._model, jnp.asarray(query[None]), selected)
            visible_logits, _ = native_forward(
                reader._model, jnp.asarray(np.concatenate([tokens[:end], query])[None]), initial(reader),
            )
            np.testing.assert_allclose(selected_logits, visible_logits, rtol=3e-5, atol=3e-5)
            assert standalone[0, world, question] == np.asarray(memory._predictions(visible_logits))[0]


def test_standalone_guard_never_allocates_fifo_payload_or_evaluates_comparative_heads(
        memory, native_weights, monkeypatch):
    reader = memory.SnapshotReader(native_weights, capacity=2)
    history = paired_history()
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 18, 4], [3, 31, 4]], np.int32),
                              (1, 2, 3, 3)).copy()
    expected = reader.predict(history, queries, include_guard=True)['guarded']

    def forbidden(*args, **kwargs):
        pytest.fail('standalone guarded reads must not use FIFO state banks or comparative heads')

    monkeypatch.setattr(memory, '_empty_banks', forbidden)
    monkeypatch.setattr(memory, '_read_queries', forbidden)
    actual = reader.predict_guarded(history, queries)
    np.testing.assert_array_equal(actual, expected)
    _, recent, keyed = reader._prefill(history.reshape(2, -1), guarded=True)
    assert recent._fields == ('keys', 'cursor')
    assert memory._state_bytes(keyed.states) == 2 * reader.capacity * reader.state_bytes


@pytest.mark.parametrize('mode', ['guarded', 'fresh', 'replayed'])
def test_standalone_reads_preserve_finite_raw_argmax_and_nonfinite_sentinel(memory, native_weights, mode):
    reader = memory.SnapshotReader(native_weights, capacity=3)
    history = replay_history()
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 17, 4], [3, 18, 4]], np.int32),
                              (1, 2, 3, 3)).copy()
    predict = getattr(reader, f'predict_{mode}')
    reader._model.lm_head.kernel[...] = jnp.zeros_like(reader._model.lm_head.kernel[...])
    np.testing.assert_array_equal(predict(history, queries), np.zeros((1, 2, 3), np.int32))
    reader._model.lm_head.kernel[...] = jnp.full_like(reader._model.lm_head.kernel[...], jnp.nan)
    np.testing.assert_array_equal(predict(history, queries), np.full((1, 2, 3), -1, np.int32))


def test_native_fresh_shares_prefill_and_matches_independent_visible_reads(
        memory, native_weights, monkeypatch):
    reader = memory.SnapshotReader(native_weights, capacity=3)
    history = replay_history()
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 17, 4], [3, 31, 4]], np.int32),
                              (1, 2, 3, 3)).copy()
    prefill = reader._prefill_fresh
    seen = []

    def shared(tokens):
        seen.append(tokens.shape)
        return prefill(tokens)

    def forbidden(*args, **kwargs):
        pytest.fail('fresh reads must not create snapshot banks')

    monkeypatch.setattr(reader, '_prefill_fresh', shared)
    monkeypatch.setattr(reader, '_prefill', forbidden)
    monkeypatch.setattr(memory, '_empty_banks', forbidden)
    monkeypatch.setattr(memory, '_empty_guarded_banks', forbidden)
    for width in (1, 24, 64, 103):
        visible = history[..., -width:]
        actual = reader.predict_fresh(visible, queries)
        expanded = np.broadcast_to(visible[:, :, None, :], (*queries.shape[:3], width))
        ids = np.concatenate([expanded, queries], axis=-1).reshape(-1, width + 3)
        logits, _ = native_forward(reader._model, jnp.asarray(ids), initial(reader, ids.shape[0]))
        assert np.isfinite(logits).all()
        assert actual.shape == (1, 2, 3) and actual.dtype == np.int32
        np.testing.assert_array_equal(actual, np.asarray(logits).argmax(axis=-1).reshape(1, 2, 3))
    assert seen == [(2, 1), (2, 24), (2, 64), (2, 103)]


def test_native_replayed_selects_true_suffix_or_original_guarded_states_before_reading(
        memory, native_weights, monkeypatch):
    reader = memory.SnapshotReader(native_weights, capacity=3)
    history = replay_history()
    queries = np.broadcast_to(np.array([[3, key, 4] for key in (16, 17, 18, 20, 31)], np.int32),
                              (1, 2, 5, 3)).copy()
    mask = memory.replay_mask(history, queries, reader.capacity)
    np.testing.assert_array_equal(mask, [[[False, True, False, False, False],
                                          [True, False, False, False, False]]])
    prefill = reader._prefill_fresh
    read = memory._read_replayed_queries
    local_shapes, read_shapes = [], []

    def local_prefill(tokens):
        local_shapes.append(tokens.shape)
        np.testing.assert_array_equal(tokens, history.reshape(2, -1)[:, -64:])
        return prefill(tokens)

    def selected_read(model, flat_queries, live, recent, keyed, fresh, replayed):
        read_shapes.append(flat_queries.shape)
        np.testing.assert_array_equal(replayed, mask.reshape(2, 5))
        return read(model, flat_queries, live, recent, keyed, fresh, replayed)

    def forbidden(*args, **kwargs):
        pytest.fail('mixed replay must select states instead of scoring unused query heads')

    with monkeypatch.context() as patch:
        patch.setattr(reader, '_prefill_fresh', local_prefill)
        patch.setattr(memory, '_read_replayed_queries', selected_read)
        patch.setattr(memory, '_read_guarded_queries', forbidden)
        patch.setattr(memory, '_read_fresh_queries', forbidden)
        patch.setattr(memory, '_read_queries', forbidden)
        actual = reader.predict_replayed(history, queries)
    assert local_shapes == [(2, 64)]
    assert read_shapes == [(2, 5, 3)]
    guarded = reader.predict_guarded(history, queries)
    fresh_predictions = reader.predict_fresh(history[..., -64:], queries)
    np.testing.assert_array_equal(actual, np.where(mask, fresh_predictions, guarded))
    live, recent, keyed = reader._prefill(history.reshape(2, -1), guarded=True)
    fresh = reader._prefill_fresh(history.reshape(2, -1)[:, -64:])
    routed = memory._fork_replayed_states(
        live, recent, keyed, fresh, jnp.asarray(queries[0, :, :, 1]), jnp.asarray(mask[0]),
    )
    for world, tokens in enumerate(history[0]):
        visible_recent, visible_keyed = reference_endpoints(tokens, reader.capacity)
        for question, query in enumerate(queries[0, world]):
            key = int(query[1])
            end = len(tokens) if key in visible_recent else visible_keyed.get(key, len(tokens))
            visible = tokens[-64:] if mask[0, world, question] else tokens[:end]
            _, expected_state = native_forward(reader._model, jnp.asarray(visible[None]), initial(reader))
            row = world * queries.shape[2] + question
            selected = jax.tree.map(lambda x: x[row:row + 1], routed)
            assert_tree_close(selected, expected_state)
            selected_logits, _ = native_forward(reader._model, jnp.asarray(query[None]), selected)
            expected_logits, _ = native_forward(
                reader._model, jnp.asarray(np.concatenate([visible, query])[None]), initial(reader),
            )
            np.testing.assert_allclose(selected_logits, expected_logits, rtol=3e-5, atol=3e-5)
            assert actual[0, world, question] == np.asarray(memory._predictions(expected_logits))[0]


def test_native_replayed_is_independent_across_families_worlds_queries_and_calls(memory, native_weights):
    reader = memory.SnapshotReader(native_weights, capacity=3)
    history = np.concatenate([replay_history(), replay_history()[:, ::-1]], axis=0)
    queries = np.broadcast_to(np.array([[3, key, 4] for key in (16, 17, 18, 31)], np.int32),
                              (2, 2, 4, 3)).copy()
    actual = reader.predict_replayed(history, queries)
    assert actual.shape == (2, 2, 4) and actual.dtype == np.int32
    independent = [reader.predict_replayed(history[i:i + 1], queries[i:i + 1]) for i in range(2)]
    np.testing.assert_array_equal(actual, np.concatenate(independent))
    np.testing.assert_array_equal(actual[1], actual[0, ::-1])
    np.testing.assert_array_equal(
        reader.predict_replayed(history, queries[..., ::-1, :].copy()), actual[..., ::-1],
    )
    np.testing.assert_array_equal(reader.predict_replayed(history, queries), actual)


def test_replayed_misses_and_short_histories_skip_all_local_prefill_work(memory, native_weights, monkeypatch):
    reader = memory.SnapshotReader(native_weights, capacity=3)
    queries = np.broadcast_to(np.array([[3, 16, 4], [3, 31, 4]], np.int32), (1, 2, 2, 3))
    histories = [np.ones((1, 2, length), np.int32) for length in (0, 2, 64, 103)]
    for history in histories:
        if history.shape[-1] >= 3:
            history[..., -3:] = [2, 16 if history.shape[-1] <= 64 else 17, 32]
    expected = [reader.predict_guarded(history, queries) for history in histories]

    def forbidden(*args, **kwargs):
        pytest.fail('replay misses and short histories must not prefill or read a reset state')

    monkeypatch.setattr(reader, '_prefill_fresh', forbidden)
    monkeypatch.setattr(memory, '_read_fresh_queries', forbidden)
    monkeypatch.setattr(memory, '_read_replayed_queries', forbidden)
    for history, guarded in zip(histories, expected, strict=True):
        np.testing.assert_array_equal(reader.predict_replayed(history, queries), guarded)


def test_replayed_all_hits_skip_global_prefill_and_snapshot_banks(memory, native_weights, monkeypatch):
    reader = memory.SnapshotReader(native_weights, capacity=3)
    history = replay_history()
    queries = np.array([[[[3, 17, 4], [3, 17, 4]], [[3, 16, 4], [3, 16, 4]]]], np.int32)
    assert memory.replay_mask(history, queries, reader.capacity).all()
    expected = reader.predict_fresh(history[..., -64:], queries)

    def forbidden(*args, **kwargs):
        pytest.fail('an all-hit replay must not compute unused global guarded states')

    monkeypatch.setattr(reader, '_prefill', forbidden)
    monkeypatch.setattr(memory, '_empty_guarded_banks', forbidden)
    monkeypatch.setattr(memory, '_read_guarded_queries', forbidden)
    monkeypatch.setattr(memory, '_read_replayed_queries', forbidden)
    np.testing.assert_array_equal(reader.predict_replayed(history, queries), expected)


def test_fresh_rejects_empty_history_before_prefill(memory, native_weights, monkeypatch):
    reader = memory.SnapshotReader(native_weights)
    queries = np.broadcast_to(np.array([[[[3, 16, 4]]]], np.int32), (1, 2, 1, 3))

    def forbidden(*args, **kwargs):
        pytest.fail('empty fresh histories must fail before neural work')

    monkeypatch.setattr(reader, '_prefill_fresh', forbidden)
    with pytest.raises(ValueError, match='nonempty'):
        reader.predict_fresh(np.ones((1, 2, 0), np.int32), queries)


def test_empty_and_no_write_histories_use_live_fallback(memory, native_weights):
    reader = memory.SnapshotReader(native_weights, capacity=1)
    queries = np.broadcast_to(np.array([[[[3, 16, 4]]]], np.int32), (1, 2, 1, 3))
    for length in (0, 2, 137):
        result = reader.predict(np.ones((1, 2, length), np.int32), queries)
        for arm in ('fifo', 'keyed'):
            np.testing.assert_array_equal(result[arm], result['live'])
        comparative = reader.predict(np.ones((1, 2, length), np.int32), queries, include_guard=True)
        standalone = reader.predict_guarded(np.ones((1, 2, length), np.int32), queries)
        np.testing.assert_array_equal(comparative['guarded'], result['live'])
        np.testing.assert_array_equal(standalone, result['live'])


@pytest.mark.parametrize('capacity', [0, -1, 1.5, True])
@pytest.mark.parametrize('mode', ['reader', 'mask'])
def test_invalid_capacity_is_rejected_before_loading(memory, capacity, mode):
    with pytest.raises(ValueError, match='positive integer'):
        if mode == 'reader':
            memory.SnapshotReader(Path('not-a-checkpoint'), capacity)
        else:
            queries = np.broadcast_to(np.array([[[[3, 16, 4]]]], np.int32), (1, 2, 1, 3))
            memory.replay_mask(np.ones((1, 2, 70), np.int32), queries, capacity)


@pytest.mark.parametrize('mode', ['legacy', 'four_arm', 'standalone', 'fresh', 'replayed', 'mask'])
def test_raw_input_contract_rejects_labels_and_malformed_queries(memory, native_weights, mode):
    reader = memory.SnapshotReader(native_weights)
    history = paired_history()
    queries = np.broadcast_to(np.array([[[[3, 16, 4]]]], np.int32), (1, 2, 1, 3)).copy()
    if mode == 'standalone':
        predict = reader.predict_guarded
    elif mode in ('fresh', 'replayed'):
        predict = getattr(reader, f'predict_{mode}')
    elif mode == 'mask':
        def predict(history, queries):
            return memory.replay_mask(history, queries, capacity=16)
    else:
        def predict(history, queries):
            return reader.predict(history, queries, include_guard=(mode == 'four_arm'))
    with pytest.raises(ValueError, match='int32'):
        predict(history.astype(np.float32), queries)
    with pytest.raises(ValueError, match='question,3'):
        predict(history, np.concatenate([queries, np.full((1, 2, 1, 1), 32, np.int32)], axis=-1))
    for position, token in ((0, 2), (1, 32), (2, 32)):
        invalid = queries.copy()
        invalid[..., position] = token
        with pytest.raises(ValueError, match='raw'):
            predict(history, invalid)
    invalid_history = history.copy()
    invalid_history[0, 0, 0] = 128
    with pytest.raises(ValueError, match='out-of-vocabulary'):
        predict(invalid_history, queries)
