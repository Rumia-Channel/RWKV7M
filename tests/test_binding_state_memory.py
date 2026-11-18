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


def test_empty_and_no_write_histories_use_live_fallback(memory, native_weights):
    reader = memory.SnapshotReader(native_weights, capacity=1)
    queries = np.broadcast_to(np.array([[[[3, 16, 4]]]], np.int32), (1, 2, 1, 3))
    for length in (0, 2, 137):
        result = reader.predict(np.ones((1, 2, length), np.int32), queries)
        for arm in ('fifo', 'keyed'):
            np.testing.assert_array_equal(result[arm], result['live'])


@pytest.mark.parametrize('capacity', [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected_before_loading(memory, capacity):
    with pytest.raises(ValueError, match='positive integer'):
        memory.SnapshotReader(Path('not-a-checkpoint'), capacity)


def test_raw_input_contract_rejects_labels_and_malformed_queries(memory, native_weights):
    reader = memory.SnapshotReader(native_weights)
    history = paired_history()
    queries = np.broadcast_to(np.array([[[[3, 16, 4]]]], np.int32), (1, 2, 1, 3)).copy()
    with pytest.raises(ValueError, match='int32'):
        reader.predict(history.astype(np.float32), queries)
    with pytest.raises(ValueError, match='question,3'):
        reader.predict(history, np.concatenate([queries, np.full((1, 2, 1, 1), 32, np.int32)], axis=-1))
    for position, token in ((0, 2), (1, 32), (2, 32)):
        invalid = queries.copy()
        invalid[..., position] = token
        with pytest.raises(ValueError, match='raw'):
            reader.predict(history, invalid)
    invalid_history = history.copy()
    invalid_history[0, 0, 0] = 128
    with pytest.raises(ValueError, match='out-of-vocabulary'):
        reader.predict(invalid_history, queries)
