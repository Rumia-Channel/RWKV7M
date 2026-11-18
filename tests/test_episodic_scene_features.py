import csv
import json
from hashlib import sha256
from pathlib import Path
import sys

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import episodic_scene_features as frontend
from benchmark_address_screening import checkpoints
from rwkv7m.model.state import init_rwkv_state, init_screen_state


@pytest.fixture(scope='module')
def encoder():
    seed, checkpoint, digest = next(checkpoints())
    frozen = frontend.WindowEncoder(checkpoint)
    assert seed == frozen.weight_seed == 42
    assert frozen.weight_sha == digest
    return frozen


@nnx.jit
def _independent_hidden(model, tokens):
    state = (init_rwkv_state(len(tokens), model.config),
             init_screen_state(len(tokens), model.config.screening))
    hidden, _, _, _ = model.compute_recurrent_hidden(
        tokens, *state, phase='read_write', deterministic=True)
    return hidden


def _independent_features(model, tokens, query_length):
    # Cross the device/host boundary before pooling: reference values are the
    # actually returned hidden states, not a compiler-fused approximation.
    last = np.asarray(_independent_hidden(model, tokens), dtype=np.float32)[:, -query_length:]
    return np.concatenate((last.mean(axis=1), last[:, -1]), axis=-1)


def test_local_is_zero_state_mean_and_last_true_hidden(encoder):
    tokens = np.random.default_rng(2271).integers(0, 128, (3, 8), dtype=np.int32)
    before = tokens.copy()
    actual = encoder.local(tokens)
    expected = np.asarray(_independent_features(encoder.model, jnp.asarray(tokens), 8))
    assert actual.shape == (3, 256) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(tokens, before)
    # Each utterance has its own zero state, even after unrelated windows ran.
    reversed_features = encoder.local(tokens[::-1].copy())
    np.testing.assert_array_equal(actual, reversed_features[::-1])


def test_ordinary_matches_full_context_and_isolates_query_branches(encoder):
    rng = np.random.default_rng(2272)
    # Cross a real prefix chunk boundary; an independent pass never chunks.
    blocks = rng.integers(0, 128, (1, 18, 8), dtype=np.int32)
    queries = rng.integers(0, 128, (1, 2, 8), dtype=np.int32)
    original_blocks, original_queries = blocks.copy(), queries.copy()
    before = dict(encoder.cost)
    actual = encoder.ordinary(blocks, queries)
    expected = []
    for question in range(2):
        complete = np.concatenate((blocks.reshape(1, -1), queries[:, question]), axis=1)
        expected.append(np.asarray(_independent_features(encoder.model, jnp.asarray(complete), 8)))
    expected = np.stack(expected, axis=1)
    assert actual.shape == (1, 2, 256) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
    assert encoder.cost['ordinary_prefixes'] - before['ordinary_prefixes'] == 1
    assert encoder.cost['ordinary_prefix_lanes'] - before['ordinary_prefix_lanes'] == frontend.ORDINARY_BATCH
    assert encoder.cost['ordinary_prefix_tokens'] - before['ordinary_prefix_tokens'] == 18 * 8
    assert encoder.cost['prefix_chunk_calls'] - before['prefix_chunk_calls'] == 2
    changed = queries.copy()
    changed[:, 0] = (changed[:, 0] + 23) % 128
    before = dict(encoder.cost)
    forked = encoder.ordinary_many(blocks, np.stack((queries, changed)))
    np.testing.assert_allclose(forked[0], actual, rtol=4e-5, atol=4e-5)
    np.testing.assert_array_equal(forked[0, :, 1], forked[1, :, 1])
    # Two surfaces still pay for only one physical prefix per true world.
    assert encoder.cost['ordinary_prefixes'] - before['ordinary_prefixes'] == 1
    assert encoder.cost['ordinary_queries'] - before['ordinary_queries'] == 4
    np.testing.assert_array_equal(blocks, original_blocks)
    np.testing.assert_array_equal(queries, original_queries)


@pytest.mark.parametrize('bad', [np.zeros((1, 8), np.int64), np.zeros((1, 7), np.int32),
                                 np.full((1, 8), 128, np.int32), np.full((1, 8), -1, np.int32)])
def test_local_rejects_nonpublic_input_before_model_access(bad):
    uninitialized = frontend.WindowEncoder.__new__(frontend.WindowEncoder)
    with pytest.raises(ValueError, match='int32|0..127'):
        uninitialized.local(bad)


def test_ordinary_rejects_world_or_question_axis_confusion_before_model_access():
    uninitialized = frontend.WindowEncoder.__new__(frontend.WindowEncoder)
    with pytest.raises(ValueError, match='two independent queries'):
        uninitialized.ordinary(np.zeros((2, 3, 8), np.int32), np.zeros((2, 1, 8), np.int32))
    with pytest.raises(ValueError, match='two independent queries'):
        uninitialized.ordinary(np.zeros((2, 3, 8), np.int32), np.zeros((1, 2, 8), np.int32))


def test_checkpoint_authentication_uses_registered_sha_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'ce.safetensors'
    checkpoint.write_bytes(b'not a model: authentication precedes deserialization')
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(frontend, 'checkpoints', lambda: iter(((42, checkpoint, digest),)))
    assert frontend._authenticated_checkpoint(checkpoint) == (42, checkpoint.resolve(), digest)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed frozen CE checkpoint'):
        frontend.WindowEncoder(checkpoint)
    with pytest.raises(ValueError, match='SHA-authenticated CE registry'):
        frontend.WindowEncoder(tmp_path / 'unregistered.safetensors')


def test_public_digest_is_only_actual_raw_worlds_and_queries():
    blocks = np.arange(2 * 3 * 8, dtype=np.int32).reshape(2, 3, 8)
    queries = np.arange(16, dtype=np.int32).reshape(2, 8)
    expected = sha256(blocks.astype('<i4').tobytes() + queries.astype('<i4').tobytes()).hexdigest()
    assert frontend.public_sha(blocks, queries) == expected
    changed_blocks = blocks.copy()
    changed_blocks[1, 2, 7] += 1
    assert frontend.public_sha(changed_blocks, queries) != expected
    changed_queries = queries.copy()
    changed_queries[1, 7] += 1
    assert frontend.public_sha(blocks, changed_queries) != expected


def test_atomic_npz_publish_is_exclusive_and_never_leaves_a_partial_target(tmp_path):
    target = tmp_path / 'cache.npz'
    frontend._atomic_npz(target, {'value': np.asarray([1, 2], np.int32)})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        frontend._atomic_npz(target, {'value': np.asarray([9], np.int32)})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]

    def failing_writer(stream):
        stream.write(b'not a complete artifact')
        raise RuntimeError('interrupted write')

    failed = tmp_path / 'failed.npz'
    with pytest.raises(RuntimeError, match='interrupted'):
        frontend._atomic_publish(failed, failing_writer)
    assert not failed.exists()
    assert list(tmp_path.iterdir()) == [target]


def _small_real_panel_plan(monkeypatch):
    original_plan = frontend._split_plan

    def small_plan(split):
        draws, _, cells, surfaces = original_plan(split)
        return draws[:1], 2, cells[:1], surfaces

    monkeypatch.setattr(frontend, '_split_plan', small_plan)


def test_real_cache_raw_boundaries_content_dedup_provenance_and_resume(encoder, tmp_path, monkeypatch):
    _small_real_panel_plan(monkeypatch)
    # Reuse the same actual authenticated neural model, not fabricated outputs.
    monkeypatch.setattr(frontend, 'WindowEncoder', lambda checkpoint: encoder)
    outputs = frontend.make_cache(tmp_path, 'train', 42)
    target = tmp_path / 'train-s42-seen-k4-g0.npz'
    assert outputs == [target]
    raw = frontend.scene_data.make_panel('train', frontend.scene_data.TRAIN_DRAWS[0], 2, 4, 0)
    with np.load(target, allow_pickle=False) as saved:
        assert set(saved.files) == set(frontend.CACHE_KEYS)
        tokens = saved['feature_tokens']
        np.testing.assert_array_equal(tokens[saved['blocks']], raw.blocks)
        np.testing.assert_array_equal(tokens[saved['queries']], raw.queries)
        np.testing.assert_array_equal(tokens, np.unique(
            np.concatenate((raw.blocks.reshape(-1, 8), raw.queries.reshape(-1, 8))), axis=0))
        assert saved['features'].shape == (len(tokens), 256)
        assert saved['features'].dtype == saved['ordinary'].dtype == np.float32
        np.testing.assert_array_equal(saved['features'], encoder.local(tokens))
        assert saved['ordinary'].shape == (2, 2, 2, 256)
        assert np.isfinite(saved['ordinary']).all()
        for name, expected in (('support', raw.support), ('answers', raw.answers)):
            np.testing.assert_array_equal(saved[name], expected)
        for name in ('blocks', 'queries', 'support', 'answers', 'draw', 'family', 'weight_seed'):
            assert saved[name].dtype == np.int32
        assert saved['weight_seed'].shape == () and saved['weight_seed'].item() == 42
        assert saved['weight_sha'].shape == () and saved['weight_sha'].item() == encoder.weight_sha
        assert saved['weight_sha'].dtype.kind == saved['input_sha'].dtype.kind == 'U'
        for family in range(2):
            assert saved['input_sha'][family] == frontend.public_sha(raw.blocks[family], raw.queries[family])
        np.testing.assert_array_equal(saved['draw'],
                                      np.full(2, frontend.scene_data.TRAIN_DRAWS[0], np.int32))
        np.testing.assert_array_equal(saved['family'], np.arange(2, dtype=np.int32))
        cost = json.loads(saved['cost_json'].item())
        expected_metadata = {name: saved[name].item() for name in
                             ('split', 'surface', 'live', 'gap', 'weight_seed', 'weight_sha', 'format_version')}
        source_sha = saved['source_sha'].item()
        assert len(source_sha) == 64 and 'episodic_scene_data.py:' in saved['source_hashes'].item()
    assert cost['ordinary_prefixes'] == 4
    assert cost['ordinary_prefix_lanes'] == frontend.ORDINARY_BATCH
    assert cost['ordinary_prefix_tokens'] == 4 * 4 * 8
    assert cost['ordinary_prefix_lane_tokens'] == frontend.ORDINARY_BATCH * 4 * 8
    assert cost['ordinary_queries'] == 8 and cost['ordinary_query_tokens'] == 64
    assert cost['local_windows'] == len(tokens)
    assert cost['local_lane_windows'] % frontend.LOCAL_BATCH == 0
    assert cost['local_seconds'] >= 0 and cost['ordinary_seconds'] >= 0
    cost_path = tmp_path / 'train-s42-seen-k4-g0-cost.csv'
    with cost_path.open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{key: str(value) for key, value in cost.items()}]
    original, original_cost = target.read_bytes(), cost_path.read_bytes()
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        frontend.make_cache(tmp_path, 'train', 42)
    assert target.read_bytes() == original and cost_path.read_bytes() == original_cost
    assert not list(tmp_path.glob('dev-*')) and not list(tmp_path.glob('test-*'))

    def must_not_reload(checkpoint):
        raise AssertionError('validated resume must not reload or run a model')

    monkeypatch.setattr(frontend, 'WindowEncoder', must_not_reload)
    assert frontend.make_cache(tmp_path, 'train', 42, reuse_existing=True) == [target]
    cost_path.unlink()  # Simulate interruption between the two atomic publishes.
    assert frontend.make_cache(tmp_path, 'train', 42, reuse_existing=True) == [target]
    assert cost_path.read_bytes() == original_cost
    with pytest.raises(ValueError, match='source provenance'):
        frontend._reuse_cache(target, cost_path, expected_metadata, '0' * 64)


def test_held_and_shift_share_real_prefix_not_query_or_label_state(encoder, tmp_path, monkeypatch):
    _small_real_panel_plan(monkeypatch)
    monkeypatch.setattr(frontend, 'WindowEncoder', lambda checkpoint: encoder)
    original = encoder.ordinary_many
    calls = []

    def recorded(blocks, queries):
        calls.append((blocks.copy(), queries.copy()))
        return original(blocks, queries)

    monkeypatch.setattr(encoder, 'ordinary_many', recorded)
    outputs = frontend.make_cache(tmp_path, 'test', 42, surfaces=('held', 'shift'))
    assert len(outputs) == 2 and len(calls) == 1
    public = {}
    costs = {}
    for surface, path in zip(('held', 'shift'), outputs, strict=True):
        assert path.name == f'test-s42-{surface}-k4-g0.npz'
        with np.load(path, allow_pickle=False) as saved:
            tokens = saved['feature_tokens']
            public[surface] = (tokens[saved['blocks']], tokens[saved['queries']],
                               saved['support'].copy(), saved['answers'].copy(),
                               saved['input_sha'].copy())
            costs[surface] = json.loads(saved['cost_json'].item())
    held, shift = public['held'], public['shift']
    np.testing.assert_array_equal(held[0], shift[0])
    np.testing.assert_array_equal(held[2], shift[2])
    np.testing.assert_array_equal(held[3], shift[3])
    assert not np.array_equal(held[1], shift[1])
    assert np.all(held[4] != shift[4])
    np.testing.assert_array_equal(calls[0][0], held[0].reshape(4, 4, 8))
    for index, surface in enumerate(('held', 'shift')):
        expected = np.broadcast_to(public[surface][1][:, None], (2, 2, 2, 8)).reshape(4, 2, 8)
        np.testing.assert_array_equal(calls[0][1][index], expected)
    assert costs['held']['ordinary_prefixes'] == 4
    assert costs['held']['ordinary_prefix_lanes'] == frontend.ORDINARY_BATCH
    assert costs['shift']['ordinary_prefixes'] == costs['shift']['ordinary_prefix_lanes'] == 0
    assert costs['shift']['shared_prefix_from'] == 'held'
    assert costs['shift']['ordinary_seconds'] == 0.0
    for cost in costs.values():
        assert cost['ordinary_queries'] == 8
        assert cost['shared_prefix_surfaces'] == 'held+shift'
    assert not list(tmp_path.glob('test-s42-seen-*'))


def test_cli_routes_explicit_filters_without_expanding_split(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(frontend, 'make_cache', lambda *args, **kwargs: calls.append((args, kwargs)))
    frontend.main([str(tmp_path), '--split', 'test', '--seed', '43', '--surface', 'shift',
                   '--cell', '16', '128', '--reuse-existing'])
    assert calls == [((tmp_path, 'test', [43]),
                      dict(surfaces=['shift'], cells=[[16, 128]], reuse_existing=True))]
