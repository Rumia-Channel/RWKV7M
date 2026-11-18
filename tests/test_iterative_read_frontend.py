import csv
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import iterative_read_frontend as frontend
from addressed_rwkv_reader import AddressedReader, FrozenBindingCodec
from benchmark_address_screening import checkpoints
from binding_state_memory import _read_fresh_queries
from iterative_read_tasks import TEST_RULES, route_labels
from probe_binding_stress import _advance_once
from route_transfer_panel import COUNT, KEY_BASE, RULES, VALUE_BASE, make_panel, questions
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, QUERY_TOKEN, STORE_TOKEN
from rwkv7m.model.state import init_rwkv_state, init_screen_state
from versioned_binding_panel import literal_reference


DRAW = 743101


def _flat(public):
    return public.history.reshape(-1, public.history.shape[-1])


def test_heldout_spellings_bind_writes_questions_and_demo_starts():
    panel = make_panel(DRAW, 8, families=2)
    original = panel.alias.copy()
    train = frontend.render_panel(panel)
    heldout = frontend.render_panel(panel, heldout=True)
    for public, low in ((train, 64), (heldout, 80)):
        history = _flat(public)
        legend = history[:, :4 * COUNT].reshape(-1, COUNT, 4)
        np.testing.assert_array_equal(np.sort(legend[:, :, 3], axis=-1),
                                      np.broadcast_to(np.arange(low, low + COUNT), (4, COUNT)))
        for row in history:
            starts = np.flatnonzero(row[4 * COUNT:] == STORE_TOKEN) + 4 * COUNT
            assert np.all((row[starts + 1] >= low) & (row[starts + 1] < low + COUNT))
        assert np.all((public.query >= low) & (public.query < low + COUNT))
        assert np.all((public.demo >= low) & (public.demo < low + COUNT))
        resolved_query = frontend.resolve_names(history, public.query.reshape(4, 7))
        resolved_demo = frontend.resolve_names(history, public.demo.reshape(4, 3))
        np.testing.assert_array_equal(resolved_query, panel.queries[..., 1].reshape(4, 7) - KEY_BASE)
        np.testing.assert_array_equal(resolved_demo, panel.examples[0, ..., 1].reshape(4, 3) - KEY_BASE)
        np.testing.assert_array_equal(frontend.known_vocabulary(history)[:, 4 * COUNT:],
                                      panel.known.reshape(4, -1)[:, 4 * COUNT:])
    # Rendering owns its remapped storage and leaves the original panel unchanged.
    np.testing.assert_array_equal(panel.alias, original)
    assert frontend._public_digest(train.history, train.demo, train.query) != frontend._public_digest(
        heldout.history, heldout.demo, heldout.query)
    with pytest.raises(ValueError, match='unknown'):
        frontend.resolve_names(_flat(heldout), np.full((4, 1), 64, np.int32))


def test_binding_uses_episode_declarations_not_alias_arithmetic():
    public = frontend.render_panel(make_panel(DRAW, 0, families=1), heldout=True)
    history = _flat(public).copy()
    legend = history[:, :4 * COUNT].reshape(2, COUNT, 4)
    names = legend[:, :2, 3].copy()
    before = frontend.resolve_names(history, names)
    legend[:, :2, 3] = legend[:, 1::-1, 3].copy()
    after = frontend.resolve_names(history, names)
    np.testing.assert_array_equal(after, before[:, ::-1])
    # Canonical keys still resolve to the same station despite changed spellings.
    np.testing.assert_array_equal(frontend.resolve_names(history, before + KEY_BASE), before)


@pytest.mark.parametrize('damage', ('duplicate_alias', 'duplicate_key', 'wrong_value', 'wrong_marker'))
def test_duplicate_or_inconsistent_legend_is_rejected(damage):
    history = _flat(frontend.render_panel(make_panel(DRAW, 0, families=1))).copy()
    legend = history[:, :4 * COUNT].reshape(2, COUNT, 4)
    if damage == 'duplicate_alias':
        legend[0, 0, 3] = legend[0, 1, 3]
    elif damage == 'duplicate_key':
        legend[0, 0, 1:3] = legend[0, 1, 1:3]
    elif damage == 'wrong_value':
        legend[0, 0, 2] = VALUE_BASE + (legend[0, 0, 2] - VALUE_BASE + 1) % COUNT
    else:
        legend[0, 0, 0] = 7
    with pytest.raises(ValueError, match='legend'):
        frontend.resolve_names(history, np.full((2, 1), KEY_BASE, np.int32))
    with pytest.raises(ValueError, match='legend'):
        frontend.known_vocabulary(history)


def test_unknown_source_alias_is_rejected_in_names_and_writes():
    history = _flat(frontend.render_panel(make_panel(DRAW, 0, families=1), heldout=True)).copy()
    with pytest.raises(ValueError, match='unknown'):
        frontend.resolve_names(history, np.full((2, 1), 127, np.int32))
    history[0, 4 * COUNT + 1] = 127
    with pytest.raises(ValueError, match='unknown'):
        frontend.known_vocabulary(history)
    # Invalid public input is rejected before either model could be consulted.
    with pytest.raises(ValueError, match='unknown'):
        frontend.observe(None, None, history)


def test_incomplete_writes_and_non_grammar_suffixes_are_not_silently_dropped():
    history = _flat(frontend.render_panel(make_panel(DRAW, 0, families=1)))
    with pytest.raises(ValueError, match='incomplete STORE'):
        frontend.known_vocabulary(history[:, :-1])
    unknown = history.copy()
    unknown[0, 4 * COUNT] = 100
    with pytest.raises(ValueError, match='STORE triples and filler'):
        frontend.known_vocabulary(unknown)


def test_rendering_contains_no_demo_answers_or_test_answer_appendix():
    panel = make_panel(DRAW, 0, families=2)
    rendered = frontend.render_panel(panel, heldout=True)
    answers_changed = panel.examples.copy()
    answers_changed[..., 3] = 127
    changed = panel._replace(examples=answers_changed, gold=np.full_like(panel.gold, 126))
    other = frontend.render_panel(changed, heldout=True)
    for actual, expected in zip(other, rendered, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert rendered.history.shape == panel.alias.shape
    assert rendered.query.shape == (2, 2, 7)
    assert rendered.demo.shape == (2, 2, 3)
    assert not np.isin(rendered.history, (QUERY_TOKEN, ANSWER_MARKER_TOKEN)).any()
    query = frontend.resolve_names(_flat(rendered), rendered.query.reshape(4, 7))
    demo = frontend.resolve_names(_flat(rendered), rendered.demo.reshape(4, 3))
    assert not np.any(demo[:, :, None] == query[:, None, :])


def test_independent_visible_truth_checks_all_four_old_tasks_and_paired_worlds():
    panel = make_panel(DRAW, 128, families=2)
    public = frontend.render_panel(panel, heldout=True)
    truth, query, demo = frontend._validated_arrays(panel, public)
    keys = np.broadcast_to(KEY_BASE + np.arange(COUNT, dtype=np.int32), (4, COUNT))
    visible = literal_reference(frontend.known_vocabulary(_flat(public)), keys)
    np.testing.assert_array_equal(truth.reshape(4, COUNT), visible - VALUE_BASE)
    for index, rule in enumerate(RULES):
        np.testing.assert_array_equal(route_labels(truth, query, TEST_RULES[rule]) + VALUE_BASE,
                                      panel.gold[index])
        np.testing.assert_array_equal(route_labels(truth, demo, TEST_RULES[rule]) + VALUE_BASE,
                                      panel.examples[index, ..., 3])
        assert np.all(panel.gold[index, :, 0] != panel.gold[index, :, 1])
    wrong = panel.gold.copy()
    wrong[0, 0, 0, 0] = VALUE_BASE + (wrong[0, 0, 0, 0] - VALUE_BASE + 1) % COUNT
    with pytest.raises(ValueError, match='visible route labels disagree'):
        frontend._validated_arrays(panel._replace(gold=wrong), public)
    wrong_example = panel.examples.copy()
    wrong_example[0, 0, 0, 0, 3] = VALUE_BASE + (
        wrong_example[0, 0, 0, 0, 3] - VALUE_BASE + 1) % COUNT
    with pytest.raises(ValueError, match='visible route labels disagree'):
        frontend._validated_arrays(panel._replace(examples=wrong_example), public)


def test_unrestricted_head_token_encoding_does_not_select_among_valid_values():
    tokens = np.array([-2, -1, 0, 16, 31, 32, 33, 47, 48, 64, 127], dtype=np.int32)
    np.testing.assert_array_equal(frontend.head_to_edges(tokens),
                                  np.array([16, 16, 16, 16, 16, 0, 1, 15, 16, 16, 16], np.int32))


@pytest.fixture(scope='module')
def frozen_codec():
    seed, checkpoint, _ = next(checkpoints())
    assert seed == 42
    return FrozenBindingCodec(checkpoint)


def test_observe_matches_real_frozen_readers_and_bounds_recurrent_chunks(frozen_codec, monkeypatch):
    public = frontend.render_panel(make_panel(DRAW, 0, families=1), heldout=True)
    history = _flat(public)
    normalized = frontend.known_vocabulary(history)
    reader = AddressedReader(frozen_codec, 'isolated')
    lengths = []

    def recorded_advance(model, tokens, state):
        lengths.append(tokens.shape[1])
        return _advance_once(model, tokens, state)

    monkeypatch.setattr(frontend, '_advance_once', recorded_advance)
    actual = frontend.observe(frozen_codec, reader, history)
    assert all(0 < length <= 64 for length in lengths)
    assert sum(lengths) == history.shape[1]
    queries = questions(np.broadcast_to(np.arange(COUNT, dtype=np.int32), (2, COUNT)))
    model = frozen_codec.model
    state = (init_rwkv_state(2, model.config), init_screen_state(2, model.config.screening))
    for start in range(0, normalized.shape[1], 64):
        state = _advance_once(model, jnp.asarray(normalized[:, start:start + 64]), state)
    ordinary_tokens = np.asarray(_read_fresh_queries(model, jnp.asarray(queries), state))
    isolated_tokens = reader.query(reader.advance(reader.start(2), normalized), queries)
    for name, tokens in (('ordinary', ordinary_tokens), ('isolated', isolated_tokens)):
        assert actual[name].shape == (2, COUNT)
        assert actual[name].dtype == np.int32
        np.testing.assert_array_equal(actual[name], frontend.head_to_edges(tokens))
    isolated_only = frontend.observe(frozen_codec, reader, history, frontends=('isolated',))
    assert tuple(isolated_only) == ('isolated',)
    np.testing.assert_array_equal(isolated_only['isolated'], actual['isolated'])


def test_real_cache_contract_costs_digest_and_no_silent_overwrite(tmp_path, monkeypatch):
    # Keep this integration test small without substituting any model prediction.
    monkeypatch.setitem(frontend.SPLITS, 'train', ((741101,), 2, (0, 8)))
    outputs = frontend.prepare_cache(tmp_path, 'train', seed=42)
    assert outputs == [tmp_path / 'train-s42.npz']
    with np.load(outputs[0], allow_pickle=False) as saved:
        assert set(saved.files) == set(frontend.CACHE_KEYS)
        for name in ('ordinary', 'isolated', 'truth'):
            assert saved[name].shape == (4, 2, COUNT)
            assert saved[name].dtype == np.int32
        for name in ('ordinary', 'isolated'):
            assert np.all((saved[name] >= 0) & (saved[name] <= COUNT))
        assert saved['query'].shape == (4, 2, 7)
        assert saved['demo'].shape == (4, 2, 3)
        for name in ('query', 'demo', 'draw', 'family', 'gap', 'weight_seed'):
            assert saved[name].dtype == np.int32
        assert saved['weight_seed'].shape == () and int(saved['weight_seed']) == 42
        assert saved['weight_sha'].shape == () and len(str(saved['weight_sha'])) == 64
        assert saved['digest'].dtype.kind == 'U'
        np.testing.assert_array_equal(saved['family'], [0, 1, 0, 1])
        np.testing.assert_array_equal(saved['gap'], [0, 0, 8, 8])
        for offset, gap in ((0, 0), (2, 8)):
            public = frontend.render_panel(make_panel(741101, gap, families=2))
            truth, query, demo = frontend._validated_arrays(make_panel(741101, gap, families=2), public)
            for name, expected in (('truth', truth), ('query', query), ('demo', demo)):
                np.testing.assert_array_equal(saved[name][offset:offset + 2], expected)
            for family in range(2):
                expected = frontend._public_digest(public.history[family], public.demo[family],
                                                   public.query[family])
                assert saved['digest'][offset + family] == expected
    with (tmp_path / 'train-s42-cost.csv').open(encoding='utf8', newline='') as stream:
        costs = list(csv.DictReader(stream))
    assert len(costs) == 4
    assert frontend.FAMILY_BATCH <= 8
    for row in costs:
        assert 1 <= int(row['families']) <= 8
        assert int(row['worlds']) == 2
        assert float(row['seconds']) >= 0 and float(row['codec_seconds']) >= 0
        assert len(row['source_sha256']) == len(row['input_sha256']) == 64
        assert 'scripts/iterative_read_frontend.py:' in row['source_hashes']
        assert int(row['head_vectors']) == 2 * int(row['families']) * COUNT
        if row['frontend'] == 'ordinary':
            assert int(row['recurrent_tokens']) == int(row['history_tokens']) + int(row['query_tokens'])
            assert int(row['parser_tokens']) == 0
        else:
            assert int(row['recurrent_tokens']) == 0
            assert int(row['parser_tokens']) == int(row['history_tokens'])
    original = outputs[0].read_bytes()
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        frontend.prepare_cache(tmp_path, 'train', seed=42)
    assert outputs[0].read_bytes() == original
    assert not list(tmp_path.glob('dev-*')) and not list(tmp_path.glob('test-*'))


def test_cli_requires_an_explicit_split_before_any_model_run(tmp_path):
    with pytest.raises(SystemExit):
        frontend.main([str(tmp_path)])
    assert not list(tmp_path.iterdir())
