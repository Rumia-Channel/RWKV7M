import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import role_binding_observations as observations


@pytest.mark.parametrize('split,surface,live,gap', [
    ('train', 'seen', 4, 8), ('dev', 'seen', 8, 0),
    ('test', 'held', 4, 128), ('test', 'shift', 16, 128),
])
def test_raw_table_preserves_both_worlds_support_flips_and_public_hashes(split, surface, live, gap):
    raw = observations.raw_cell(split, surface, live, gap)
    draws, families, _, _ = observations._split_plan(split)
    kind = split if split in ('train', 'dev') else ('seen' if surface == 'seen' else 'held')
    panels = [observations.scene_data.make_panel(
        kind, draw, families, live, gap, shift=surface == 'shift') for draw in draws]
    blocks = np.concatenate([panel.blocks for panel in panels])
    queries = np.concatenate([panel.queries for panel in panels])
    tokens = raw['feature_tokens']
    assert set(raw) == set(observations.RAW_FIELDS)
    assert tokens.dtype == np.int32 and tokens.shape[1] == 8
    np.testing.assert_array_equal(tokens[raw['blocks']], blocks)
    np.testing.assert_array_equal(tokens[raw['queries']], queries)
    np.testing.assert_array_equal(tokens, np.unique(
        np.concatenate((blocks.reshape(-1, 8), queries.reshape(-1, 8))), axis=0))
    for name in ('support', 'answers'):
        values = np.concatenate([getattr(panel, name) for panel in panels])
        assert raw[name].dtype == np.int32 and raw[name].shape == (len(blocks), 2, 2)
        np.testing.assert_array_equal(raw[name], values)
        np.testing.assert_array_equal(raw[name][:, 0], raw[name][:, 1, ::-1])
        assert np.all(raw[name][:, 0] != raw[name][:, 1])
    np.testing.assert_array_equal(raw['draw'], np.repeat(draws, families))
    np.testing.assert_array_equal(raw['family'], np.tile(np.arange(families), len(draws)))
    for family, digest in enumerate(raw['input_sha']):
        expected = sha256(blocks[family].astype('<i4').tobytes() +
                          queries[family].astype('<i4').tobytes()).hexdigest()
        assert digest == expected
        # A different actual counterfactual world is not the same public input.
        assert observations.public_sha(blocks[family, ::-1], queries[family]) != digest


@pytest.mark.parametrize('split,surface,live,gap', [
    ('train', 'held', 4, 0), ('train', 'seen', 4, 128),
    ('dev', 'shift', 8, 8), ('test', 'seen', 8, 8),
    ('test', 'unknown', 4, 0), ('unknown', 'seen', 4, 0),
    ('test', 'seen', 4.0, 0),
])
def test_unregistered_conditions_fail_before_generator_access(monkeypatch, split, surface, live, gap):
    def forbidden(*args, **kwargs):
        raise AssertionError('unregistered conditions cannot access data or neural weights')

    monkeypatch.setattr(observations.scene_data, 'make_panel', forbidden)
    monkeypatch.setattr(observations, 'WindowEncoder', forbidden)
    monkeypatch.setattr(observations, 'checkpoints', forbidden)
    with pytest.raises(ValueError):
        observations.raw_cell(split, surface, live, gap)


def test_raw_features_and_input_hashes_do_not_depend_on_labels_or_weights(monkeypatch):
    original_plan = observations._split_plan
    monkeypatch.setattr(observations, '_split_plan', lambda split:
        (original_plan(split)[0][:1], 2, original_plan(split)[2], original_plan(split)[3]))
    original_panel = observations.scene_data.make_panel
    baseline = observations.raw_cell('train', 'seen', 4, 0)

    def changed_supervision(*args, **kwargs):
        panel = original_panel(*args, **kwargs)
        return SimpleNamespace(blocks=panel.blocks, queries=panel.queries,
                               support=(panel.support + 1) % 4,
                               answers=(panel.answers + 1) % 8)

    def forbidden(*args, **kwargs):
        raise AssertionError('the raw boundary must not access a neural encoder or weight registry')

    monkeypatch.setattr(observations.scene_data, 'make_panel', changed_supervision)
    monkeypatch.setattr(observations, 'WindowEncoder', forbidden)
    monkeypatch.setattr(observations, 'checkpoints', forbidden)
    changed = observations.raw_cell('train', 'seen', 4, 0)
    for name in ('feature_tokens', 'blocks', 'queries', 'input_sha', 'draw', 'family'):
        np.testing.assert_array_equal(changed[name], baseline[name])
    assert np.all(changed['support'] != baseline['support'])
    assert np.all(changed['answers'] != baseline['answers'])


@pytest.fixture(scope='module')
def encoder():
    _, checkpoint, _ = next(observations.checkpoints())
    return observations.WindowEncoder(checkpoint)


def test_real_cache_forks_every_world_recomputes_shift_and_seals_boundaries(encoder, tmp_path, monkeypatch):
    original_plan = observations._split_plan

    def small_plan(split):
        draws, _, cells, surfaces = original_plan(split)
        return draws[:1], 2, cells[:1], surfaces

    monkeypatch.setattr(observations, '_split_plan', small_plan)
    # The real authenticated CPU RWKV is shared only as a test fixture, not
    # replaced by invented hidden vectors or an alternate inference path.
    monkeypatch.setattr(observations, 'WindowEncoder', lambda checkpoint: encoder)
    outputs = observations.make_cache(tmp_path, 'test', encoder.weight_seed)
    assert len(outputs) == 3
    cells = {}
    for surface, path in zip(('seen', 'held', 'shift'), outputs, strict=True):
        cell = observations.load_cell(tmp_path, 'test', encoder.weight_seed,
                                      encoder.weight_sha, surface, 4, 0)
        cells[surface] = cell
        raw = observations.raw_cell('test', surface, 4, 0)
        for name in observations.RAW_FIELDS:
            np.testing.assert_array_equal(cell[name], raw[name])
        tokens = cell['feature_tokens']
        np.testing.assert_array_equal(cell['features'], encoder.local(tokens))
        for world in range(2):
            expected = encoder.ordinary(tokens[cell['blocks'][:, world]], tokens[cell['queries']])
            np.testing.assert_array_equal(cell['ordinary'][:, world], expected)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive['metadata_json'].item())
        cost = metadata['cost']
        assert cost['ordinary_prefixes'] == 4
        assert cost['ordinary_prefix_tokens'] == 4 * 4 * 8
        assert cost['ordinary_queries'] == 8
        assert cost['local_windows'] == len(tokens)
        assert cost['local_tokens'] == len(tokens) * 8
        assert cost['ordinary_prefix_lanes'] == observations.frozen.ORDINARY_BATCH
        assert cost['ordinary_prefix_lane_tokens'] == observations.frozen.ORDINARY_BATCH * 4 * 8
        for name in ('local_seconds', 'ordinary_seconds', 'local_cpu_seconds', 'ordinary_cpu_seconds'):
            assert cost[name] >= 0
        if surface != 'seen':
            assert cost['weight_setup_seconds'] == cost['weight_setup_cpu_seconds'] == 0
        cost_path = path.with_name(path.stem + '-cost.csv')
        with cost_path.open(encoding='utf8', newline='') as stream:
            assert list(csv.DictReader(stream)) == [{key: str(value) for key, value in cost.items()}]
    held, shifted = cells['held'], cells['shift']
    for name in ('support', 'answers', 'draw', 'family'):
        np.testing.assert_array_equal(held[name], shifted[name])
    for name in ('blocks', 'queries'):
        assert not np.array_equal(held['feature_tokens'][held[name]], shifted['feature_tokens'][shifted[name]])
    assert np.all(held['input_sha'] != shifted['input_sha'])
    assert not list(tmp_path.glob('train-*')) and not list(tmp_path.glob('dev-*'))

    originals = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        observations.make_cache(tmp_path, 'test', encoder.weight_seed)
    assert originals == {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='frozen weight'):
        observations.load_cell(tmp_path, 'test', encoder.weight_seed, '0' * 64, 'seen', 4, 0)

    path = outputs[0]
    with np.load(path, allow_pickle=False) as archive:
        original = {name: archive[name] for name in archive.files}

    def reject(mutate, message):
        corrupted = {name: value.copy() for name, value in original.items()}
        mutate(corrupted)
        with path.open('wb') as stream:
            np.savez_compressed(stream, **corrupted)
        try:
            with pytest.raises(ValueError, match=message):
                observations.load_cell(tmp_path, 'test', encoder.weight_seed,
                                       encoder.weight_sha, 'seen', 4, 0)
        finally:
            path.write_bytes(originals[path])

    reject(lambda data: data.update(metadata_sha=np.asarray('0' * 64)), 'metadata SHA')
    reject(lambda data: data['input_sha'].__setitem__(0, '0' * 64), 'raw world/query digest')
    reject(lambda data: data['blocks'].__setitem__((0, 1, 0), len(data['feature_tokens'])),
           'blocks out of bounds')
    reject(lambda data: data['features'].__setitem__((0, 0), np.nan), 'nonfinite')
    reject(lambda data: data['ordinary'].__setitem__((0, 1, 0, 0),
           data['ordinary'][0, 1, 0, 0] + np.float32(1)), 'observation array SHA')


def test_cli_selects_exactly_one_split_and_optional_weight(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(observations, 'make_cache', lambda *args: calls.append(args))
    observations.main([str(tmp_path), '--split', 'dev', '--seed', '43'])
    observations.main([str(tmp_path), '--split', 'train'])
    assert calls == [(tmp_path, 'dev', 43), (tmp_path, 'train', None)]
