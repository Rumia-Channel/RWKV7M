"""Sealed raw-scene and genuine frozen RWKV observations for §22.22.

Run ``python scripts/role_binding_observations.py CACHE --split train --seed 42``
(and separately dev/test). Omitting --seed builds CE42/43/44. Each condition is
published exclusively as SPLIT-sSEED-SURFACE-kK-gG.npz and a matching -cost.csv.
The CSV separates local/continuous wall and process CPU time, actual/padded
encoder work, and one-time weight setup. No prefixes are shared across surfaces:
shift changes statements as well as questions. Packaging is outside encode time.
"""
import argparse
from hashlib import sha256
import json
from operator import index
from pathlib import Path
import re
from time import perf_counter, process_time

import numpy as np

import episodic_scene_features as frozen
from episodic_scene_features import WindowEncoder, feature_table, public_sha
from benchmark_address_screening import checkpoints, SEEDS
import role_binding_data as scene_data

FORMAT_VERSION = 1
RAW_FIELDS = ('feature_tokens', 'blocks', 'queries', 'support', 'answers',
              'draw', 'family', 'input_sha')
ARRAY_FIELDS = (*RAW_FIELDS, 'features', 'ordinary')
COST_FIELDS = ('cache_file', 'split', 'weight_seed', 'weight_sha', 'surface',
               'live', 'gap', 'families', 'unique_windows',
               'weight_setup_seconds', 'weight_setup_cpu_seconds',
               'local_seconds', 'local_cpu_seconds',
               'ordinary_seconds', 'ordinary_cpu_seconds',
               *frozen.COST_COUNTERS, 'cpu_device')


def _split_plan(split):
    if split == 'train':
        return scene_data.TRAIN_DRAWS, 24, scene_data.TRAIN_CELLS, ('seen',)
    if split == 'dev':
        return scene_data.DEV_DRAWS, 8, scene_data.TRAIN_CELLS, ('seen',)
    if split == 'test':
        return scene_data.TEST_DRAWS, 8, scene_data.TEST_CELLS, ('seen', 'held', 'shift')
    raise ValueError(f'unknown observation split: {split!r}')


def _condition(split, surface, live, gap):
    draws, families, cells, surfaces = _split_plan(split)
    try:
        live, gap = index(live), index(gap)
    except TypeError as error:
        raise ValueError('live and gap must be integer cell coordinates') from error
    if surface not in surfaces or (live, gap) not in cells:
        raise ValueError(f'unregistered {split} condition: {surface}, {(live, gap)}')
    return draws, families, live, gap


def _int32(array, shape, name, upper=None):
    if array.dtype != np.int32 or array.shape != shape:
        raise ValueError(f'expected int32 {name}{shape}')
    if upper is not None and np.any((array < 0) | (array >= upper)):
        raise ValueError(f'{name} out of bounds [0, {upper})')


def _swaps(support, answers):
    for name, values in (('support', support), ('answers', answers)):
        if (not np.array_equal(values[:, 0], values[:, 1, ::-1]) or
                np.any(values[:, 0, 0] == values[:, 0, 1])):
            raise ValueError(f'{name} must flip for each fixed query between worlds')


def raw_cell(split, surface, live, gap):
    """Return content-deduplicated public int32 utterances and separate labels.

    blocks[F,2,K+G] and queries[F,2] index feature_tokens[U,8]. Both support
    and answers have [F,world=2,query=2] axes; only raw tokens enter an encoder.
    input_sha[F] hashes both actual worlds and their shared questions, in order.
    """
    draws, families, live, gap = _condition(split, surface, live, gap)
    kind = split if split in ('train', 'dev') else ('seen' if surface == 'seen' else 'held')
    parts = {name: [] for name in ('blocks', 'queries', 'support', 'answers',
                                  'draw', 'family', 'input_sha')}
    for draw in draws:
        panel = scene_data.make_panel(kind, draw, families, live, gap, shift=surface == 'shift')
        for name, shape, upper in (
                ('blocks', (families, 2, live + gap, 8), 128),
                ('queries', (families, 2, 8), 128),
                ('support', (families, 2, 2), live),
                ('answers', (families, 2, 2), 8)):
            array = np.asarray(getattr(panel, name))
            _int32(array, shape, name, upper)
            parts[name].append(array)
        _swaps(panel.support, panel.answers)
        parts['draw'].append(np.full(families, draw, np.int32))
        parts['family'].append(np.arange(families, dtype=np.int32))
        parts['input_sha'].append(np.asarray(
            [public_sha(panel.blocks[f], panel.queries[f]) for f in range(families)], dtype='<U64'))
    raw = {name: np.concatenate(values) for name, values in parts.items()}
    # Use the established content-only table builder without computing any neural
    # or label-derived feature. Supervision never enters this call.
    table = feature_table(lambda tokens: None, raw.pop('blocks'), raw.pop('queries'))
    table.pop('features')
    return table | raw


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _array_sha(array):
    header = _json({'dtype': array.dtype.str, 'shape': array.shape}).encode('utf8')
    return sha256(header + b'\n' + array.tobytes(order='C')).hexdigest()


def _source_provenance():
    # Preserve the existing real encoder/model provenance, adding this boundary
    # and the role grammar rather than substituting a synthetic encoder identity.
    _, existing = frozen._source_provenance()
    sources = dict(entry.rsplit(':', 1) for entry in existing.split(';'))
    for path in (Path(__file__), Path(scene_data.__file__)):
        sources[path.resolve().relative_to(frozen.ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return sha256(_json(sources).encode('utf8')).hexdigest(), sources


def _paths(cache, split, seed, surface, live, gap):
    stem = f'{split}-s{seed}-{surface}-k{live}-g{gap}'
    return Path(cache) / f'{stem}.npz', Path(cache) / f'{stem}-cost.csv'


def _validate_arrays(arrays, split, surface, live, gap):
    draws, families, live, gap = _condition(split, surface, live, gap)
    rows = len(draws) * families
    tokens = arrays['feature_tokens']
    if tokens.ndim != 2 or not len(tokens):
        raise ValueError('empty or malformed feature_tokens table')
    unique = len(tokens)
    _int32(tokens, (unique, 8), 'feature_tokens', 128)
    for name, shape, upper in (
            ('blocks', (rows, 2, live + gap), unique),
            ('queries', (rows, 2), unique),
            ('support', (rows, 2, 2), live),
            ('answers', (rows, 2, 2), 8),
            ('draw', (rows,), None), ('family', (rows,), None)):
        _int32(arrays[name], shape, name, upper)
    if (not np.array_equal(arrays['draw'], np.repeat(draws, families)) or
            not np.array_equal(arrays['family'], np.tile(np.arange(families), len(draws)))):
        raise ValueError('wrong draw/family manifest for the requested split')
    _swaps(arrays['support'], arrays['answers'])
    used = np.unique(np.concatenate((arrays['blocks'].ravel(), arrays['queries'].ravel())))
    if (not np.array_equal(tokens, np.unique(tokens, axis=0)) or
            not np.array_equal(used, np.arange(unique))):
        raise ValueError('feature_tokens must be an exact content-deduplicated public table')
    hashes = arrays['input_sha']
    if (hashes.shape != (rows,) or hashes.dtype.kind != 'U' or
            any(re.fullmatch('[0-9a-f]{64}', value) is None for value in hashes)):
        raise ValueError('bad raw input digest manifest')
    for family, digest in enumerate(hashes):
        if public_sha(tokens[arrays['blocks'][family]], tokens[arrays['queries'][family]]) != digest:
            raise ValueError('raw world/query digest mismatch')
    for name, shape in (('features', (unique, frozen.FEATURE_DIM)),
                        ('ordinary', (rows, 2, 2, frozen.FEATURE_DIM))):
        array = arrays[name]
        if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'bad or nonfinite frozen RWKV {name}')


def load_cell(cache, split, seed, sha, surface, live, gap):
    """Load one sealed condition without loading a model or reading another split."""
    _, _, live, gap = _condition(split, surface, live, gap)
    if seed not in SEEDS or not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
        raise ValueError('expected a CE42/43/44 seed and its registered SHA-256')
    path, _ = _paths(cache, split, seed, surface, live, gap)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_FIELDS) | {'metadata_json', 'metadata_sha'}:
            raise ValueError(f'wrong observation archive fields: {path}')
        saved = {name: archive[name] for name in archive.files}
    for name in ('metadata_json', 'metadata_sha'):
        if saved[name].shape != () or saved[name].dtype.kind != 'U':
            raise ValueError(f'bad sealed {name}: {path}')
    text = saved.pop('metadata_json').item()
    if sha256(text.encode('utf8')).hexdigest() != saved.pop('metadata_sha').item():
        raise ValueError(f'sealed metadata SHA mismatch: {path}')
    try:
        metadata = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ValueError(f'bad sealed metadata JSON: {path}') from error
    source_sha, sources = _source_provenance()
    expected = dict(format_version=FORMAT_VERSION, split=split, surface=surface,
                    live=live, gap=gap, weight_seed=seed, weight_sha=sha,
                    source_sha=source_sha, source_hashes=sources)
    if (not isinstance(metadata, dict) or
            set(metadata) != set(expected) | {'array_sha256', 'cost'} or
            any(type(metadata[key]) is not type(value) or metadata[key] != value
                for key, value in expected.items()) or _json(metadata) != text):
        raise ValueError(f'wrong sealed condition, frozen weight, or source provenance: {path}')
    _validate_arrays(saved, split, surface, live, gap)
    if metadata['array_sha256'] != {name: _array_sha(saved[name]) for name in ARRAY_FIELDS}:
        raise ValueError(f'sealed observation array SHA mismatch: {path}')
    cost = metadata['cost']
    if (not isinstance(cost, dict) or set(cost) != set(COST_FIELDS) or
            any(cost.get(key) != value for key, value in
                dict(cache_file=path.name, split=split, weight_seed=seed, weight_sha=sha,
                     surface=surface, live=live, gap=gap, families=len(saved['family']),
                     unique_windows=len(saved['feature_tokens'])).items())):
        raise ValueError(f'wrong sealed cost association: {path}')
    return saved


def _timed(call):
    wall, cpu = perf_counter(), process_time()
    result = call()
    return result, perf_counter() - wall, process_time() - cpu


def make_cache(cache, split, seed=None):
    """Build every registered cell/surface in exactly one split; never clobber."""
    _, _, cells, surfaces = _split_plan(split)
    if seed is not None and seed not in SEEDS:
        raise ValueError('seed must be 42, 43, or 44')
    registered = [entry for entry in checkpoints() if seed is None or entry[0] == seed]
    if not registered or {entry[0] for entry in registered} != (set(SEEDS) if seed is None else {seed}):
        raise ValueError('missing ordinary frozen CE checkpoint registration')
    targets = [(weight_seed, surface, live, gap, *_paths(cache, split, weight_seed, surface, live, gap))
               for weight_seed, _, _ in registered for live, gap in cells for surface in surfaces]
    for _, _, _, _, path, cost_path in targets:
        if path.exists() or cost_path.exists():
            raise FileExistsError(f'refusing to overwrite existing observation artifact: {path}')
    Path(cache).mkdir(parents=True, exist_ok=True)
    source_sha, sources = _source_provenance()
    for weight_seed, checkpoint, weight_sha in registered:
        encoder, setup_wall, setup_cpu = _timed(lambda: WindowEncoder(checkpoint))
        if encoder.weight_seed != weight_seed or encoder.weight_sha != weight_sha:
            raise ValueError('encoder disagrees with the authenticated CE registry')
        for live, gap in cells:
            for surface in surfaces:
                arrays = raw_cell(split, surface, live, gap)
                tokens, rows = arrays['feature_tokens'], len(arrays['family'])
                blocks = tokens[arrays['blocks']].reshape(rows * 2, live + gap, 8)
                queries = np.repeat(tokens[arrays['queries']], 2, axis=0)
                before = dict(encoder.cost)
                arrays['features'], local_wall, local_cpu = _timed(lambda: encoder.local(tokens))
                # Each flattened (family, world) gets its own zero-state physical
                # history. WindowEncoder forks its two questions independently.
                ordinary, ordinary_wall, ordinary_cpu = _timed(lambda: encoder.ordinary(blocks, queries))
                arrays['ordinary'] = ordinary.reshape(rows, 2, 2, frozen.FEATURE_DIM)
                _validate_arrays(arrays, split, surface, live, gap)
                path, cost_path = _paths(cache, split, weight_seed, surface, live, gap)
                cost = dict(cache_file=path.name, split=split, weight_seed=weight_seed,
                            weight_sha=weight_sha, surface=surface, live=live, gap=gap,
                            families=rows, unique_windows=len(tokens),
                            weight_setup_seconds=setup_wall, weight_setup_cpu_seconds=setup_cpu,
                            local_seconds=local_wall, local_cpu_seconds=local_cpu,
                            ordinary_seconds=ordinary_wall, ordinary_cpu_seconds=ordinary_cpu,
                            **{name: encoder.cost[name] - before[name] for name in frozen.COST_COUNTERS},
                            cpu_device=str(encoder.cpu))
                metadata = dict(format_version=FORMAT_VERSION, split=split, surface=surface,
                                live=live, gap=gap, weight_seed=weight_seed, weight_sha=weight_sha,
                                source_sha=source_sha, source_hashes=sources, cost=cost,
                                array_sha256={name: _array_sha(arrays[name]) for name in ARRAY_FIELDS})
                text = _json(metadata)
                frozen._atomic_npz(path, arrays | dict(metadata_json=np.asarray(text),
                    metadata_sha=np.asarray(sha256(text.encode('utf8')).hexdigest())))
                try:
                    frozen._atomic_cost(cost_path, cost, COST_FIELDS)
                except BaseException:
                    path.unlink()
                    raise
                setup_wall = setup_cpu = 0.0
                print(f'{path.name}: {rows} paired families; {len(tokens)} local windows; '
                      f'{cost["ordinary_prefixes"]} independent world prefixes; '
                      f'{local_wall + ordinary_wall:.3f}s encode; {cost_path.name}', flush=True)
    return [entry[-2] for entry in targets]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--split', required=True, choices=('train', 'dev', 'test'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    make_cache(args.cache, args.split, args.seed)


if __name__ == '__main__':
    main()
