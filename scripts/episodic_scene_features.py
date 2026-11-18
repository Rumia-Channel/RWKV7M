"""Frozen CPU RWKV features and exclusive per-condition caches for §22.20.

Only public eight-token utterances enter the encoder. Local windows start at
zero recurrent state; ordinary questions fork independently from the complete
physical history. Neither path calls the vocabulary head or accepts labels.
"""
import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
import io
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
# A direct Windows ``python scripts/episodic_scene_features.py ...`` invocation
# must not depend on the caller having installed this sibling worktree.
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

import episodic_scene_data as scene_data
from benchmark_address_screening import checkpoints
from rwkv7m.io import load_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state

from rwkv7m.model import rwkv_core, wkv
LOCAL_BATCH = 32
ORDINARY_BATCH = 16
PREFIX_TOKENS = 128
FEATURE_DIM = 256
COST_COUNTERS = (
    'local_windows', 'local_lane_windows', 'local_tokens', 'local_lane_tokens',
    'ordinary_prefixes', 'ordinary_prefix_lanes', 'ordinary_prefix_tokens',
    'ordinary_prefix_lane_tokens', 'ordinary_queries', 'ordinary_query_lanes',
    'ordinary_query_tokens', 'ordinary_query_lane_tokens', 'prefix_chunk_calls',
)

CACHE_KEYS = (
    'features', 'feature_tokens', 'blocks', 'queries', 'ordinary', 'support',
    'answers', 'draw', 'family', 'input_sha', 'weight_sha', 'weight_seed',
    'split', 'surface', 'live', 'gap', 'format_version', 'source_sha',
    'source_hashes', 'cost_json',
)
COST_FIELDS = (
    'split', 'weight_seed', 'weight_sha', 'surface', 'live', 'gap', 'families',
    'draws', 'unique_windows', 'local_seconds', 'ordinary_seconds',
    'weight_setup_seconds', *COST_COUNTERS, 'shared_prefix_from',
    'shared_prefix_surfaces', 'cpu_device', 'local_batch', 'ordinary_batch',
    'prefix_chunk_tokens', 'source_sha', 'source_hashes',
)


def _window_hidden(model, tokens, state):
    hidden, _, _, _ = model.compute_recurrent_hidden(
        tokens, *state, phase='read_write', deterministic=True)
    return hidden


_local_hidden = nnx.jit(_window_hidden)


@jax.jit
def _pool_hidden(hidden):
    # A separate compiled call materializes the checkpoint's actual BF16 hidden.
    # Fusing this upcast into the encoder can silently bypass BF16 output rounding on CPU.
    x = hidden.astype(jnp.float32)
    return jnp.concatenate((x.mean(axis=-2), x[..., -1, :]), axis=-1)


@nnx.jit
def _prefix_step(model, tokens, state):
    _, rwkv, screen, _ = model.compute_recurrent_hidden(
        tokens, *state, phase='read_write', deterministic=True)
    return rwkv, screen


@nnx.jit
def _query_hidden(model, queries, state):
    # Vectorizing the independent query dimension broadcasts the immutable
    # prefix state, rather than serially advancing one question into the next.
    hidden = jax.vmap(lambda tokens: _window_hidden(model, tokens, state))(
        queries.swapaxes(0, 1))
    return hidden.swapaxes(0, 1)


def _authenticated_checkpoint(checkpoint_path):
    path = Path(checkpoint_path).resolve()
    for seed, registered, digest in checkpoints():
        if path == Path(registered).resolve():
            actual = sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f'changed frozen CE checkpoint: {path}')
            return seed, path, digest
    raise ValueError(f'checkpoint is not in the SHA-authenticated CE registry: {path}')


def _tokens(values, ndim, name):
    values = np.asarray(values)
    if values.dtype != np.int32 or values.ndim != ndim or values.shape[-1] != 8:
        raise ValueError(f'expected int32 {name} with {ndim} dimensions and eight tokens per utterance')
    if np.any((values < 0) | (values >= 128)):
        raise ValueError(f'{name} contains a token outside the public 0..127 vocabulary')
    return values


def _finite(features):
    if not np.isfinite(features).all():
        raise FloatingPointError('nonfinite frozen RWKV hidden feature')
    return features


class WindowEncoder:
    """The actual frozen ordinary D=128 NNx checkpoint, always run on CPU."""

    def __init__(self, checkpoint_path):
        self.weight_seed, path, self.weight_sha = _authenticated_checkpoint(checkpoint_path)
        self.checkpoint_path = path
        self.cpu = jax.devices('cpu')[0]
        with jax.default_device(self.cpu):
            params, config, _ = load_model_safetensors(path)
            if (config is None or config.use_screening or config.d_model != 128 or
                    config.vocab_size != 128):
                raise ValueError('expected an ordinary D=128, 128-ID RWKV CE checkpoint')
            config = replace(config, remat_blocks=False)
            self.model = load_linen_params_into_nnx(
                initialize_nnx_model(jax.random.key(0), config), params)
            self._local_zero = self._zero(LOCAL_BATCH)
            self._ordinary_zero = self._zero(ORDINARY_BATCH)
        self.cost = dict.fromkeys(COST_COUNTERS, 0)

    def _zero(self, batch):
        config = self.model.config
        return init_rwkv_state(batch, config), init_screen_state(batch, config.screening)

    def local(self, tokens):
        """int32[N,8] -> float32[N,256], mean then last true recurrent hidden."""
        tokens = _tokens(tokens, 2, 'local tokens[N,8]')
        result = np.empty((len(tokens), FEATURE_DIM), dtype=np.float32)
        with jax.default_device(self.cpu):
            for first in range(0, len(tokens), LOCAL_BATCH):
                count = min(LOCAL_BATCH, len(tokens) - first)
                padded = np.zeros((LOCAL_BATCH, 8), dtype=np.int32)
                padded[:count] = tokens[first:first + count]
                hidden = _local_hidden(self.model, jax.device_put(padded, self.cpu), self._local_zero)
                result[first:first + count] = np.asarray(_pool_hidden(hidden))[:count]
        _finite(result)
        lanes = ((len(tokens) + LOCAL_BATCH - 1) // LOCAL_BATCH) * LOCAL_BATCH
        self.cost['local_windows'] += len(tokens)
        self.cost['local_lane_windows'] += lanes
        self.cost['local_tokens'] += tokens.size
        self.cost['local_lane_tokens'] += lanes * 8
        return result

    def ordinary(self, blocks, queries):
        """int32[B,N,8], int32[B,2,8] -> float32[B,2,256]."""
        return self.ordinary_many(blocks, np.asarray(queries)[None])[0]

    def ordinary_many(self, blocks, queries):
        """Encode query surfaces [S,B,2,8] with one shared continuous prefix.

        This saves the expensive held/shift prefix pass without sharing any
        query state or changing the public ``ordinary`` representation.
        """
        blocks = _tokens(blocks, 3, 'blocks[B,N,8]')
        queries = _tokens(queries, 4, 'queries[S,B,2,8]')
        surfaces, batch, questions, _ = queries.shape
        if questions != 2 or batch != len(blocks) or surfaces < 1:
            raise ValueError('ordinary requires two independent queries per prefix and surface')
        result = np.empty((surfaces, batch, 2, FEATURE_DIM), dtype=np.float32)
        prefix_length = blocks.shape[1] * 8
        with jax.default_device(self.cpu):
            for first in range(0, batch, ORDINARY_BATCH):
                count = min(ORDINARY_BATCH, batch - first)
                prefix = np.zeros((ORDINARY_BATCH, prefix_length), dtype=np.int32)
                prefix[:count] = blocks[first:first + count].reshape(count, prefix_length)
                state = self._ordinary_zero
                for start in range(0, prefix_length, PREFIX_TOKENS):
                    state = _prefix_step(self.model, jax.device_put(
                        prefix[:, start:start + PREFIX_TOKENS], self.cpu), state)
                questions_padded = np.zeros((ORDINARY_BATCH, 2 * surfaces, 8), dtype=np.int32)
                questions_padded[:count] = queries[:, first:first + count].transpose(
                    1, 0, 2, 3).reshape(count, 2 * surfaces, 8)
                hidden = _query_hidden(self.model, jax.device_put(questions_padded, self.cpu), state)
                encoded = np.asarray(_pool_hidden(hidden))[:count]
                result[:, first:first + count] = encoded.reshape(
                    count, surfaces, 2, FEATURE_DIM).transpose(1, 0, 2, 3)
        _finite(result)
        batches = (batch + ORDINARY_BATCH - 1) // ORDINARY_BATCH
        lanes = batches * ORDINARY_BATCH
        self.cost['ordinary_prefixes'] += batch
        self.cost['ordinary_prefix_lanes'] += lanes
        self.cost['ordinary_prefix_tokens'] += batch * prefix_length
        self.cost['ordinary_prefix_lane_tokens'] += lanes * prefix_length
        self.cost['ordinary_queries'] += surfaces * batch * 2
        self.cost['ordinary_query_lanes'] += surfaces * lanes * 2
        self.cost['ordinary_query_tokens'] += surfaces * batch * 2 * 8
        self.cost['ordinary_query_lane_tokens'] += surfaces * lanes * 2 * 8
        self.cost['prefix_chunk_calls'] += batches * (
            (prefix_length + PREFIX_TOKENS - 1) // PREFIX_TOKENS)
        return result


def public_sha(blocks, queries):
    """SHA-256 of both actual worlds and both raw questions, never supervision."""
    digest = sha256(np.asarray(blocks, dtype='<i4').tobytes(order='C'))
    digest.update(np.asarray(queries, dtype='<i4').tobytes(order='C'))
    return digest.hexdigest()


def feature_table(encode, blocks, queries):
    """Content-deduplicate public utterances within exactly one condition."""
    family_count, worlds, count, _ = blocks.shape
    combined = np.concatenate((blocks.reshape(-1, 8), queries.reshape(-1, 8)))
    unique, inverse = np.unique(combined, axis=0, return_inverse=True)
    boundary = blocks.size // 8
    return dict(
        features=encode(unique),
        feature_tokens=unique.astype(np.int32, copy=False),
        blocks=inverse[:boundary].reshape(family_count, worlds, count).astype(np.int32),
        queries=inverse[boundary:].reshape(family_count, 2).astype(np.int32),
    )


def _atomic_publish(path, writer):
    """Publish a complete artifact without ever replacing an existing path."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link is both atomic and exclusive on Windows
        # and POSIX. Unlike os.replace, it cannot clobber a concurrent writer.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_npz(path, arrays):
    _atomic_publish(path, lambda stream: np.savez_compressed(stream, **arrays))


def _atomic_cost(path, row, fields):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerow(row)
    payload = text.getvalue().encode('utf8')
    _atomic_publish(path, lambda stream: stream.write(payload))


def _split_plan(split):
    if split == 'train':
        return scene_data.TRAIN_DRAWS, 24, scene_data.TRAIN_CELLS, ('seen',)
    if split == 'dev':
        return scene_data.DEV_DRAWS, 8, scene_data.TRAIN_CELLS, ('seen',)
    if split == 'test':
        return scene_data.TEST_DRAWS, 8, scene_data.TEST_CELLS, ('seen', 'held', 'shift')
    if split == 'confirm':
        return scene_data.CONFIRM_DRAWS, 8, scene_data.TEST_CELLS, ('held',)
    raise ValueError(f'unknown cache split: {split}')


def _source_provenance():
    functions = (initialize_nnx_model, load_linen_params_into_nnx,
                 load_model_safetensors, init_rwkv_state)
    paths = {Path(__file__), Path(scene_data.__file__), Path(rwkv_core.__file__),
             Path(wkv.__file__)}
    paths.update(Path(sys.modules[function.__module__].__file__) for function in functions)
    sources = ';'.join(
        f'{path.resolve().relative_to(ROOT).as_posix()}:{sha256(path.read_bytes()).hexdigest()}'
        for path in sorted(paths))
    return sha256(sources.encode('utf8')).hexdigest(), sources


def _raw_cell(split, surface, live, gap):
    draws, families, _, _ = _split_plan(split)
    generation_split = split if split in ('train','dev') else ('seen' if surface == 'seen' else 'held')
    values = {name: [] for name in ('raw_blocks', 'raw_queries', 'support', 'answers',
                                  'draw', 'family', 'input_sha')}
    for draw in draws:
        panel = scene_data.make_panel(
            generation_split, draw, families, live, gap, shift=surface == 'shift')
        blocks = _tokens(panel.blocks, 4, 'raw blocks[F,2,N,8]')
        queries = _tokens(panel.queries, 3, 'raw queries[F,2,8]')
        support, answers = np.asarray(panel.support), np.asarray(panel.answers)
        if (blocks.shape != (families, 2, live + gap, 8) or
                queries.shape != (families, 2, 8) or support.shape != (families, 2) or
                support.dtype != np.int32 or answers.shape != (families, 2, 2) or
                answers.dtype != np.int32 or np.any((support < 0) | (support >= live + gap)) or
                np.any((answers < 0) | (answers >= 8))):
            raise ValueError('scene generator violated the paired raw-input/supervision contract')
        for name, array in (('raw_blocks', blocks), ('raw_queries', queries),
                            ('support', support), ('answers', answers)):
            values[name].append(array)
        values['draw'].append(np.full(families, draw, dtype=np.int32))
        values['family'].append(np.arange(families, dtype=np.int32))
        values['input_sha'].append(np.asarray(
            [public_sha(blocks[family], queries[family]) for family in range(families)],
            dtype='<U64'))
    return {name: np.concatenate(arrays, axis=0) for name, arrays in values.items()}


def _cache_paths(cache, split, seed, surface, live, gap):
    stem = f'{split}-s{seed}-{surface}-k{live}-g{gap}'
    return cache / f'{stem}.npz', cache / f'{stem}-cost.csv'


def _reuse_cache(path, cost_path, expected, source_sha):
    """Validate and reuse only complete, same-code, same-weight public features."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(CACHE_KEYS):
            raise ValueError(f'incomplete or incompatible cache fields: {path}')
        for name, value in expected.items():
            if saved[name].shape != () or saved[name].item() != value:
                raise ValueError(f'cache provenance mismatch for {name}: {path}')
        if saved['source_sha'].shape != () or saved['source_sha'].item() != source_sha:
            raise ValueError(f'cache source provenance mismatch: {path}')
        features, tokens = saved['features'], saved['feature_tokens']
        blocks, queries = saved['blocks'], saved['queries']
        count = len(saved['family'])
        live, gap = expected['live'], expected['gap']
        if (features.dtype != np.float32 or features.shape != (len(tokens), FEATURE_DIM) or
                tokens.dtype != np.int32 or tokens.shape != (len(tokens), 8) or
                np.any((tokens < 0) | (tokens >= 128)) or not np.isfinite(features).all() or
                blocks.dtype != np.int32 or blocks.shape != (count, 2, live + gap) or
                queries.dtype != np.int32 or queries.shape != (count, 2) or
                np.any((blocks < 0) | (blocks >= len(tokens))) or
                np.any((queries < 0) | (queries >= len(tokens)))):
            raise ValueError(f'invalid public feature table or indices: {path}')
        draws, families, _, _ = _split_plan(expected['split'])
        for name, shape in (('support', (count, 2)), ('answers', (count, 2, 2)),
                            ('draw', (count,)), ('family', (count,))):
            if saved[name].dtype != np.int32 or saved[name].shape != shape:
                raise ValueError(f'invalid cache supervision metadata {name}: {path}')
        if (count != len(draws) * families or
                not np.array_equal(saved['draw'], np.repeat(np.asarray(draws, np.int32), families)) or
                not np.array_equal(saved['family'], np.tile(np.arange(families, dtype=np.int32), len(draws))) or
                np.any((saved['support'] < 0) | (saved['support'] >= live + gap)) or
                np.any((saved['answers'] < 0) | (saved['answers'] >= 8)) or
                saved['input_sha'].shape != (count,) or saved['input_sha'].dtype.kind != 'U'):
            raise ValueError(f'invalid cache family or label boundaries: {path}')
        actual_sha = np.asarray([
            public_sha(tokens[blocks[index]], tokens[queries[index]]) for index in range(count)])
        if not np.array_equal(actual_sha, saved['input_sha']):
            raise ValueError(f'cache public input SHA-256 mismatch: {path}')
        row = json.loads(saved['cost_json'].item())
        if set(row) != set(COST_FIELDS):
            raise ValueError(f'incomplete physical cost provenance: {path}')
    # A process can stop after publishing its NPZ but before publishing CSV.
    # The complete row is embedded in that NPZ, so resumption never invents it.
    if not cost_path.exists():
        _atomic_cost(cost_path, row, COST_FIELDS)
    else:
        with cost_path.open(encoding='utf8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0] != {key: str(value) for key, value in row.items()}:
            raise ValueError(f'cache cost log disagrees with embedded measurements: {cost_path}')


def make_cache(cache, split, seed=None, *, surfaces=None, cells=None, reuse_existing=False):
    """Build only the requested split/weight/cells; never silently overwrite.

    ``reuse_existing`` explicitly resumes validated complete conditions. CPU
    timings include actual feature computation and compilation, not packaging.
    The padded-lane counters record physical work as well as true input counts.
    """
    _, _, available_cells, available_surfaces = _split_plan(split)
    selected_surfaces = available_surfaces if surfaces is None else tuple(surfaces)
    selected_cells = tuple(available_cells) if cells is None else tuple(map(tuple, cells))
    if (not selected_surfaces or len(set(selected_surfaces)) != len(selected_surfaces) or
            any(surface not in available_surfaces for surface in selected_surfaces)):
        raise ValueError(f'{split} surfaces must be distinct choices from {available_surfaces}')
    if (not selected_cells or len(set(selected_cells)) != len(selected_cells) or
            any(cell not in available_cells for cell in selected_cells)):
        raise ValueError(f'{split} cells must be distinct choices from {available_cells}')
    # Canonical ordering makes shared held/shift prefix cost attribution stable.
    selected_surfaces = tuple(surface for surface in available_surfaces if surface in selected_surfaces)
    requested_seeds = None if seed is None else ((seed,) if isinstance(seed, int) else tuple(seed))
    registered = list(checkpoints())
    if requested_seeds is not None and (
            not requested_seeds or len(set(requested_seeds)) != len(requested_seeds) or
            any(value not in {entry[0] for entry in registered} for value in requested_seeds)):
        raise ValueError('weight seeds must be distinct entries in the SHA-authenticated CE registry')
    chosen = [entry for entry in registered if requested_seeds is None or entry[0] in requested_seeds]
    cache = Path(cache)
    targets = [(weight_seed, surface, live, gap,
                *_cache_paths(cache, split, weight_seed, surface, live, gap))
               for weight_seed, _, _ in chosen for live, gap in selected_cells
               for surface in selected_surfaces]
    for _, _, _, _, path, cost_path in targets:
        if (path.exists() or cost_path.exists()) and not reuse_existing:
            raise FileExistsError(f'refusing to overwrite existing cache artifact: {path}')
        if cost_path.exists() and not path.exists():
            raise FileExistsError(f'orphaned cost log has no complete cache: {cost_path}')
    cache.mkdir(parents=True, exist_ok=True)
    source_sha, sources = _source_provenance()
    for weight_seed, checkpoint, weight_sha in chosen:
        pending = {}
        for live, gap in selected_cells:
            pending[live, gap] = []
            for surface in selected_surfaces:
                path, cost_path = _cache_paths(cache, split, weight_seed, surface, live, gap)
                expected = dict(split=split, surface=surface, live=live, gap=gap,
                                weight_seed=weight_seed, weight_sha=weight_sha, format_version=1)
                if path.exists():
                    _reuse_cache(path, cost_path, expected, source_sha)
                    print(f'{path}: reusing validated complete features', flush=True)
                else:
                    pending[live, gap].append(surface)
        if not any(pending.values()):
            continue
        start = perf_counter()
        encoder = WindowEncoder(checkpoint)
        setup_seconds = perf_counter() - start
        if encoder.weight_seed != weight_seed or encoder.weight_sha != weight_sha:
            raise ValueError('loaded encoder disagrees with the authenticated checkpoint registry')
        for (live, gap), missing_surfaces in pending.items():
            groups = [(surface,) for surface in missing_surfaces if surface == 'seen']
            shared = tuple(surface for surface in missing_surfaces if surface != 'seen')
            if shared:
                groups.append(shared)
            for group in groups:
                raw = {surface: _raw_cell(split, surface, live, gap) for surface in group}
                first = raw[group[0]]
                for surface in group[1:]:
                    for name in ('raw_blocks', 'support', 'answers', 'draw', 'family'):
                        if not np.array_equal(first[name], raw[surface][name]):
                            raise ValueError('held and shift must share actual histories and facts')
                family_count = len(first['family'])
                queries = np.stack([
                    np.broadcast_to(raw[surface]['raw_queries'][:, None],
                                    (family_count, 2, 2, 8)).reshape(2 * family_count, 2, 8)
                    for surface in group])
                before = dict(encoder.cost)
                start = perf_counter()
                ordinary = encoder.ordinary_many(
                    first['raw_blocks'].reshape(2 * family_count, live + gap, 8), queries)
                ordinary_seconds = perf_counter() - start
                ordinary_cost = {name: encoder.cost[name] - before[name] for name in COST_COUNTERS}
                for index, surface in enumerate(group):
                    public = raw[surface]
                    before = dict(encoder.cost)
                    start = perf_counter()
                    arrays = feature_table(encoder.local, public['raw_blocks'], public['raw_queries'])
                    local_seconds = perf_counter() - start
                    cost = {name: encoder.cost[name] - before[name] for name in COST_COUNTERS}
                    for name, value in ordinary_cost.items():
                        if name.startswith('ordinary_query') or name == 'ordinary_queries':
                            cost[name] += value // len(group)
                        elif index == 0:
                            cost[name] += value
                    row = dict(
                        split=split, weight_seed=weight_seed, weight_sha=weight_sha,
                        surface=surface, live=live, gap=gap, families=family_count,
                        draws=len(_split_plan(split)[0]), unique_windows=len(arrays['features']),
                        local_seconds=local_seconds,
                        ordinary_seconds=ordinary_seconds if index == 0 else 0.0,
                        weight_setup_seconds=setup_seconds, **cost,
                        shared_prefix_from='' if index == 0 else group[0],
                        shared_prefix_surfaces='+'.join(group), cpu_device=str(encoder.cpu),
                        local_batch=LOCAL_BATCH, ordinary_batch=ORDINARY_BATCH,
                        prefix_chunk_tokens=PREFIX_TOKENS, source_sha=source_sha, source_hashes=sources)
                    setup_seconds = 0.0
                    arrays.update({name: public[name] for name in
                                   ('support', 'answers', 'draw', 'family', 'input_sha')})
                    arrays.update(
                        ordinary=ordinary[index].reshape(family_count, 2, 2, FEATURE_DIM),
                        weight_sha=np.asarray(weight_sha), weight_seed=np.asarray(weight_seed, np.int32),
                        split=np.asarray(split), surface=np.asarray(surface),
                        live=np.asarray(live, np.int32), gap=np.asarray(gap, np.int32),
                        format_version=np.asarray(1, np.int32), source_sha=np.asarray(source_sha),
                        source_hashes=np.asarray(sources),
                        cost_json=np.asarray(json.dumps(row, sort_keys=True, separators=(',', ':'))))
                    path, cost_path = _cache_paths(cache, split, weight_seed, surface, live, gap)
                    _atomic_npz(path, arrays)
                    _atomic_cost(cost_path, row, COST_FIELDS)
                    print(f'{path}: {family_count} paired families; '
                          f'{row["ordinary_prefixes"]} true/{row["ordinary_prefix_lanes"]} physical prefix lanes; '
                          f'{row["ordinary_prefix_lane_tokens"]} physical prefix tokens; '
                          f'{local_seconds + row["ordinary_seconds"]:.3f}s encoding', flush=True)
    return [entry[-2] for entry in targets]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--split', choices=('train', 'dev', 'test', 'confirm'), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), action='append',
                        help='registered frozen weight seed; repeat to select several (default: all)')
    parser.add_argument('--surface', choices=('seen', 'held', 'shift'), action='append',
                        help='one applicable query surface; repeat to select several')
    parser.add_argument('--cell', nargs=2, type=int, metavar=('LIVE', 'GAP'), action='append',
                        help='one registered live/gap cell; repeat to select several')
    parser.add_argument('--reuse-existing', action='store_true',
                        help='resume only validated same-code, same-weight complete conditions')
    args = parser.parse_args(argv)
    make_cache(args.cache, args.split, args.seed, surfaces=args.surface, cells=args.cell,
               reuse_existing=args.reuse_existing)


if __name__ == '__main__':
    main()
