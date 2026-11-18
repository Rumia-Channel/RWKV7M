"""Visible legend binding and real frozen-RWKV edge caches for §22.19.

The lexical adapter is an exact public-dictionary lookup, not a learned result.
``observe`` converts real unrestricted-head predictions to zero-based addresses;
non-value head outputs become the controller's missing-address ID 16, not labels.
"""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import addressed_binding_tokens
import addressed_rwkv_reader
import binding_state_memory
import iterative_read_tasks
import probe_binding_stress
import route_transfer_panel
from addressed_binding_tokens import COUNT, KEY_BASE, VALUE_BASE
from addressed_rwkv_reader import AddressedReader, FrozenBindingCodec
from benchmark_address_screening import checkpoints
from binding_state_memory import _read_fresh_queries
from iterative_read_tasks import SPLITS, TEST_RULES, route_labels
from probe_binding_stress import _advance_once
from route_transfer_panel import ALIAS_BASE, LEGEND, RULES, make_panel, questions
from rwkv7m.cli.prepare_retrieval import FILLER_TOKEN, STORE_TOKEN
from rwkv7m.model.state import init_rwkv_state, init_screen_state
from versioned_binding_panel import literal_reference

ROOT = Path(__file__).resolve().parents[1]
FRONTENDS = ('ordinary', 'isolated')
PREFILL_BLOCK = 64
FAMILY_BATCH = 8
HELDOUT_ALIAS_BASE = 80
CACHE_KEYS = ('ordinary', 'isolated', 'truth', 'query', 'demo', 'draw', 'family',
              'gap', 'digest', 'weight_sha', 'weight_seed')
COST_FIELDS = ('split', 'weight_seed', 'weight_sha', 'draw', 'gap', 'family_start',
               'families', 'worlds', 'frontend', 'history_tokens', 'query_tokens',
               'recurrent_tokens', 'parser_tokens', 'head_vectors', 'seconds',
               'codec_seconds', 'input_sha256', 'source_sha256',
               'source_hashes')


class PublicInputs(NamedTuple):
    history: np.ndarray  # int32[F,2,T], raw aliased writes plus their visible legend
    query: np.ndarray    # int32[F,2,7], source spellings only; no target answer
    demo: np.ndarray     # int32[F,2,3], source spellings only; answers supplied later


def _legend(history):
    rows = np.asarray(history)
    if (rows.ndim != 2 or rows.dtype != np.int32 or rows.shape[0] < 1 or
            rows.shape[1] < 4 * COUNT):
        raise ValueError('expected int32 history[B,T] with a complete visible legend')
    entries = rows[:, :4 * COUNT].reshape(len(rows), COUNT, 4)
    canonical, aliases = entries[:, :, 1], entries[:, :, 3]
    if (not np.all(entries[:, :, 0] == LEGEND) or
            not np.all(np.sort(canonical, axis=1) == KEY_BASE + np.arange(COUNT)) or
            not np.all(entries[:, :, 2] == canonical - KEY_BASE + VALUE_BASE) or
            np.any((aliases < ALIAS_BASE) | (aliases >= 128)) or
            np.any(np.diff(np.sort(aliases, axis=1), axis=1) == 0)):
        raise ValueError('expected a complete visible bijective canonical/value/alias legend')
    return rows, canonical, aliases


def _resolve(canonical, aliases, tokens):
    names = np.asarray(tokens)
    if names.ndim < 1 or names.dtype != np.int32 or names.shape[0] != len(canonical):
        raise ValueError('expected int32 names[B,...] matching the history batch')
    flat = names.reshape(len(canonical), -1)
    matches = ((flat[:, :, None] == canonical[:, None, :]) |
               (flat[:, :, None] == aliases[:, None, :]))
    if not np.all(matches.sum(axis=-1) == 1):
        raise ValueError('unknown or ambiguous source name in the visible legend')
    nodes = np.take_along_axis(canonical, matches.argmax(axis=-1), axis=1) - KEY_BASE
    return nodes.reshape(names.shape).astype(np.int32, copy=False)


def resolve_names(history, tokens):
    """Resolve declared canonical keys or opaque aliases by visible equality only."""
    _, canonical, aliases = _legend(history)
    return _resolve(canonical, aliases, tokens)


def known_vocabulary(history):
    """Copy a raw prefix, replacing only declared STORE source names with keys.

    Legend spellings and literal values are untouched. Unknown names, incomplete
    writes and non-STORE/non-filler suffixes are errors, not silently dropped
    events. This routine never executes the graph or reads a target answer.
    """
    rows, canonical, aliases = _legend(history)
    normalized = rows.copy()
    for b, row in enumerate(rows):
        starts = np.flatnonzero(row[4 * COUNT:] == STORE_TOKEN) + 4 * COUNT
        if len(starts) and starts[-1] + 2 >= len(row):
            raise ValueError('incomplete STORE in the visible history')
        arguments = row[starts + 2]
        if np.any((arguments < VALUE_BASE) | (arguments >= VALUE_BASE + COUNT)):
            raise ValueError('STORE requires a visible literal value')
        filler = np.ones(len(row) - 4 * COUNT, dtype=bool)
        for offset in range(3):
            filler[starts + offset - 4 * COUNT] = False
        if np.any(row[4 * COUNT:][filler] != FILLER_TOKEN):
            raise ValueError('expected only STORE triples and filler after the legend')
        names = row[starts + 1][None, :]
        nodes = _resolve(canonical[b:b + 1], aliases[b:b + 1], names)[0]
        normalized[b, starts + 1] = nodes + KEY_BASE
    return normalized


def _alias_names(history, tokens):
    _, canonical, aliases = _legend(history)
    nodes = _resolve(canonical, aliases, tokens)
    # Resolve canonical node indices back through each episode's declaration order.
    positions = ((nodes.reshape(len(canonical), -1)[:, :, None] + KEY_BASE) ==
                 canonical[:, None, :]).argmax(axis=-1)
    return np.take_along_axis(aliases, positions, axis=1).reshape(nodes.shape).astype(np.int32)


def render_panel(panel, *, heldout=False):
    """Expose only raw aliased histories and aliased demo/question source names.

    Held-out spelling changes declarations and their source uses, not arbitrary
    token IDs. No rule-specific demo answer or test answer enters this rendering.
    """
    raw = np.asarray(panel.alias)
    if raw.ndim != 3 or raw.shape[1] != 2 or raw.dtype != np.int32:
        raise ValueError('expected int32 paired-world panel histories[F,2,T]')
    history = raw.copy().reshape(-1, raw.shape[-1])
    known_vocabulary(history)  # Validate the original declared writes before remapping.
    if heldout:
        _, canonical, aliases = _legend(history)
        if not np.all(np.sort(aliases, axis=1) == ALIAS_BASE + np.arange(COUNT)):
            raise ValueError('held-out remapping requires the declared training alias alphabet')
        replacement = HELDOUT_ALIAS_BASE + aliases - ALIAS_BASE
        for b, row in enumerate(history):
            starts = np.flatnonzero(row[4 * COUNT:] == STORE_TOKEN) + 4 * COUNT
            source = row[starts + 1]
            matches = source[:, None] == aliases[b]
            # Only declared aliases are remapped; canonical source spellings stay keys.
            declared = matches.any(axis=-1)
            row[starts[declared] + 1] = replacement[b, matches[declared].argmax(axis=-1)]
        history[:, :4 * COUNT].reshape(-1, COUNT, 4)[:, :, 3] = replacement
    families = len(raw)
    query = np.asarray(panel.queries)
    examples = np.asarray(panel.examples)
    if (query.shape != (families, 2, 7, 3) or query.dtype != np.int32 or
            examples.shape != (len(RULES), families, 2, 3, 4) or examples.dtype != np.int32 or
            not np.all(examples[:, :, :, :, :3] == examples[0, :, :, :, :3])):
        raise ValueError('expected shared public demo sources and seven paired questions')
    public_query = _alias_names(history, query[..., 1].reshape(2 * families, 7))
    public_demo = _alias_names(history, examples[0, ..., 1].reshape(2 * families, 3))
    return PublicInputs(history.reshape(raw.shape), public_query.reshape(families, 2, 7),
                        public_demo.reshape(families, 2, 3))


def observe(codec, isolated_reader, history, *, frontends=FRONTENDS):
    """Return int32[B,16] addresses 0..15, or 16 for a non-value head output.

    Inputs are public raw histories. The same lexical normalization is applied
    to both frozen readers. No graph truth, rule, demo answer or test label is
    accepted, and each query branches from the prefixed state independently.
    The argmax uses all 128 head logits before token IDs are encoded as edges.
    """
    selected = tuple(frontends)
    if len(set(selected)) != len(selected) or any(name not in FRONTENDS for name in selected):
        raise ValueError('frontends must be distinct ordinary/isolated names')
    prefix = known_vocabulary(history)
    lanes = len(prefix)
    queries = questions(np.broadcast_to(np.arange(COUNT, dtype=np.int32), (lanes, COUNT)))
    predictions = {}
    if 'ordinary' in selected:
        model = codec.model
        state = (init_rwkv_state(lanes, model.config),
                 init_screen_state(lanes, model.config.screening))
        for start in range(0, prefix.shape[1], PREFILL_BLOCK):
            state = _advance_once(model, jnp.asarray(prefix[:, start:start + PREFILL_BLOCK]), state)
        predictions['ordinary'] = np.asarray(_read_fresh_queries(model, jnp.asarray(queries), state))
    if 'isolated' in selected:
        if isolated_reader.codec is not codec:
            raise ValueError('isolated reader must use the supplied frozen codec')
        state = isolated_reader.start(lanes)
        for start in range(0, prefix.shape[1], PREFILL_BLOCK):
            state = isolated_reader.advance(state, prefix[:, start:start + PREFILL_BLOCK])
        predictions['isolated'] = np.asarray(isolated_reader.query(state, queries))
    for name, pred in predictions.items():
        if pred.shape != (lanes, COUNT) or pred.dtype != np.int32:
            raise ValueError(f'{name} did not return int32 unrestricted-head predictions[B,16]')
    return {name: head_to_edges(pred) for name, pred in predictions.items()}


def head_to_edges(predictions):
    """Encode actual head token IDs as controller addresses, preserving misses."""
    pred = np.asarray(predictions)
    if pred.dtype != np.int32:
        raise ValueError('expected int32 unrestricted-head token IDs')
    return np.where((pred >= VALUE_BASE) & (pred < VALUE_BASE + COUNT),
                    pred - VALUE_BASE, COUNT).astype(np.int32)


def _validated_arrays(panel, public):
    """Independently check generator labels; results are cache supervision only."""
    families = len(public.history)
    history = public.history.reshape(2 * families, -1)
    query = resolve_names(history, public.query.reshape(2 * families, 7)).reshape(families, 2, 7)
    demo = resolve_names(history, public.demo.reshape(2 * families, 3)).reshape(families, 2, 3)
    keys = np.broadcast_to(KEY_BASE + np.arange(COUNT, dtype=np.int32), (2 * families, COUNT))
    literals = literal_reference(known_vocabulary(history), keys)
    if np.any((literals < VALUE_BASE) | (literals >= VALUE_BASE + COUNT)):
        raise ValueError('visible history does not bind every station to a literal')
    truth = (literals - VALUE_BASE).reshape(families, 2, COUNT).astype(np.int32)
    if not np.all(np.sort(truth, axis=-1) == np.arange(COUNT)):
        raise ValueError('visible route graph must be a permutation in each world')
    if (np.any(np.all(truth[:, 0] == truth[:, 1], axis=-1)) or
            not np.array_equal(query[:, 0], query[:, 1]) or
            not np.array_equal(demo[:, 0], demo[:, 1]) or
            np.any(demo[..., :, None] == query[..., None, :])):
        raise ValueError('expected distinct paired worlds and disjoint demo/target stations')
    for index, rule in enumerate(RULES):
        answer = route_labels(truth, query, TEST_RULES[rule])
        example = route_labels(truth, demo, TEST_RULES[rule])
        if (not np.array_equal(answer + VALUE_BASE, panel.gold[index]) or
                not np.array_equal(example + VALUE_BASE, panel.examples[index, ..., 3]) or
                not np.all(answer[:, 0] != answer[:, 1]) or
                np.any(example[..., :, None] == query[..., None, :])):
            raise ValueError(f'independent visible route labels disagree: {rule}')
    return truth, query, demo


def _public_digest(history, demo, query):
    digest = sha256(np.asarray(history, dtype='<i4').tobytes())
    digest.update(np.asarray(demo, dtype='<i4').tobytes())
    digest.update(np.asarray(query, dtype='<i4').tobytes())
    return digest.hexdigest()


def _source_hashes():
    paths = [Path(__file__), Path(addressed_binding_tokens.__file__),
             Path(addressed_rwkv_reader.__file__), Path(binding_state_memory.__file__),
             Path(iterative_read_tasks.__file__), Path(probe_binding_stress.__file__),
             Path(route_transfer_panel.__file__)]
    entries = ';'.join(f'{p.resolve().relative_to(ROOT).as_posix()}:{sha256(p.read_bytes()).hexdigest()}'
                       for p in sorted(paths))
    return sha256(entries.encode('utf8')).hexdigest(), entries


def prepare_cache(directory, split, *, seed=None):
    """Prepare exactly one explicitly selected split, without replacing artifacts."""
    if split not in SPLITS:
        raise ValueError(f'unknown split: {split}')
    registered = list(checkpoints())  # The registry verifies every returned SHA-256.
    chosen = [entry for entry in registered if seed is None or entry[0] == seed]
    if not chosen:
        raise ValueError(f'weight seed {seed} is not in the SHA-checked registry')
    directory = Path(directory)
    targets = [(directory / f'{split}-s{s}.npz', directory / f'{split}-s{s}-cost.csv')
               for s, _, _ in chosen]
    for paths in targets:
        for path in paths:
            if path.exists():
                raise FileExistsError(f'refusing to overwrite existing cache artifact: {path}')
    directory.mkdir(parents=True, exist_ok=True)
    source_sha, sources = _source_hashes()
    draws, families, gaps = SPLITS[split]
    completed = []
    for (weight_seed, checkpoint, weight_sha), (target, cost_path) in zip(chosen, targets, strict=True):
        start = perf_counter()
        codec = FrozenBindingCodec(checkpoint)
        isolated_reader = AddressedReader(codec, 'isolated')
        codec_seconds = perf_counter() - start
        arrays = {name: [] for name in CACHE_KEYS if name not in ('weight_sha', 'weight_seed')}
        costs = []
        for draw in draws:
            for gap in gaps:
                panel = make_panel(draw, gap, families=families)
                public = render_panel(panel, heldout=split == 'test')
                truth, query, demo = _validated_arrays(panel, public)
                digests = np.asarray([_public_digest(public.history[f], public.demo[f], public.query[f])
                                      for f in range(families)], dtype='<U64')
                for first in range(0, families, FAMILY_BATCH):
                    end = min(first + FAMILY_BATCH, families)
                    count = end - first
                    raw_history = public.history[first:end].reshape(2 * count, -1)
                    for frontend in FRONTENDS:
                        start = perf_counter()
                        edges = observe(codec, isolated_reader, raw_history,
                                        frontends=(frontend,))[frontend]
                        seconds = perf_counter() - start
                        arrays[frontend].append(edges.reshape(count, 2, COUNT))
                        query_tokens = 2 * count * COUNT * 3
                        costs.append(dict(
                            split=split, weight_seed=weight_seed, weight_sha=weight_sha,
                            draw=draw, gap=gap, family_start=first, families=count, worlds=2,
                            frontend=frontend, history_tokens=raw_history.size,
                            query_tokens=query_tokens,
                            recurrent_tokens=raw_history.size + query_tokens if frontend == 'ordinary' else 0,
                            parser_tokens=raw_history.size if frontend == 'isolated' else 0,
                            head_vectors=edges.size, seconds=seconds, codec_seconds=codec_seconds,
                            input_sha256=_public_digest(public.history[first:end], public.demo[first:end],
                                                       public.query[first:end]),
                            source_sha256=source_sha, source_hashes=sources))
                    for name, values in (('truth', truth), ('query', query), ('demo', demo)):
                        arrays[name].append(values[first:end])
                    arrays['draw'].append(np.full(count, draw, np.int32))
                    arrays['family'].append(np.arange(first, end, dtype=np.int32))
                    arrays['gap'].append(np.full(count, gap, np.int32))
                    arrays['digest'].append(digests[first:end])
        packed = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
        packed['weight_sha'] = np.asarray(weight_sha)
        packed['weight_seed'] = np.asarray(weight_seed, dtype=np.int32)
        # Exclusive creation also protects against a file appearing during model execution.
        with target.open('xb') as stream:
            np.savez_compressed(stream, **packed)
        with cost_path.open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COST_FIELDS)
            writer.writeheader()
            writer.writerows(costs)
        completed.append(target)
        print(f'{target}: {len(packed["family"])} paired families; real frozen readers cached', flush=True)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--split', choices=tuple(SPLITS), required=True)
    parser.add_argument('--seed', type=int, help='one registered frozen weight seed (default: all)')
    args = parser.parse_args(argv)
    prepare_cache(args.cache, args.split, seed=args.seed)


if __name__ == '__main__':
    main()
