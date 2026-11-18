"""Run frozen RWKV on visible writes; optionally add a causal recent-write guard study.

Mixed-condition gaps count distraction before four true target overwrites. Their
identical eight-token filler suffix is not included in the reported gap.
"""
import argparse
import csv
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np

from probe_retrieval_distance import BLOCK, histories
from retrieval_frontier import FILLER, KEY_TOKENS, STORE, VALUE_TOKENS, make_examples


DIRECT_SEED = 731907
OVERWRITE_SEED = 731908
GUARD_DIRECT_SEED = 731927
GUARD_OVERWRITE_SEED = 731928
ROOT = Path(__file__).resolve().parents[1]


def _panel_seeds(direct_seed, overwrite_seed, guard_study):
    if direct_seed is None:
        direct_seed = GUARD_DIRECT_SEED if guard_study else DIRECT_SEED
    if overwrite_seed is None:
        overwrite_seed = GUARD_OVERWRITE_SEED if guard_study else OVERWRITE_SEED
    return direct_seed, overwrite_seed


def _visible_writes(history):
    """Read only completed legal public STORE/key/value events."""
    for marker, key, value in zip(history, history[1:], history[2:]):
        if marker == STORE and key in KEY_TOKENS and value in VALUE_TOKENS:
            yield int(key), int(value)


def _recent_flags(history, queries, capacity):
    """Independent reporting-only membership in the last capacity write events."""
    flags = np.empty((len(history), queries.shape[2]), np.int32)
    for family, worlds in enumerate(history):
        chronology = [tuple(key for key, _ in _visible_writes(world)) for world in worlds]
        if chronology[0] != chronology[1]:
            raise ValueError('counterfactual worlds must share visible write-key chronology')
        if not np.array_equal(queries[family, 0, :, 1], queries[family, 1, :, 1]):
            raise ValueError('counterfactual worlds must share query keys')
        recent = set(chronology[0][-capacity:])
        flags[family] = [int(key in recent) for key in queries[family, 0, :, 1]]
    return flags


def _mixed_history(history, queries, seed):
    """Rotate public latest values among a family-specific four-key subset."""
    # Four stride-four writes, each preceded by filler, then exactly eight filler
    # tokens after the final value. This suffix is identical at both mixed gaps.
    tail = np.full((*history.shape[:2], 24), FILLER, dtype=history.dtype)
    updates = tail[..., :16].reshape(*history.shape[:2], 4, 4)
    updates[..., 1] = STORE
    labels = np.empty(queries.shape[:-1], np.int32)
    for family, worlds in enumerate(history):
        keys = tuple(map(int, queries[family, 0, :, 1]))
        rng = np.random.default_rng(np.random.SeedSequence((seed, family, *keys)))
        selected = rng.choice(len(keys), 4, replace=False)
        update_keys = np.asarray(keys, np.int32)[selected]
        for world, tokens in enumerate(worlds):
            latest = dict(_visible_writes(tokens))
            values = np.roll([latest[key] for key in update_keys], 1)
            updates[family, world, :, 2] = update_keys
            updates[family, world, :, 3] = values
            latest.update(zip(update_keys, values, strict=True))
            labels[family, world] = [latest[key] for key in queries[family, world, :, 1]]
    return np.concatenate((history, tail), axis=-1), labels


def panel(pairs, direct_seed=None, overwrite_seed=None, *, guard_study=False):
    """Matched queries; mixed near/far labels reflect true latest public writes."""
    direct_seed, overwrite_seed = _panel_seeds(direct_seed, overwrite_seed, guard_study)
    prefix, block, queries, labels = histories(direct_seed, pairs, 'other-writes')
    def condition(name, gap, tokens):
        return name, gap, direct_seed, tokens.reshape(pairs, 2, -1), queries, labels
    yield condition('near', 0, prefix)
    for gap in (128, 1024):
        count = gap // BLOCK
        repeated = np.tile(block, (1, count))
        yield condition('repeat', gap, np.concatenate((prefix, repeated), axis=1))
        changing = repeated.reshape(len(prefix), count, BLOCK)
        # The prefix initialized these eight non-target keys. Rotate only their eight values;
        # every subsequent write changes that key, including the first write and wraparound.
        for t in range(count):
            changing[:, t, 2::8] = np.roll(block[:, 2::8], (t + 1) % 8, axis=1)
        changing_history = np.concatenate((prefix, changing.reshape(len(prefix), gap)), axis=1)
        yield condition('change', gap, changing_history)
        if guard_study and gap == 1024:
            far_change = changing_history.reshape(pairs, 2, -1)
    filler = np.full((len(prefix), 1024), FILLER, dtype=prefix.dtype)
    yield condition('filler', 1024, np.concatenate((prefix, filler), axis=1))
    ids, labels = make_examples(overwrite_seed, 2_000_000, pairs, gap=32, writes=0, anchor=True)
    yield 'overwrite', 32, overwrite_seed, ids[:, :, 0, :-3], ids[..., -3:], labels
    if guard_study:
        for gap, history in ((0, prefix.reshape(pairs, 2, -1)), (1024, far_change)):
            mixed, mixed_labels = _mixed_history(history, queries, direct_seed)
            yield 'mixed', gap, direct_seed, mixed, queries, mixed_labels


def _pack(xs):
    return ' '.join(map(str, xs))


def measure(directory, parents, seeds, pairs, batch, capacity, direct_seed, overwrite_seed, *, guard_study=False):
    from safetensors import safe_open
    from binding_state_memory import ARMS, SnapshotReader

    direct_seed, overwrite_seed = _panel_seeds(direct_seed, overwrite_seed, guard_study)
    arms = (*ARMS, 'guarded') if guard_study else ARMS
    if not seeds or len(set(seeds)) != len(seeds) or min(pairs, batch, capacity) < 1 or direct_seed == overwrite_seed:
        raise ValueError('use unique weight seeds, distinct panel seeds and positive sizes')
    directory.mkdir(parents=True, exist_ok=False)
    columns = ('weight_seed', 'families', 'task', 'gap', 'data_seed', 'family', 'input_sha256', 'keys',
               *(('recent',) if guard_study else ()), 'truth0', 'truth1',
               *(f'{arm}{w}' for arm in arms for w in (0, 1)))
    with (directory/'readouts.csv').open('x', newline='', encoding='utf-8') as obs, \
         (directory/'weights.csv').open('x', newline='', encoding='utf-8') as metadata:
        raw = csv.writer(obs, lineterminator='\n'); raw.writerow(columns)
        costs = csv.writer(metadata, lineterminator='\n')
        costs.writerow(('weight_seed', 'checkpoint_sha256', 'parameters', 'slots', 'bytes_per_state', 'cache_state_bytes_per_history'))
        for seed in seeds:
            path = parents / f'long-s{seed}.safetensors'
            reader = SnapshotReader(path, capacity=capacity)
            with safe_open(path, framework='numpy') as weights:
                params = sum(prod(weights.get_slice(k).get_shape()) for k in weights.keys())
            costs.writerow((seed, sha256(path.read_bytes()).hexdigest(), params, capacity, reader.state_bytes, capacity*reader.state_bytes))
            for task, gap, data_seed, history, queries, labels in panel(pairs, direct_seed, overwrite_seed, guard_study=guard_study):
                for first in range(0, pairs, batch):
                    hx, qx = history[first:first+batch], queries[first:first+batch]
                    # Ground truth is used only below this call, never as model input or cache metadata.
                    predictions = reader.predict(hx, qx, include_guard=True) if guard_study else reader.predict(hx, qx)
                    recent = _recent_flags(hx, qx, capacity) if guard_study else None
                    for i in range(len(hx)):
                        digest = sha256(hx[i].astype('<i4', copy=False).tobytes()); digest.update(qx[i].astype('<i4', copy=False).tobytes())
                        family = first+i
                        raw.writerow((seed, pairs, task, gap, data_seed, family, digest.hexdigest(), _pack(qx[i, 0, :, 1]),
                                      *((_pack(recent[i]),) if guard_study else ()),
                                      *(_pack(x) for x in labels[family]), *(_pack(x) for arm in arms for x in predictions[arm][i])))
                obs.flush()
                print(f'parent {seed}, {task}, gap {gap}: {pairs} paired histories', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, help='new output directory')
    p.add_argument('--parents', type=Path, default=ROOT/'experiments/retention/parents')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--pairs', type=int, default=16); p.add_argument('--batch', type=int, default=4)
    p.add_argument('--capacity', type=int, default=16)
    p.add_argument('--guard-study', action='store_true', help='four-arm recent-write guard study with mixed-age target keys')
    p.add_argument('--direct-seed', type=int, help=f'default {DIRECT_SEED}, or {GUARD_DIRECT_SEED} with --guard-study')
    p.add_argument('--overwrite-seed', type=int, help=f'default {OVERWRITE_SEED}, or {GUARD_OVERWRITE_SEED} with --guard-study')
    args = p.parse_args()
    measure(args.directory, args.parents, args.seeds, args.pairs, args.batch, args.capacity, args.direct_seed, args.overwrite_seed,
            guard_study=args.guard_study)
