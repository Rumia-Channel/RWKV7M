"""Run frozen RWKV on visible writes; compare live state, recent snapshots, and keyed snapshots."""
import argparse
import csv
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np

from probe_retrieval_distance import BLOCK, histories
from retrieval_frontier import FILLER, make_examples


DIRECT_SEED = 731907
OVERWRITE_SEED = 731908
ROOT = Path(__file__).resolve().parents[1]


def panel(pairs, direct_seed=DIRECT_SEED, overwrite_seed=OVERWRITE_SEED):
    """Same queried keys and world-specific answers in every direct condition. No label in a query."""
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
        yield condition('change', gap, np.concatenate((prefix, changing.reshape(len(prefix), gap)), axis=1))
    filler = np.full((len(prefix), 1024), FILLER, dtype=prefix.dtype)
    yield condition('filler', 1024, np.concatenate((prefix, filler), axis=1))
    ids, labels = make_examples(overwrite_seed, 2_000_000, pairs, gap=32, writes=0, anchor=True)
    yield 'overwrite', 32, overwrite_seed, ids[:, :, 0, :-3], ids[..., -3:], labels


def _pack(xs):
    return ' '.join(map(str, xs))


def measure(directory, parents, seeds, pairs, batch, capacity, direct_seed, overwrite_seed):
    from safetensors import safe_open
    from binding_state_memory import ARMS, SnapshotReader

    if not seeds or len(set(seeds)) != len(seeds) or min(pairs, batch, capacity) < 1 or direct_seed == overwrite_seed:
        raise ValueError('use unique weight seeds, distinct panel seeds and positive sizes')
    directory.mkdir(parents=True, exist_ok=False)
    columns = ('weight_seed', 'families', 'task', 'gap', 'data_seed', 'family', 'input_sha256', 'keys',
               'truth0', 'truth1', *(f'{arm}{w}' for arm in ARMS for w in (0, 1)))
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
            for task, gap, data_seed, history, queries, labels in panel(pairs, direct_seed, overwrite_seed):
                for first in range(0, pairs, batch):
                    hx, qx = history[first:first+batch], queries[first:first+batch]
                    # Ground truth is used only below this call, never as model input or cache metadata.
                    predictions = reader.predict(hx, qx)
                    for i in range(len(hx)):
                        digest = sha256(hx[i].astype('<i4', copy=False).tobytes()); digest.update(qx[i].astype('<i4', copy=False).tobytes())
                        family = first+i
                        raw.writerow((seed, pairs, task, gap, data_seed, family, digest.hexdigest(), _pack(qx[i, 0, :, 1]),
                                      *(_pack(x) for x in labels[family]), *(_pack(x) for arm in ARMS for x in predictions[arm][i])))
                obs.flush()
                print(f'parent {seed}, {task}, gap {gap}: {pairs} paired histories', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, help='new output directory')
    p.add_argument('--parents', type=Path, default=ROOT/'experiments/retention/parents')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--pairs', type=int, default=16); p.add_argument('--batch', type=int, default=4)
    p.add_argument('--capacity', type=int, default=16)
    p.add_argument('--direct-seed', type=int, default=DIRECT_SEED); p.add_argument('--overwrite-seed', type=int, default=OVERWRITE_SEED)
    args = p.parse_args()
    measure(args.directory, args.parents, args.seeds, args.pairs, args.batch, args.capacity, args.direct_seed, args.overwrite_seed)
