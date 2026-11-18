"""Score frozen RWKV weights on identical, calibration-independent histories; never feed answers."""

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import time

import jax
import numpy as np

from probe_retrieval_distance import histories
from retrieval_frontier import make_examples
from retrieval_paired_readout import paired_final_logits
from rwkv7m import load_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


DIRECT_SEED = 96173
OVERWRITE_SEED = 96174
GAPS = (0, 64, 128, 256, 1024)
COLUMNS = ('task', 'gap', 'arm', 'weight_seed', 'family', 'both_correct', 'key_pairs', 'nonfinite',
           'data_seed', 'input_sha256', 'world0_labels', 'world1_labels', 'world0_predictions', 'world1_predictions')


def panel(pairs, direct_seed=DIRECT_SEED, overwrite_seed=OVERWRITE_SEED):
    for kind in ('filler', 'other-writes'):
        prefix, block, queries, labels = histories(direct_seed, pairs, kind)
        for gap in GAPS:
            ids = np.empty((*queries.shape[:-1], 63 + gap + 3), dtype=np.int32)
            ids[..., :63] = prefix.reshape(pairs, 2, 1, 63)
            if gap:
                ids[..., 63:-3] = np.tile(block, (1, gap // block.shape[-1])).reshape(pairs, 2, 1, gap)
            ids[..., -3:] = queries
            yield kind, gap, direct_seed, ids, labels
    ids, labels = make_examples(overwrite_seed, 2_000_000, pairs, gap=32, writes=0, anchor=True)
    yield 'overwrite', 32, overwrite_seed, ids, labels


def score(models, output, pairs=32, *, direct_seed=DIRECT_SEED, overwrite_seed=OVERWRITE_SEED):
    # Use the same histories and raw row order for every model; no adaptive controller is imported here.
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n'); writer.writerow(COLUMNS)
        for arm, seed, weights in models:
            start = time.monotonic()
            params, cfg, _ = load_model_safetensors(weights)
            if cfg.use_screening or cfg.vocab_size != 128:
                raise ValueError(f'not an ordinary retrieval RWKV: {weights}')
            model = initialize_nnx_model(jax.random.key(0), replace(cfg, remat_blocks=False))
            load_linen_params_into_nnx(model, params)
            state = init_rwkv_state(16, cfg), init_screen_state(16, cfg.screening)
            for task, gap, data_seed, ids, labels in panel(pairs, direct_seed, overwrite_seed):
                for first in range(0, pairs, 8):
                    batch = ids[first:first + 8]
                    logits = np.asarray(paired_final_logits(model, batch, *state))
                    valid = np.isfinite(logits).all(axis=-1)
                    pred = np.where(valid, logits.argmax(axis=-1), -1)
                    for i, y in enumerate(labels[first:first + 8]):
                        pr = pred[i]
                        vectors = (' '.join(map(str, a)) for a in (*y, *pr))
                        writer.writerow((task, gap, arm, seed, first + i, int((pr == y).all(axis=0).sum()),
                                         y.shape[-1], int((~valid[i]).sum()), data_seed,
                                         sha256(batch[i].astype('<i4', copy=False).tobytes()).hexdigest(), *vectors))
            stream.flush()
            print(f'{arm} seed={seed}: {pairs} families per condition in {time.monotonic()-start:.2f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--model', nargs=3, action='append', required=True, metavar=('ARM', 'SEED', 'WEIGHTS'))
    p.add_argument('--pairs', type=int, default=32)
    p.add_argument('--direct-seed', type=int, default=DIRECT_SEED)
    p.add_argument('--overwrite-seed', type=int, default=OVERWRITE_SEED)
    a = p.parse_args()
    if a.pairs <= 0 or a.pairs % 8:
        p.error('pairs must be a positive multiple of the eight-family scoring batch')
    models = [(arm, int(seed), Path(path)) for arm, seed, path in a.model]
    if len({(arm, seed) for arm, seed, _ in models}) != len(models):
        p.error('duplicate arm and seed')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    score(models, a.output, a.pairs, direct_seed=a.direct_seed, overwrite_seed=a.overwrite_seed)
