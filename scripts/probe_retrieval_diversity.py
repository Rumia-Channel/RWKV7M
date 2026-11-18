"""Hold distance and number of writes fixed; vary how many unqueried keys are refreshed."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import jax
import numpy as np

from probe_retrieval_distance import histories
from retrieval_paired_readout import paired_final_logits
from rwkv7m import load_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


GAP = 128
DIVERSITIES = (1, 2, 4, 8)


def query_batch(prefix, block, queries, distinct):
    """Cycle through n distinct unqueried facts; balance the starting prefix slot by family."""
    pairs = len(queries)
    if distinct not in DIVERSITIES:
        raise ValueError('distinct must divide the eight available distractor keys')
    # Histories use an eight-token STORE,key,value,filler... record for each unrelated fact.
    facts = block.reshape(pairs, 2, 8, 8)
    slots = (np.arange(pairs)[:, None] + np.arange(GAP // 8)[None, :] % distinct) % 8
    gap = np.take_along_axis(facts, slots[:, None, :, None], axis=2).reshape(pairs, 2, GAP)
    ids = np.empty((*queries.shape[:-1], 63 + GAP + queries.shape[-1]), dtype=np.int32)
    ids[..., :63] = prefix.reshape(pairs, 2, 1, 63)
    ids[..., 63:-3] = gap[:, :, None, :]
    ids[..., -3:] = queries
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', type=Path, default=Path('experiments/retention/parents'))
    parser.add_argument('--seed', type=int, default=26519)
    parser.add_argument('--pairs', type=int, default=64)
    args = parser.parse_args()
    if args.pairs <= 0 or args.pairs % 8:
        parser.error('pairs must contain complete groups of eight counterbalanced starting slots')
    prefix, block, queries, labels = histories(args.seed, args.pairs, 'other-writes')
    out = csv.writer(sys.stdout, lineterminator='\n')
    out.writerow(('weight_seed', 'data_seed', 'gap', 'writes', 'distinct', 'family', 'both_correct', 'key_pairs', 'nonfinite'))
    for seed in (42, 43, 44):
        params, cfg, _ = load_model_safetensors(args.parents / f'long-s{seed}.safetensors')
        model = initialize_nnx_model(jax.random.key(0), replace(cfg, remat_blocks=False))
        load_linen_params_into_nnx(model, params)
        # Eight families per scoring batch, two independent recurrent histories per family.
        state = init_rwkv_state(16, cfg), init_screen_state(16, cfg.screening)
        for distinct in DIVERSITIES:
            ids = query_batch(prefix, block, queries, distinct)
            for start in range(0, len(ids), 8):
                sl = slice(start, start + 8)
                logits = np.asarray(paired_final_logits(model, ids[sl], *state))
                valid = np.isfinite(logits).all(axis=-1)
                correct = (logits.argmax(axis=-1) == labels[sl]) & valid
                for i, (hits, invalid) in enumerate(zip(correct.all(axis=1).sum(axis=1), (~valid).sum(axis=(1, 2)), strict=True)):
                    out.writerow((seed, args.seed, GAP, GAP // 8, distinct, start + i, hits, labels.shape[-1], invalid))


if __name__ == '__main__':
    main()
