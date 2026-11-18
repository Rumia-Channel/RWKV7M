"""Stream a trained normal RWKV through a controlled gap, then branch queries without feeding answers.

Distance counts intervening tokens after the last value in the initial prefix. A scored key's
own gap is 0..56 tokens longer. Both counterfactual histories use the same keys and value set.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m import load_model_safetensors
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, FILLER_TOKEN, QUERY_TOKEN, STORE_TOKEN
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


BLOCK = 64


def histories(seed, pairs, kind):
    rng = np.random.default_rng(seed)
    prefix = np.full((pairs, 2, 63), FILLER_TOKEN, np.int32)
    block = np.full((pairs, 2, BLOCK), FILLER_TOKEN, np.int32)
    queries = np.empty((pairs, 2, 8, 3), np.int32)
    labels = np.empty((pairs, 2, 8), np.int32)
    for i in range(pairs):
        keys = rng.permutation(np.arange(16, 32))
        values = rng.permutation(np.arange(32, 48))
        prefix[i, :, 0::4] = STORE_TOKEN
        prefix[i, :, 1::4] = keys
        prefix[i, :, 2::4] = values
        # Permute only the eight queried values. Every question changes its answer,
        # while unigram counts, write positions, other keys and query tokens stay fixed.
        prefix[i, 1, 6::8] = np.roll(values[1::2], 1)
        queries[i, :, :, 0] = QUERY_TOKEN
        queries[i, :, :, 1] = keys[1::2]
        queries[i, :, :, 2] = ANSWER_MARKER_TOKEN
        labels[i] = prefix[i, :, 6::8]
        if kind == 'other-writes':
            # Periodically refresh eight *other* keys; no target key/value in the gap.
            block[i, :, 0::8] = STORE_TOKEN
            block[i, :, 1::8] = keys[::2]
            block[i, :, 2::8] = values[::2]
    return prefix.reshape(-1, 63), block.reshape(-1, BLOCK), queries, labels


def bit_identical(left, right):
    return all(np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def score_answers(log_probs, labels):
    finite = np.isfinite(log_probs).all(axis=-1)
    predictions = np.where(finite, log_probs.argmax(axis=-1), -1)
    correct = predictions == labels
    finite_pairs = finite.all(axis=1)
    changed = predictions[:, 0] != predictions[:, 1]
    nll = -np.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return {'correct': int(correct.sum()), 'questions': int(labels.size),
            'nonfinite_questions': int(np.count_nonzero(~finite)),
            'both_worlds_correct': int(correct.all(axis=1).sum()), 'key_pairs': int(labels.size // 2),
            'prediction_changes': int(np.count_nonzero(changed & finite_pairs)),
            'finite_key_pairs': int(finite_pairs.sum()),
            'mean_nll': float(nll.mean(dtype=np.float64)) if finite.all() else None,
            'predictions': predictions.tolist()}  # -1 means no finite distribution, never a correct answer


def probe(weights, seed, pairs, kind, gaps, compute_dtype=None):
    source_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    params, cfg, _ = load_model_safetensors(weights)
    if cfg.use_screening or cfg.vocab_size != 128:
        raise ValueError('expected a normal RWKV synthetic retrieval checkpoint with vocabulary 128')
    # No backward pass: do not wrap layers in the training-only NNX remat transform.
    execution_cfg = replace(cfg, remat_blocks=False, dtype=compute_dtype or cfg.dtype)
    model = initialize_nnx_model(jax.random.key(0), execution_cfg)
    load_linen_params_into_nnx(model, params)
    prefix, block, queries, labels = histories(seed, pairs, kind)
    gap_block = jnp.asarray(block)
    flat_queries = jnp.asarray(queries.reshape(-1, 3))

    def advance(ids, state):
        _, rwkv, screen, _ = model.compute_recurrent_hidden(ids, *state, phase='read_write', deterministic=True)
        return rwkv, screen

    @jax.jit
    def repeat(state, count):
        return jax.lax.fori_loop(0, count, lambda _, s: advance(gap_block, s), state)

    @jax.jit
    def score(state):
        copies = jax.tree.map(lambda a: jnp.repeat(a, queries.shape[2], axis=0), state)
        hidden, _, _, _ = model.compute_recurrent_hidden(flat_queries, *copies, phase='read_write', deterministic=True)
        logits = model.compute_logits(hidden[:, -1:, :])[:, 0].astype(jnp.float32)
        return jax.nn.log_softmax(logits).reshape(*labels.shape, cfg.vocab_size)

    state = jax.jit(advance)(jnp.asarray(prefix), (init_rwkv_state(len(prefix), cfg),
                                                  init_screen_state(len(prefix), cfg.screening)))
    at = executed = 0  # measured in 64-token blocks
    period = None
    certificate = None
    results = []
    for gap in gaps:
        target = gap // BLOCK
        while at < target:
            if period and target - at >= period:
                # F^p(s) == s byte-for-byte. F is the same deterministic full-model
                # block at every step; skip only whole periods, never a residual.
                at += (target - at) // period * period
                continue
            n = min(64, target - at)
            old = state
            state = repeat(state, jnp.int32(n))
            at += n
            executed += n
            if period is None and bit_identical(old, state):
                period = n
                certificate = {'from_gap': (at - n) * BLOCK, 'to_gap': at * BLOCK, 'period_tokens': n * BLOCK}
        row = score_answers(np.asarray(score(state)), labels)
        row.update(gap=gap, physical_gap_tokens_per_history=executed * BLOCK,
                   cycle_skipped_tokens_per_history=gap - executed * BLOCK,
                   nonfinite_wkv_lanes_by_layer=[
                       np.flatnonzero(~np.isfinite(np.asarray(layer.wkv)).reshape(len(prefix), -1).all(axis=1)).tolist()
                       for layer in state[0]])
        print(f"{weights.parent.parent.name} {kind} gap={gap}: {row['both_worlds_correct']}/{row['key_pairs']} paired, "
              f"{row['correct']}/{row['questions']} individual, {row['nonfinite_questions']} nonfinite; "
              f"{row['cycle_skipped_tokens_per_history']} cycle-skipped", flush=True)
        results.append(row)
    return {'weights': str(weights), 'weights_sha256': sha256(weights.read_bytes()).hexdigest(),
            'probe_sha256': source_hash, 'jax': jax.__version__, 'device': str(jax.devices()[0]),
            'activation_dtype': execution_cfg.dtype, 'checkpoint_activation_dtype': cfg.dtype, 'wkv_state_dtype': 'float32',
            'data_seed': seed, 'history_pairs': pairs, 'gap_kind': kind, 'block_tokens': BLOCK,
            'prefixes': prefix.reshape(pairs, 2, -1).tolist(), 'gap_blocks': block.reshape(pairs, 2, -1).tolist(),
            'query_keys': queries[:, 0, :, 1].tolist(), 'labels': labels.tolist(),
            'extra_gap_by_key': list(range(56, -1, -8)), 'exact_state_cycle': certificate, 'results': results}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--gap-kind', choices=('filler', 'other-writes'), required=True)
    p.add_argument('--compute-dtype', choices=('bfloat16', 'float32'), help='default: checkpoint activation dtype')
    p.add_argument('--data-seed', type=int, default=9501)
    p.add_argument('--history-pairs', type=int, default=1)
    p.add_argument('--gaps', type=int, nargs='+', default=[0, 1024, 4096, 16_384, 65_536, 262_144, 1_000_000, 10_000_000])
    a = p.parse_args()
    if a.history_pairs < 1 or a.gaps != sorted(set(a.gaps)) or any(g < 0 or g % BLOCK for g in a.gaps):
        p.error('supply positive history pairs and strictly increasing nonnegative gaps divisible by 64')
    r = probe(a.weights, a.data_seed, a.history_pairs, a.gap_kind, a.gaps, a.compute_dtype)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + '\n', encoding='utf-8')
