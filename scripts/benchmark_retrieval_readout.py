"""Compare full and shared-prefix evaluation on the same weights and answer-hidden inputs."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from probe_retrieval_distance import histories
from retrieval_paired_readout import _prefill_and_fork, paired_final_logits
from rwkv7m import load_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


@nnx.jit
def full_logits(model, ids, rwkv, screen):
    hidden, *_ = model.compute_recurrent_hidden(ids, rwkv, screen, phase='read_write', deterministic=True)
    return model.compute_logits(hidden[:, -1:, :])[:, 0, :].astype(jnp.float32)


def benchmark(weights, repeats):
    params, saved_cfg, _ = load_model_safetensors(weights)
    families, worlds, tokens = 8, 2, 256
    prefix, block, queries, labels = histories(12001, families, 'other-writes')
    early = np.full((families * worlds, tokens - 3 - prefix.shape[-1] - block.shape[-1]), 1, np.int32)
    shared = np.concatenate((early, prefix, block), axis=1).reshape(families, worlds, 1, -1)
    for dtype in ('bfloat16', 'float32'):
        cfg = replace(saved_cfg, dtype=dtype)
        model = initialize_nnx_model(jax.random.key(0), cfg)
        load_linen_params_into_nnx(model, params)
        for q in (2, 8):
            ids = np.concatenate((np.repeat(shared, q, axis=2), queries[:, :, :q]), axis=-1)
            full_state = init_rwkv_state(families * worlds * q, cfg), init_screen_state(families * worlds * q, cfg.screening)
            fork_state = init_rwkv_state(families * worlds, cfg), init_screen_state(families * worlds, cfg.screening)

            def full():
                return full_logits(model, jnp.asarray(ids.reshape(-1, tokens)), *full_state).reshape(families, worlds, q, -1)

            def fork():
                return paired_final_logits(model, ids, *fork_state)

            a, b = np.asarray(full()), np.asarray(fork())
            elapsed = {'full': [], 'fork': []}
            methods = [('full', full), ('fork', fork)]
            for i in range(repeats):
                for name, fn in (methods if i % 2 == 0 else reversed(methods)):
                    start = time.perf_counter()
                    jax.block_until_ready(fn())
                    elapsed[name].append(time.perf_counter() - start)
            full_exe = full_logits.lower(model, jnp.asarray(ids.reshape(-1, tokens)), *full_state).compile()
            fork_exe = _prefill_and_fork.lower(model, jnp.asarray(shared[:, :, 0]), jnp.asarray(queries[:, :, :q]),
                                               *fork_state).compile()
            def nll(logits):
                return float(-np.take_along_axis(np.asarray(jax.nn.log_softmax(logits)), labels[:, :, :q, None], -1).mean())
            yield dict(dtype=dtype, queries=q, pairs=families, tokens=tokens, answer_rows=families * worlds * q,
                       changed_top1=int(np.count_nonzero(a.argmax(-1) != b.argmax(-1))),
                       max_abs_logit_diff=float(np.max(np.abs(a - b))), nll_full=nll(a), nll_fork=nll(b),
                       full_median_ms=1000 * statistics.median(elapsed['full']),
                       fork_median_ms=1000 * statistics.median(elapsed['fork']),
                       full_temp_bytes=full_exe.memory_analysis().temp_size_in_bytes,
                       fork_temp_bytes=fork_exe.memory_analysis().temp_size_in_bytes)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=Path('experiments/retention/parents/long-s42.safetensors'))
    p.add_argument('--repeats', type=int, default=30)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be positive')
    rows = benchmark(args.weights, args.repeats)
    first = next(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(first), lineterminator='\n')
    writer.writeheader(); writer.writerow(first); writer.writerows(rows)
