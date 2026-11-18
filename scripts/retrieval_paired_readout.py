"""Score answer-hidden query branches without replaying the common history for each key."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, QUERY_TOKEN


def branched_final_logits(model, prefix, queries, rwkv_state, screen_state):
    """Read one history per family/world; fork only for the final query and head."""
    families, worlds, questions, suffix = queries.shape
    _, rwkv_state, screen_state, _ = model.compute_recurrent_hidden(
        prefix.reshape(families * worlds, -1), rwkv_state, screen_state,
        phase='read_write', deterministic=True,
    )
    rwkv_state, screen_state = jax.tree.map(lambda x: jnp.repeat(x, questions, axis=0), (rwkv_state, screen_state))
    hidden, _, _, _ = model.compute_recurrent_hidden(
        queries.reshape(-1, suffix), rwkv_state, screen_state, phase='read_write', deterministic=True,
    )
    logits = model.compute_logits(hidden[:, -1:, :])[:, 0, :]
    return logits.astype(jnp.float32).reshape(families, worlds, questions, -1)


# Lift the graph only at the inference boundary, not inside another training transform.
compiled_branched_logits = nnx.jit(branched_final_logits)


def paired_final_logits(model, ids, rwkv_state, screen_state):
    """Use host-generated [family,world,question,time] inputs; initial states have family*world rows.

    Never silently discard a different prefix. Only transfer the shared prefix and the
    three-token suffixes; callers should not device-put the expanded host batch first.
    """
    if ids.ndim != 4 or ids.shape[2] < 1 or ids.shape[-1] <= 3:
        raise ValueError('expected paired histories ending in a three-token query')
    prefix, queries = ids[:, :, 0, :-3], ids[..., -3:]
    for q in range(1, ids.shape[2]):
        if not np.array_equal(prefix, ids[:, :, q, :-3]):
            raise ValueError('query branches do not share a prefix')
    if not (np.all(queries[..., 0] == QUERY_TOKEN) and np.all(queries[..., 2] == ANSWER_MARKER_TOKEN)):
        raise ValueError('expected [QUERY, key, ANSWER_MARKER] at every scored position')
    return compiled_branched_logits(model, jnp.asarray(prefix), jnp.asarray(queries), rwkv_state, screen_state)
