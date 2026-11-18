"""Blind value-token reads from bounded post-STORE neural checkpoints.

All arms use one frozen ordinary synthetic RWKV. The caches hold complete
neural states plus public keys and bounded replacement metadata, never values
or model answers. A query forks a state; reading never updates a cache.
"""

from dataclasses import replace
from math import gcd
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m import load_model_safetensors
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, QUERY_TOKEN, STORE_TOKEN
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


ARMS = ('live', 'fifo', 'keyed')
_BLOCK = 64


class _Fifo(NamedTuple):
    states: object
    keys: jax.Array
    cursor: jax.Array


class _Recent(NamedTuple):
    keys: jax.Array
    cursor: jax.Array


class _Keyed(NamedTuple):
    states: object
    keys: jax.Array
    ranks: jax.Array  # 0 = newest distinct key; bounded by capacity - 1.


def _state_bytes(state):
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(state))


def _empty_banks(state, capacity):
    recent, keyed = _empty_guarded_banks(state, capacity)
    # Sharing immutable empty leaves is safe; the two replacement policies diverge.
    return _Fifo(keyed.states, recent.keys, recent.cursor), keyed


def _empty_guarded_banks(state, capacity):
    batch = jax.tree.leaves(state)[0].shape[0]
    states = jax.tree.map(
        lambda x: jnp.zeros((batch, capacity, *x.shape[1:]), dtype=x.dtype), state,
    )
    keys = jnp.full((batch, capacity), -1, jnp.int32)
    return (_Recent(keys, jnp.zeros(batch, jnp.int32)),
            _Keyed(states, keys, jnp.zeros((batch, capacity), jnp.int32)))


def _put(states, live, slots, valid):
    rows = jnp.arange(slots.shape[0])

    def put(bank, value):
        mask = valid.reshape((len(rows),) + (1,) * (value.ndim - 1))
        return bank.at[rows, slots].set(jnp.where(mask, value, bank[rows, slots]))

    return jax.tree.map(put, states, live)


def _commit_recent(recent, keys, valid):
    rows = jnp.arange(keys.shape[0])
    tags = recent.keys.at[rows, recent.cursor].set(
        jnp.where(valid, keys, recent.keys[rows, recent.cursor]),
    )
    cursor = jnp.where(valid, (recent.cursor + 1) % recent.keys.shape[1], recent.cursor)
    return _Recent(tags, cursor)


def _commit_keyed(live, keyed, keys, valid):
    rows = jnp.arange(keys.shape[0])
    capacity = keyed.keys.shape[1]
    matches = keyed.keys == keys[:, None]
    found = matches.any(axis=1)
    empty = keyed.keys < 0
    slots = jnp.where(
        found, matches.argmax(axis=1),
        jnp.where(empty.any(axis=1), empty.argmax(axis=1), keyed.ranks.argmax(axis=1)),
    )
    # Move the written key to rank zero without an unbounded timestamp counter.
    old_rank = jnp.where(found, keyed.ranks[rows, slots], capacity)
    ranks = keyed.ranks + (
        valid[:, None] & (keyed.keys >= 0) & (keyed.ranks < old_rank[:, None])
    ).astype(jnp.int32)
    ranks = ranks.at[rows, slots].set(jnp.where(valid, 0, ranks[rows, slots]))
    keyed_keys = keyed.keys.at[rows, slots].set(
        jnp.where(valid, keys, keyed.keys[rows, slots]),
    )
    return _Keyed(_put(keyed.states, live, slots, valid), keyed_keys, ranks)


@jax.jit
def _commit(live, fifo, keyed, keys):
    """Commit already-computed states; -1 means this lane had no full event."""
    valid = (keys >= 16) & (keys < 32)
    recent = _commit_recent(_Recent(fifo.keys, fifo.cursor), keys, valid)
    return (_Fifo(_put(fifo.states, live, fifo.cursor, valid), recent.keys, recent.cursor),
            _commit_keyed(live, keyed, keys, valid))


@jax.jit
def _commit_guarded(live, recent, keyed, keys):
    valid = (keys >= 16) & (keys < 32)
    return _commit_recent(recent, keys, valid), _commit_keyed(live, keyed, keys, valid)


def _fork_states(live, fifo, keyed, query_keys):
    """Return arm-major independent query rows, falling back to live on misses."""
    batch, questions = query_keys.shape
    rows = jnp.arange(batch)[:, None]
    capacity = fifo.keys.shape[1]
    matches = fifo.keys[:, None, :] == query_keys[:, :, None]
    distance = (fifo.cursor[:, None] - 1 - jnp.arange(capacity)[None, :]) % capacity
    fifo_slots = jnp.where(matches, distance[:, None, :], capacity).argmin(axis=2)
    fifo_hits = matches.any(axis=2)
    matches = keyed.keys[:, None, :] == query_keys[:, :, None]
    keyed_slots, keyed_hits = matches.argmax(axis=2), matches.any(axis=2)

    def fork(current, fifo_bank, keyed_bank):
        shape = (batch * questions, *current.shape[1:])
        mask_shape = (batch, questions) + (1,) * (current.ndim - 1)
        current = jnp.broadcast_to(current[:, None], (batch, questions, *current.shape[1:]))
        fifo_values = jnp.where(fifo_hits.reshape(mask_shape), fifo_bank[rows, fifo_slots], current)
        keyed_values = jnp.where(keyed_hits.reshape(mask_shape), keyed_bank[rows, keyed_slots], current)
        return jnp.concatenate([x.reshape(shape) for x in (current, fifo_values, keyed_values)], axis=0)

    return jax.tree.map(fork, live, fifo.states, keyed.states)


def _recent_hits(recent, query_keys):
    return (recent.keys[:, None, :] == query_keys[:, :, None]).any(axis=2)


def _fork_guarded_states(live, recent, keyed, query_keys):
    """Route every query independently, without creating unused arm states."""
    batch, questions = query_keys.shape
    rows = jnp.arange(batch)[:, None]
    matches = keyed.keys[:, None, :] == query_keys[:, :, None]
    slots = matches.argmax(axis=2)
    use_cache = (matches.any(axis=2) & ~_recent_hits(recent, query_keys)
                 & (query_keys >= 16) & (query_keys < 32))

    def fork(current, bank):
        mask_shape = (batch, questions) + (1,) * (current.ndim - 1)
        current = jnp.broadcast_to(current[:, None], (batch, questions, *current.shape[1:]))
        selected = jnp.where(use_cache.reshape(mask_shape), bank[rows, slots], current)
        return selected.reshape(batch * questions, *current.shape[2:])

    return jax.tree.map(fork, live, keyed.states)


def _commit_grid(history):
    """Find a common write-boundary grid using at most 64 positions of scratch.

    This only schedules computation. No future token enters a checkpoint: the
    recurrent loop commits only after consuming the event's value token.
    Nonsynchronous lanes are supported, down to a one-token device-side loop.
    """
    first = last = None
    spacing = 0
    for start in range(2, history.shape[1], _BLOCK):
        stop = min(start + _BLOCK, history.shape[1])
        key = history[:, start - 1:stop - 1]
        value = history[:, start:stop]
        events = ((history[:, start - 2:stop - 2] == STORE_TOKEN)
                  & (key >= 16) & (key < 32) & (value >= 32) & (value < 48))
        for offset in np.flatnonzero(events.any(axis=0)):
            end = start + int(offset) + 1
            if first is None:
                first = end
            else:
                spacing = gcd(spacing, end - last)
            last = end
    stride = next(n for n in range(_BLOCK, 0, -1) if not spacing or spacing % n == 0)
    return first, last, stride


@nnx.jit(static_argnames=('block_size', 'cache_writes', 'guarded'))
def _advance_blocks(model, history, live, fifo, keyed, start, stop, *,
                    block_size, cache_writes, guarded=False):
    """One bounded device loop: no timestep outputs or prefix-state tape."""
    commit = _commit_guarded if guarded else _commit

    def step(i, carry):
        state, fifo, keyed = carry
        begin = start + i * block_size
        ids = jax.lax.dynamic_slice_in_dim(history, begin, block_size, axis=1)
        _, rwkv, screen, _ = model.compute_recurrent_hidden(
            ids, *state, phase='read_write', deterministic=True,
        )
        state = rwkv, screen
        if cache_writes:
            end = begin + block_size
            event = jax.lax.dynamic_slice_in_dim(history, end - 3, 3, axis=1)
            valid = ((event[:, 0] == STORE_TOKEN)
                     & (event[:, 1] >= 16) & (event[:, 1] < 32)
                     & (event[:, 2] >= 32) & (event[:, 2] < 48))
            fifo, keyed = commit(state, fifo, keyed, jnp.where(valid, event[:, 1], -1))
        return state, fifo, keyed

    return jax.lax.fori_loop(0, (stop - start) // block_size, step, (live, fifo, keyed))


def _predictions(logits):
    # A finite argmax must not conceal a NaN/Inf elsewhere in its logit vector.
    finite = jnp.isfinite(logits).all(axis=-1)
    tokens = logits.argmax(axis=-1).astype(jnp.int32)
    return jnp.where(finite, tokens, jnp.int32(-1))


@nnx.jit
def _read_queries(model, queries, live, fifo, keyed):
    batch, questions, _ = queries.shape
    state = _fork_states(live, fifo, keyed, queries[:, :, 1])
    ids = jnp.tile(queries.reshape(-1, 3), (len(ARMS), 1))
    hidden, _, _, _ = model.compute_recurrent_hidden(
        ids, *state, phase='read_write', deterministic=True,
    )
    logits = model.compute_logits(hidden[:, -1:, :])[:, 0, :].astype(jnp.float32)
    return _predictions(logits).reshape(len(ARMS), batch, questions)


@nnx.jit
def _read_guarded_queries(model, queries, live, recent, keyed):
    batch, questions, _ = queries.shape
    state = _fork_guarded_states(live, recent, keyed, queries[:, :, 1])
    hidden, _, _, _ = model.compute_recurrent_hidden(
        queries.reshape(-1, 3), *state, phase='read_write', deterministic=True,
    )
    logits = model.compute_logits(hidden[:, -1:, :])[:, 0, :].astype(jnp.float32)
    return _predictions(logits).reshape(batch, questions)


class SnapshotReader:
    """Read one frozen synthetic RWKV checkpoint without labels or a fitted gate.

    Predictions are unrestricted-head argmax token IDs for every fully finite
    logit vector; only nonfinite vectors become -1. Comparative reads retain
    (1 + 2 * capacity) complete neural states per lane. Standalone guarded reads
    retain one live state and one capacity-slot keyed state bank: capacity * S
    cache bytes plus (3 * capacity + 1) int32 keys/ranks/recent-ring metadata.
    Recent means membership in the last capacity completed writes, not distinct
    keys; recent queries use live, others use keyed with live fallback on misses.
    Prefill workspace is bounded at 64 tokens; independent query workspace
    scales with the number of requested questions.
    """

    def __init__(self, weights: Path, capacity: int = 16):
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        params, config, _ = load_model_safetensors(Path(weights))
        if config is None or config.use_screening or config.vocab_size != 128:
            raise ValueError('expected an ordinary non-screening synthetic RWKV with vocabulary 128')
        self.capacity = int(capacity)
        self._config = replace(config, remat_blocks=False)
        self._model = initialize_nnx_model(jax.random.key(0), self._config)
        load_linen_params_into_nnx(self._model, params)
        self.state_bytes = _state_bytes(self._initial(1))

    def _initial(self, batch):
        return init_rwkv_state(batch, self._config), init_screen_state(batch, self._config.screening)

    def _prefill(self, history, *, guarded=False):
        live = self._initial(history.shape[0])
        fifo, keyed = (_empty_guarded_banks if guarded else _empty_banks)(live, self.capacity)
        first, last, stride = _commit_grid(history)
        ids = jnp.asarray(history)

        def advance(start, stop, block_size, cache_writes):
            nonlocal live, fifo, keyed
            if stop > start:
                live, fifo, keyed = _advance_blocks(
                    self._model, ids, live, fifo, keyed, jnp.int32(start), jnp.int32(stop),
                    block_size=block_size, cache_writes=cache_writes, guarded=guarded,
                )

        def plain(start, stop, cache_writes=False):
            boundary = start + (stop - start) // _BLOCK * _BLOCK
            advance(start, boundary, _BLOCK, cache_writes)
            advance(boundary, stop, stop - boundary, cache_writes)

        if first is None:
            plain(0, history.shape[1])
        else:
            plain(0, first, True)
            advance(first, last, stride, True)
            plain(last, history.shape[1])
        return live, fifo, keyed

    def _inputs(self, history, queries):
        history, queries = np.asarray(history), np.asarray(queries)
        if (history.ndim != 3 or history.shape[0] < 1 or history.shape[1] != 2
                or queries.ndim != 4 or queries.shape[:2] != history.shape[:2]
                or queries.shape[2] < 1 or queries.shape[3] != 3):
            raise ValueError('expected history [family,2,time] and queries [family,2,question,3]')
        if history.dtype != np.int32 or queries.dtype != np.int32:
            raise ValueError('history and queries must have dtype int32')
        if np.any((history < 0) | (history >= self._config.vocab_size)):
            raise ValueError('history contains an out-of-vocabulary token')
        if not (np.all(queries[..., 0] == QUERY_TOKEN)
                and np.all(queries[..., 2] == ANSWER_MARKER_TOKEN)
                and np.all((queries[..., 1] >= 16) & (queries[..., 1] < 32))):
            raise ValueError('expected raw [QUERY, key 16..31, ANSWER_MARKER] queries')
        return history, queries

    def predict(self, history, queries, *, include_guard=False):
        """Compare raw [family,2,time] histories and [family,2,question,3] queries.

        The default three-arm output is unchanged. The optional guarded arm
        selects already-evaluated deterministic live/keyed predictions per key.
        """
        history, queries = self._inputs(history, queries)
        families, worlds, questions, _ = queries.shape
        flat = history.reshape(families * worlds, history.shape[2])
        states = self._prefill(flat)
        flat_queries = jnp.asarray(queries.reshape(-1, questions, 3))
        predictions = np.asarray(_read_queries(
            self._model, flat_queries, *states,
        ), dtype=np.int32).reshape(len(ARMS), families, worlds, questions)
        result = dict(zip(ARMS, predictions, strict=True))
        if include_guard:
            recent = np.asarray(_recent_hits(states[1], flat_queries[:, :, 1]))
            result['guarded'] = np.where(
                recent.reshape(families, worlds, questions), result['live'], result['keyed'],
            )
        return result

    def predict_guarded(self, history, queries):
        """Return int32 [family,2,question] using one routed neural read/head.

        No FIFO neural payload or unused-arm query state is allocated here.
        """
        history, queries = self._inputs(history, queries)
        families, worlds, questions, _ = queries.shape
        flat = history.reshape(families * worlds, history.shape[2])
        states = self._prefill(flat, guarded=True)
        return np.asarray(_read_guarded_queries(
            self._model, jnp.asarray(queries.reshape(-1, questions, 3)), *states,
        ), dtype=np.int32).reshape(families, worlds, questions)
