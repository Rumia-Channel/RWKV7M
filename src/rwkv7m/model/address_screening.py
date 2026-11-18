"""FP32 delta memory for explicit address events, independent of grammar or logits.

Read ``q[d] @ M``; write ``M += outer(u[d], value - q[d] @ M)``. Coupled
addresses share directions (``u=q``); dual addresses satisfy ``q @ u.T = I``
in exact arithmetic. Isolation in FP32 depends on conditioning and write history.
COPY snapshots its source before updating its destination; an unbound source
clears the destination and a self-copy is a bitwise no-op. NOP or invalid
events leave their lane untouched. Only the matrix and bound flags persist.
Reverse-mode differentiation can retain residuals per event during training.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


EVENT_NOP = 0
EVENT_LITERAL = 1
EVENT_COPY = 2

ADDRESS_MODES = ('isolated', 'coupled')

#: Fixed geometry gate: the ratio of the largest to the smallest singular
#: value of the row-normalised key embedding matrix must not exceed it.
MAX_CONDITION_NUMBER = 1e4

__all__ = [
    'ADDRESS_MODES',
    'AddressMemory',
    'EVENT_COPY',
    'EVENT_LITERAL',
    'EVENT_NOP',
    'MAX_CONDITION_NUMBER',
    'apply_events',
    'apply_events_jit',
    'initial_memory',
    'make_address_codes',
    'read_memory',
    'read_memory_jit',
]


class AddressMemory(NamedTuple):
    """Delta memory state.

    Attributes:
        matrix: ``[B, K, V]`` float32 feature/value matrix.
        bound: ``[B, K]`` bool flag, true when the address holds a value.
    """

    matrix: jnp.ndarray
    bound: jnp.ndarray


def _checked_size(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer, got {type(value).__name__}')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value}')
    return value


def make_address_codes(embeddings, mode: str = 'isolated'):
    """Build the read codes ``Q`` and write codes ``U`` for a set of keys.

    Args:
        embeddings: ``[K, H]`` key embeddings.  Only the row directions
            matter; every row is normalised to unit length.
        mode: ``'isolated'`` (private direction per address) or ``'coupled'``
            (one shared direction).

    Returns:
        ``(Q, U)``: two ``[K, K]`` float32 arrays.  In ``'isolated'`` mode
        ``Q @ U.T`` is the identity up to float32 rounding; in ``'coupled'``
        mode ``U`` equals ``Q``.

    Raises:
        ValueError: the geometry is rank deficient or ill conditioned, the
            mode is unknown, or the embeddings are not a finite ``[K, H]``
            array with non-zero rows.
    """
    if mode not in ADDRESS_MODES:
        raise ValueError(
            f'unknown address mode {mode!r}; expected one of {list(ADDRESS_MODES)}'
        )
    emb = np.asarray(embeddings)
    if emb.ndim != 2:
        raise ValueError(f'embeddings must be a 2-D [K, H] array, got shape {emb.shape}')
    key_count, hidden = emb.shape
    if key_count < 1 or hidden < 1:
        raise ValueError(
            f'embeddings must hold at least one key and one feature, got shape {emb.shape}'
        )
    x = np.asarray(emb, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('embeddings must be finite')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError('every key embedding must have a non-zero norm')
    x = x / norms

    # Thin SVD of the row-normalised embeddings gives K-dimensional feature
    # coordinates that keep the cosine Gram matrix of the rows.
    left, singular, _ = np.linalg.svd(x, full_matrices=False)
    if singular.size < key_count:
        raise ValueError(
            'key embeddings are rank deficient: '
            f'{key_count} keys need {key_count} independent directions but the '
            f'hidden dimension is only {hidden}'
        )
    smallest = float(singular[-1])
    if smallest <= 0.0:
        raise ValueError('key embeddings are rank deficient: zero singular value')
    condition = float(singular[0]) / smallest
    if condition > MAX_CONDITION_NUMBER:
        raise ValueError(
            f'key embeddings are ill conditioned: cond={condition:.6e} exceeds the '
            f'fixed limit {MAX_CONDITION_NUMBER:.0e}'
        )

    coords = left * singular  # [K, K] float64 feature coordinates
    if mode == 'isolated':
        try:
            inverse = np.linalg.solve(coords, np.eye(key_count, dtype=np.float64))
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                'key embeddings are rank deficient; address codes are undefined'
            ) from exc
        write = inverse.T  # Q @ U.T == I
    else:
        write = coords  # U == Q
    return (
        np.ascontiguousarray(coords, dtype=np.float32),
        np.ascontiguousarray(write, dtype=np.float32),
    )


def initial_memory(batch: int, key_count: int, value_dim: int) -> AddressMemory:
    """Return an empty FP32 memory with no bound addresses."""
    batch = _checked_size(batch, 'batch', minimum=0)
    key_count = _checked_size(key_count, 'key_count', minimum=1)
    value_dim = _checked_size(value_dim, 'value_dim', minimum=1)
    return AddressMemory(
        matrix=jnp.zeros((batch, key_count, value_dim), dtype=jnp.float32),
        bound=jnp.zeros((batch, key_count), dtype=jnp.bool_),
    )


def apply_events(state, Q, U, payloads, events):
    """Apply causal events to the memory.

    Args:
        state: :class:`AddressMemory` to update.
        Q: ``[K, K]`` read codes from :func:`make_address_codes`.
        U: ``[K, K]`` write codes from :func:`make_address_codes`.
        payloads: ``[K, L, V]`` literal payload bank.  ``payloads[d, l]`` is
            the value written by ``EVENT_LITERAL`` into address ``d`` with
            payload index ``l``.
        events: ``[B, E, 3]`` int32 triples ``(kind, destination, argument)``.

    Returns:
        A new :class:`AddressMemory`; the input state is not modified.
    """
    q = jnp.asarray(Q, dtype=jnp.float32)
    u = jnp.asarray(U, dtype=jnp.float32)
    bank = jnp.asarray(payloads, dtype=jnp.float32)
    ev = jnp.asarray(events, dtype=jnp.int32)
    matrix = jnp.asarray(state.matrix, dtype=jnp.float32)
    bound = jnp.asarray(state.bound, dtype=jnp.bool_)

    if matrix.ndim != 3:
        raise ValueError(f'state.matrix must be [B, K, V], got shape {matrix.shape}')
    if bound.shape != matrix.shape[:2]:
        raise ValueError(
            f'state.bound must be [B, K] matching the matrix, got shape {bound.shape}'
        )
    batch, key_count, value_dim = matrix.shape
    if q.shape != (key_count, key_count):
        raise ValueError(f'Q must be [{key_count}, {key_count}], got shape {q.shape}')
    if u.shape != q.shape:
        raise ValueError(f'U must match the Q shape {q.shape}, got shape {u.shape}')
    if bank.ndim != 3 or bank.shape[0] != key_count or bank.shape[2] != value_dim:
        raise ValueError(
            f'payloads must be [K={key_count}, L, V={value_dim}], got shape {bank.shape}'
        )
    if bank.shape[1] < 1:
        raise ValueError('payloads must hold at least one literal payload')
    if ev.ndim != 3 or ev.shape[0] != batch or ev.shape[2] != 3:
        raise ValueError(f'events must be [B={batch}, E, 3], got shape {ev.shape}')

    def step(carry, event):
        current, flags = carry
        kind = event[:, 0]
        destination = event[:, 1]
        argument = event[:, 2]

        destination_ok = (destination >= 0) & (destination < key_count)
        destination_index = jnp.clip(destination, 0, key_count - 1)
        source_index = jnp.clip(argument, 0, key_count - 1)
        source_ok = (argument >= 0) & (argument < key_count)
        literal_index = jnp.clip(argument, 0, bank.shape[1] - 1)
        literal_ok = (argument >= 0) & (argument < bank.shape[1])

        is_literal = (kind == EVENT_LITERAL) & destination_ok & literal_ok
        is_copy = (kind == EVENT_COPY) & destination_ok & source_ok
        # A copy onto the source itself cannot change anything, so it is
        # skipped instead of being executed as a zero sized write.
        is_self_copy = is_copy & (destination_index == source_index)
        active = (is_literal | is_copy) & ~is_self_copy

        read_destination = jnp.einsum('bk,bkv->bv', q[destination_index], current)
        copied_value = jnp.einsum('bk,bkv->bv', q[source_index], current)
        literal_value = bank[destination_index, literal_index]

        source_bound = jnp.take_along_axis(flags, source_index[:, None], axis=1)[:, 0]
        destination_bound = jnp.take_along_axis(flags, destination_index[:, None], axis=1)[:, 0]

        # An unbound source has nothing to copy, so the destination is
        # erased: writing the zero payload removes what it held.
        requested = jnp.where(
            is_literal[:, None], literal_value, jnp.where(source_bound[:, None], copied_value, 0.0)
        )
        residual = jnp.where(active[:, None], requested - read_destination, 0.0)
        delta = jnp.einsum('bk,bv->bkv', u[destination_index], residual)
        updated = jnp.where(active[:, None, None], current + delta, current)

        new_flag = jnp.where(active, is_literal | source_bound, destination_bound)
        rows = jnp.arange(batch)
        flags = flags.at[rows, destination_index].set(new_flag)
        return (updated, flags), None

    (matrix, bound), _ = jax.lax.scan(step, (matrix, bound), ev.transpose(1, 0, 2))
    return AddressMemory(matrix=matrix, bound=bound)


def read_memory(state, Q, keys):
    """Read the payloads held by zero-based addresses.

    Args:
        state: :class:`AddressMemory` to read.
        Q: ``[K, K]`` read codes from :func:`make_address_codes`.
        keys: ``[B, N]`` int32 zero-based addresses.

    Returns:
        ``(payloads, valid)``: ``payloads`` is ``[B, N, V]`` float32 and is
        zero wherever the address holds nothing or is out of range;
        ``valid`` is ``[B, N]`` bool and marks the addresses that hold a
        value.
    """
    q = jnp.asarray(Q, dtype=jnp.float32)
    address = jnp.asarray(keys, dtype=jnp.int32)
    matrix = jnp.asarray(state.matrix, dtype=jnp.float32)
    flags = jnp.asarray(state.bound, dtype=jnp.bool_)

    if matrix.ndim != 3:
        raise ValueError(f'state.matrix must be [B, K, V], got shape {matrix.shape}')
    if flags.shape != matrix.shape[:2]:
        raise ValueError(
            f'state.bound must be [B, K] matching the matrix, got shape {flags.shape}'
        )
    if address.ndim != 2 or address.shape[0] != matrix.shape[0]:
        raise ValueError(
            f'keys must be [B={matrix.shape[0]}, N], got shape {address.shape}'
        )
    key_count = matrix.shape[1]
    if q.shape != (key_count, key_count):
        raise ValueError(f'Q must be [{key_count}, {key_count}], got shape {q.shape}')

    index = jnp.clip(address, 0, key_count - 1)
    in_range = (address >= 0) & (address < key_count)
    rows = q[index]  # [B, N, K]
    raw = jnp.einsum('bnk,bkv->bnv', rows, matrix)
    valid = jnp.take_along_axis(flags, index, axis=1) & in_range
    payloads = jnp.where(valid[..., None], raw, 0.0)
    return payloads, valid


#: JIT compiled :func:`apply_events` with identical semantics.
apply_events_jit = jax.jit(apply_events)

#: JIT compiled :func:`read_memory` with identical semantics.
read_memory_jit = jax.jit(read_memory)
