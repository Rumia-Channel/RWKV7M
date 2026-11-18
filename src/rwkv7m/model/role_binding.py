"""One-probe policies with routes induced from unlabeled eight-token questions.

The marked runtime retains only a 64-cell payload bank and occupancy bits. Its
router sees raw words locally; neither the router nor the bank sees supervision,
source positions, or the preceding token history.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import episodic_address
from .episodic_address import CLASSES, SLOTS, project

ARMS = ('ordinary', 'bag', 'learned', 'marked-count', 'marked-rwkv')
PUBLIC_TOKENS = 128
WINDOW = 8


class Router(NamedTuple):
    markers: np.ndarray
    lexicon: np.ndarray


def init_policy(seed, arm):
    """Initialize exactly the parameters used by the selected policy arm."""
    if arm == 'ordinary':
        return episodic_address.init_head(seed, ordinary=True)
    if arm == 'bag':
        return episodic_address.init_head(seed, counts_only=True)
    if arm == 'learned':
        return episodic_address.init_head(seed)
    if arm not in ARMS:
        raise ValueError(f'unknown role-binding arm: {arm}')
    width, hidden = (128, 129) if arm == 'marked-count' else (256, 67)
    rng = np.random.default_rng(seed)

    def matrix(d, n):
        return jnp.asarray(rng.normal(0., d ** -.5, size=(d, n)).astype(np.float32))

    return {'input': matrix(width, hidden), 'bias': jnp.zeros(hidden, jnp.float32),
            'out': matrix(hidden, 1 + CLASSES), 'out_bias': jnp.zeros(1 + CLASSES, jnp.float32)}


def _validate_router(router):
    if not isinstance(router, Router):
        raise ValueError('expected a Router with int32 markers and lexicon')
    markers, lexicon = router
    for name, values in (('markers', markers), ('lexicon', lexicon)):
        if (not isinstance(values, np.ndarray) or values.dtype != np.int32 or values.ndim != 1 or
                np.any(values < 0) or np.any(values >= PUBLIC_TOKENS) or np.any(np.diff(values) <= 0)):
            raise ValueError(f'router {name} must contain sorted distinct public int32 token IDs')
    if markers.shape != (2,) or not 2 <= len(lexicon) <= 8:
        raise ValueError('router requires two markers and a lexicon of 2..8 terms')


def _raw_tokens(tokens):
    tokens = np.asarray(tokens)
    if tokens.shape[-1:] != (WINDOW,) or tokens.dtype.kind not in 'iu':
        raise ValueError('expected integer raw tokens with final dimension 8')
    return tokens


def induce_router(training_queries):
    """Find repeated marker syntax using only the unlabeled question words.

    Eligible markers occur once, after a word, in every question. Pair ranking
    uses only distinct ordered predecessor pairs, with a numeric marker tie-break.
    The two marker vocabularies must be the same nonconstant set of 2..8 words.
    """
    queries = _raw_tokens(training_queries).reshape(-1, WINDOW)
    if not len(queries) or np.any(queries < 0) or np.any(queries >= PUBLIC_TOKENS):
        raise ValueError('no router in empty or non-public training questions')
    rows = np.arange(len(queries))
    candidates = []
    for token in range(PUBLIC_TOKENS):
        hits = queries == token
        if not np.all(hits.sum(axis=1) == 1):
            continue
        positions = hits.argmax(axis=1)
        if np.any(positions == 0):
            continue
        predecessors = queries[rows, positions - 1]
        vocabulary = np.unique(predecessors)
        if 2 <= len(vocabulary) <= 8:
            candidates.append((token, predecessors, vocabulary))
    best, score = None, 1
    for i, (left, before_left, vocabulary) in enumerate(candidates):
        for right, before_right, other_vocabulary in candidates[i + 1:]:
            if not np.array_equal(vocabulary, other_vocabulary):
                continue
            distinct = len(np.unique(np.stack((before_left, before_right), axis=1), axis=0))
            if distinct > score:
                best = Router(np.array((left, right), np.int32), vocabulary.astype(np.int32))
                score = distinct
    if best is None:
        raise ValueError('no variable shared-lexicon marker pair in training questions')
    return best


def route(router, tokens):
    """Return a local 0..63 route, or -1 without aliasing malformed phrases."""
    _validate_router(router)
    tokens = _raw_tokens(tokens)
    flat = tokens.reshape(-1, WINDOW)
    hits = flat[:, None, :] == router.markers[None, :, None]
    positions = hits.argmax(axis=-1)
    valid = (hits.sum(axis=-1) == 1) & (positions > 0)
    predecessors = flat[np.arange(len(flat))[:, None], np.maximum(positions - 1, 0)]
    known = predecessors[..., None] == router.lexicon
    valid &= known.any(axis=-1)
    codes = known.argmax(axis=-1).astype(np.int32)
    public = ((flat >= 0) & (flat < PUBLIC_TOKENS)).all(axis=-1)
    address = codes[:, 0] + 8 * codes[:, 1]
    return np.where(valid.all(axis=-1) & public, address, -1).astype(np.int32).reshape(tokens.shape[:-1])


def _check_marked_head(params):
    if params['out'].shape[-1:] != (1 + CLASSES,) or params['out_bias'].shape != (1 + CLASSES,):
        raise ValueError('marked heads require one gate and eight payload logits')


def _marked_inputs(params, blocks, valid, write_addr, read_addr):
    blocks, valid = jnp.asarray(blocks), jnp.asarray(valid)
    write_addr, read_addr = jnp.asarray(write_addr), jnp.asarray(read_addr)
    if (blocks.ndim != 3 or valid.shape != blocks.shape[:2] or write_addr.shape != valid.shape or
            read_addr.ndim != 2 or read_addr.shape[0] != blocks.shape[0]):
        raise ValueError('expected blocks [B,T,D], validity/write routes [B,T] and read routes [B,Q]')
    if write_addr.dtype.kind not in 'iu' or read_addr.dtype.kind not in 'iu':
        raise ValueError('marked routes must be integers')
    _check_marked_head(params)
    return blocks, valid.astype(bool), write_addr, read_addr


@jax.jit
def marked_write(params, state, x, valid, addr):
    """Consume one local window per lane, without any question or history search."""
    x, valid, addr = jnp.asarray(x), jnp.asarray(valid), jnp.asarray(addr)
    if (x.ndim != 2 or valid.shape != x.shape[:1] or addr.shape != valid.shape or
            state.bank.shape[0] != x.shape[0]):
        raise ValueError('expected features [B,D], validity/routes [B] and a matching bank')
    if addr.dtype.kind not in 'iu':
        raise ValueError('marked routes must be integers')
    _check_marked_head(params)
    logits = project(params, x)
    safe = (addr >= 0) & (addr < SLOTS)
    index = jnp.where(safe, addr, 0).astype(jnp.int32)
    enabled = valid.astype(bool) & safe & (logits[:, 0] > 0)
    state = episodic_address._store(state, index, enabled, logits[:, 1:])
    return state, jnp.where(safe, index, -1), enabled


@jax.jit
def read_lanes(state, lanes, addr):
    """One (lane, slot) gather per requested question, not a copy of each lane's bank."""
    safe = (addr >= 0) & (addr < SLOTS)
    slot = jnp.where(safe, addr, 0).astype(jnp.int32)
    prediction = state.bank[lanes, slot].argmax(axis=-1).astype(jnp.int32)
    answers = jnp.where(safe & state.occupied[lanes, slot], prediction, -1)
    return answers, jnp.where(safe, slot, -1)


@jax.jit
def marked_read(state, addr):
    """Read one indexed cell per [lane, query], without mutating the bank."""
    addr = jnp.asarray(addr)
    if addr.ndim != 2 or addr.shape[0] != state.bank.shape[0]:
        raise ValueError('expected read routes [B,Q] and a matching bank')
    if addr.dtype.kind not in 'iu':
        raise ValueError('marked routes must be integers')
    return read_lanes(state, jnp.arange(state.bank.shape[0])[:, None], addr)


@jax.jit
def hard_marked(params, blocks, valid, write_addr, read_addr):
    """Scan local writes, then perform one indexed bank read per question.

    Only a positive gate at a present window with a safe route can write. Invalid
    routes are reported as -1; a safe but unwritten cell also abstains.
    """
    blocks, valid, write_addr, read_addr = _marked_inputs(params, blocks, valid, write_addr, read_addr)

    def step(state, row):
        state, address, enabled = marked_write(params, state, *row)
        return state, (address, enabled)

    state, (writes, enabled) = jax.lax.scan(
        step, episodic_address.empty_state(blocks.shape[0]),
        (blocks.swapaxes(0, 1), valid.T, write_addr.T))
    answers, reads = marked_read(state, read_addr)
    return episodic_address.Decisions(answers, reads, writes.T, enabled.T)


def soft_marked(params, blocks, valid, write_addr, read_addr, beta):
    """Training-only marginal over Bernoulli writes at fixed inferred addresses.

    A write owns a query exactly when it fires and every later write at that
    address does not. Empty/invalid reads contribute a uniform eight-class miss.
    """
    blocks, valid, write_addr, read_addr = _marked_inputs(params, blocks, valid, write_addr, read_addr)
    logits = project(params, blocks)
    safe_write = (write_addr >= 0) & (write_addr < SLOTS)
    safe_read = (read_addr >= 0) & (read_addr < SLOTS)
    gates = jax.nn.sigmoid(beta * logits[..., 0]) * (valid & safe_write)
    same_address = (read_addr[:, :, None] == write_addr[:, None, :]) & safe_read[:, :, None]
    writes = gates[:, None, :] * same_address
    after = jnp.cumprod(1. - writes[..., 1:][..., ::-1], axis=-1)[..., ::-1]
    after = jnp.concatenate((after, jnp.ones_like(writes[..., :1])), axis=-1)
    owners = writes * after
    values = jax.nn.softmax(logits[..., 1:], axis=-1)
    miss = jnp.maximum(1. - owners.sum(axis=-1, keepdims=True), 0.)
    return episodic_address.SoftResult(jnp.einsum('bqt,btc->bqc', owners, values) + miss / CLASSES, owners)
