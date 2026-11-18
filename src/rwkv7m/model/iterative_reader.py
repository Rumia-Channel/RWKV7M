"""A learned bounded read/emit policy over two-field RWKV memory records.

Records are the frozen model's *predictions*, not ground truth. At every round
both heads learn which field to match against a cursor and which field to
return. Demonstration and query cursors share the transition. Only demonstration
endpoints are visible to the policy. No stride candidates, step embeddings,
operation labels, inverse table or execution targets are inputs.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

NODES = 16
HEADS = 2
DEFAULT_CAP = 6
MISSING = NODES
NO_ANSWER = -2


class SoftRead(NamedTuple):
    mass: jax.Array           # [batch, question, node or missing], excludes timeout mass
    expected_reads: jax.Array
    timeout_mass: jax.Array


class Decisions(NamedTuple):
    answer: jax.Array         # zero-based, or -2 on missing/timeout
    reads: jax.Array          # logical rounds per question
    head: jax.Array           # -1 on timeout
    timed_out: jax.Array
    rounds: jax.Array         # physical rounds until the last live question in the batch


def init_controller(seed: int):
    """Small symmetry break between heads; no direction, distance or halt teacher."""
    rng = np.random.default_rng(seed)
    def noise(shape, scale): return jnp.asarray(rng.normal(0, scale, shape), dtype=jnp.float32)
    return {
        'field_base': 0.25 + noise((2,), 0.06),  # independent match and return logits
        'field_context': noise((4, 2), 0.04),
        'relevance': noise((4,), 0.03),
        'emit_linear': noise((4,), 0.04),
        'emit_hidden': noise((4, 8), 0.24),
        'emit_hidden_bias': jnp.zeros(8, jnp.float32),
        'emit_out': noise((8,), 0.05),
        'emit_bias': jnp.array([-1.25], jnp.float32),
    }


class _Context(NamedTuple):
    values: jax.Array
    start: jax.Array
    demo_targets: jax.Array
    relevance: jax.Array
    role: jax.Array


def _context(p, edges, demo_src, demo_dst, query, cap):
    if cap < 1 or edges.ndim != 2 or edges.shape[1] != NODES:
        raise ValueError('expected a positive read cap and [batch,16] observed edges')
    b = edges.shape[0]
    if (not b or demo_src.ndim != 2 or not demo_src.shape[1] or demo_src.shape != demo_dst.shape
            or demo_src.shape[0] != b or query.ndim != 2 or query.shape[0] != b or not query.shape[1]):
        raise ValueError('expected [batch,examples] starts/answers and [batch,questions] starts')
    if any(not jnp.issubdtype(x.dtype, jnp.integer) for x in (edges, demo_src, demo_dst, query)):
        raise TypeError('addresses must be integer node indices')
    # Invalid edge/hidden demo answer -> no match, never clamp it onto a real node.
    values = jax.nn.one_hot(edges, NODES)
    targets = jax.nn.one_hot(demo_dst, NODES)
    names = jnp.concatenate((demo_src, query), axis=1)
    initial = jax.nn.one_hot(jnp.where((names >= 0) & (names < NODES), names, MISSING), NODES+1)
    start = jnp.broadcast_to(initial[:,None], (b, HEADS, *initial.shape[1:]))
    xor = query[:,:,None] ^ demo_src[:,None,:]
    same_bits = ((xor[...,None] >> jnp.arange(4)) & 1) == 0
    relevance = jax.nn.softmax(same_bits.astype(jnp.float32) @ p['relevance'], axis=-1)
    role = (jnp.arange(names.shape[1]) >= demo_src.shape[1]).astype(jnp.float32)
    return _Context(values, start, targets, relevance, role)


def _hits(cursors, ctx):
    d = ctx.demo_targets.shape[1]
    demo = jnp.einsum('bhdk,bdk->bhd', cursors[:,:,:d,:NODES], ctx.demo_targets)
    relevant = jnp.einsum('bqd,bhd->bhq', ctx.relevance, demo)
    return demo, relevant


def _round(p, ctx, cursor):
    demo, relevant = _hits(cursor, ctx)
    seen = jnp.concatenate((demo, relevant), axis=-1)
    feat = jnp.stack((seen, cursor[..., :NODES].max(-1), (cursor * ctx.start).sum(-1),
                      jnp.broadcast_to(ctx.role, seen.shape)), axis=-1)
    # Complementary base biases encourage coverage of the two record fields.
    # Feedback features are per cursor and head, so the choice can change at each read.
    sign = jnp.array([1., -1.], jnp.float32)[None,:,None,None]
    match, emit_value = jnp.moveaxis(jax.nn.sigmoid(sign * (p['field_base'] + feat @ p['field_context'])), -1, 0)
    nodes = cursor[..., :NODES]
    value_match = jnp.einsum('bhpk,bik->bhpi', nodes, ctx.values)
    attention = match[...,None] * nodes + (1.-match[...,None]) * value_match
    total = attention.sum(-1, keepdims=True)
    attention = attention * (nodes.sum(-1, keepdims=True) / jnp.maximum(total, 1e-12))
    right = jnp.einsum('bhpi,bik->bhpk', attention, ctx.values)
    nxt = (1.-emit_value[...,None]) * attention + emit_value[...,None] * right
    nxt = jnp.concatenate((nxt, jnp.maximum(1.-nxt.sum(-1,keepdims=True), 0.)), axis=-1)
    demo, relevant = _hits(nxt, ctx)
    q = nxt[:,:,ctx.demo_targets.shape[1]:,:]
    evidence = jnp.stack((relevant, jnp.broadcast_to(demo.mean(-1,keepdims=True), relevant.shape),
                          q[...,:NODES].max(-1), relevant - jnp.flip(relevant, axis=1)), axis=-1)
    stop = (p['emit_bias'][0] + evidence @ p['emit_linear'] +
            jnp.tanh(evidence @ p['emit_hidden'] + p['emit_hidden_bias']) @ p['emit_out'])
    stop = stop.transpose(0,2,1)
    actions = jnp.concatenate((jnp.zeros_like(stop[:,:,:1]), stop), axis=-1)  # continue, emit head 0/1
    return nxt, q.transpose(0,2,1,3), actions


@partial(jax.jit, static_argnames=('cap', 'feedback'))
def soft_read(params, edges, demo_src, demo_dst, query, *, cap=DEFAULT_CAP, feedback=True):
    """Marginalize all halt rounds for endpoint-only training. No forced final emission."""
    ctx = _context(params, edges, demo_src, demo_dst, query, cap)
    live = jnp.ones(query.shape, jnp.float32)
    mass = jnp.zeros((*query.shape, NODES+1), jnp.float32)
    def step(_, carry):
        cursor, alive, acc, used = carry
        nxt, choices, logits = _round(params, ctx, cursor)
        prob = jax.nn.softmax(logits, axis=-1)
        emitted = jnp.einsum('bqh,bqhk->bqk', prob[...,1:], choices)
        return (nxt if feedback else ctx.start, alive*prob[...,0], acc+alive[...,None]*emitted, used+alive)
    _, alive, mass, used = jax.lax.fori_loop(0, cap, step, (ctx.start, live, mass, jnp.zeros_like(live)))
    return SoftRead(mass, used, alive)


@partial(jax.jit, static_argnames=('cap', 'feedback'))
def hard_read(params, edges, demo_src, demo_dst, query, *, cap=DEFAULT_CAP, feedback=True):
    """Autonomous inference: freeze emitted answers; exit when no question is pending."""
    ctx = _context(params, edges, demo_src, demo_dst, query, cap)
    d = demo_src.shape[1]
    def running(carry): return (carry[0] < cap) & carry[2].any()
    def step(carry):
        t, cursor, active, answer, used, head = carry
        nxt, choices, logits = _round(params, ctx, cursor)
        action = logits.argmax(-1).astype(jnp.int32)
        picked = jnp.maximum(action-1, 0)
        value = jnp.take_along_axis(choices, picked[...,None,None], axis=2)[:,:,0,:].argmax(-1).astype(jnp.int32)
        emit = active & (action > 0)
        answer = jnp.where(emit, jnp.where(value == MISSING, NO_ANSWER, value), answer)
        head = jnp.where(emit, picked, head)
        if feedback:
            moving = jnp.concatenate((jnp.broadcast_to(active.any(-1,keepdims=True), (active.shape[0],d)), active), axis=-1)
            cursor = jnp.where(moving[:,None,:,None], nxt, cursor)
        return t+1, cursor, active & ~emit, answer, used+active, head
    initial = (jnp.int32(0), ctx.start, jnp.ones(query.shape, jnp.bool_),
               jnp.full(query.shape, NO_ANSWER, jnp.int32), jnp.zeros(query.shape, jnp.int32),
               jnp.full(query.shape, -1, jnp.int32))
    t, _, active, answer, used, head = jax.lax.while_loop(running, step, initial)
    return Decisions(answer, used, head, active, t)
