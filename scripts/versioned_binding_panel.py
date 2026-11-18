"""Versioned copy-chain panel for the 22.17 state-screening trial.

Public grammar: key tokens 16..31, literal tokens 32..47, ``STORE key value``,
``COPY dst src`` (the value the source holds at the instant of the copy),
``QUERY key ANSWER`` and filler.  Every family starts from the shared 63-token
prefix of ``probe_retrieval_distance.histories`` and adds four disjoint value
paths over the eight queried keys; the remaining eight keys are the changing
gap distractors. At each stage each live wire is copied onto a distinct dead
slot and the very next command poisons its old source. The routing prohibits
a wire from returning to its origin, so the original value of
a key cannot answer the panel.  Labels follow the lineage of the four values
through the route and are re-derived from the visible tokens alone by
:func:`literal_reference`.
"""

from itertools import permutations
from typing import NamedTuple

import numpy as np

from addressed_binding_tokens import COPY, COUNT, KEY_BASE, VALUE_BASE
from probe_retrieval_distance import histories
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, FILLER_TOKEN, QUERY_TOKEN, STORE_TOKEN


MISSING = -2
BLOCK = 64
PREFIX_TOKENS = 63
SUFFIX_TOKENS = 8
CHAINS = 4
TARGETS = COUNT // 2          # eight queried keys, eight changing distractors
STAGE_TOKENS = 6              # COPY dst src, then STORE src poison
ROUTES = tuple(permutations(range(CHAINS)))


class Path(NamedTuple):
    """Route of one family: the stage events and the state after every stage."""
    copies: tuple          # [stage][j] = (destination slot, source slot)
    stores: tuple          # [stage][j] = (overwritten slot, lineage)
    states: tuple          # [stage] = (live slot -> lineage, poisoned slot -> lineage)


def _plan(seed, family, labels):
    """Pairing, query order and poison of one family; fixed by seed and family."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy=(int(seed), int(family))))
    pairing = rng.permutation(TARGETS)
    order = rng.permutation(TARGETS)
    # The original values of the queried keys are protected, so a poison literal
    # can never answer a live query.  Both worlds protect the same eight values.
    protected = {int(value) for value in labels.ravel()}
    candidates = np.array([v for v in range(VALUE_BASE, VALUE_BASE + COUNT) if v not in protected], np.int32)
    if len(candidates) < CHAINS:
        raise ValueError('no literal is free of the protected original values')
    poison = rng.choice(candidates, CHAINS, replace=False)
    return pairing, order, poison


def _paths(seed, family, pairing, stages):
    """Route the four live wires onto the four dead slots, stage by stage.

    The route of a stage depends only on the seed, the family and the stage, so
    the first stages of a deep panel are exactly the stages of a shallow panel.
    A wire never returns to the slot it started from, which keeps the original
    value of a key from answering the panel.
    """
    origins = [int(pairing[2 * c]) for c in range(CHAINS)]
    live = tuple(sorted(origins))
    lineage = {slot: c for c, slot in enumerate(origins)}
    copies, stores, states = [], [], []
    for stage in range(stages):
        dead = tuple(slot for slot in range(TARGETS) if slot not in live)
        rng = np.random.default_rng(np.random.SeedSequence(entropy=(int(seed), int(family), int(stage))))
        start = int(rng.integers(len(ROUTES)))
        for step in range(len(ROUTES)):
            route = ROUTES[(start + step) % len(ROUTES)]
            if all(dead[j] != origins[lineage[live[route[j]]]] for j in range(CHAINS)):
                break
        copies.append(tuple((dead[j], live[route[j]]) for j in range(CHAINS)))
        stores.append(tuple((live[route[j]], lineage[live[route[j]]]) for j in range(CHAINS)))
        lineage = {dead[j]: lineage[live[route[j]]] for j in range(CHAINS)}
        live = dead
        states.append((dict(lineage), dict(stores[-1])))
    return Path(tuple(copies), tuple(stores), tuple(states))


def _build(plans, paths, prefix, queries, labels, block, depth, gap):
    """Assemble one (depth, gap) case from the per-family plans and routes."""
    families = len(plans)
    half = depth // 2
    total = PREFIX_TOKENS + gap + depth * CHAINS * STAGE_TOKENS + SUFFIX_TOKENS
    history = np.empty((families, 2, total), np.int32)
    history[:, :, :PREFIX_TOKENS] = prefix
    # The intervening gap sits halfway through the stages, so a snapshot has to
    # survive unrelated writes before and after it.
    gap_start = PREFIX_TOKENS + half * CHAINS * STAGE_TOKENS
    if gap:
        # The changing 64-token-block rotation of the distractor keys.
        values = block[:, :, 2::8]
        for t in range(gap // BLOCK):
            start = gap_start + t * BLOCK
            history[:, :, start:start + BLOCK] = block
            history[:, :, start + 2:start + BLOCK:8] = np.roll(values, (t + 1) % TARGETS, axis=-1)
    keys = queries[:, 0, :, 1]                                # [families, TARGETS]
    copy_dst = np.empty((families, depth, CHAINS), np.int64)
    copy_src = np.empty((families, depth, CHAINS), np.int64)
    store_slot = np.empty((families, depth, CHAINS), np.int64)
    store_lineage = np.empty((families, depth, CHAINS), np.int64)
    for f, path in enumerate(paths):
        for stage in range(depth):
            for j in range(CHAINS):
                copy_dst[f, stage, j], copy_src[f, stage, j] = path.copies[stage][j]
                store_slot[f, stage, j], store_lineage[f, stage, j] = path.stores[stage][j]
    poison = np.stack([plan[2] for plan in plans])            # [families, CHAINS]
    families_index = np.arange(families)
    for stage in range(depth):
        base = PREFIX_TOKENS + stage * CHAINS * STAGE_TOKENS
        if stage >= half:
            base = gap_start + gap + (stage - half) * CHAINS * STAGE_TOKENS
        for j in range(CHAINS):
            pos = base + j * STAGE_TOKENS
            history[:, :, pos] = COPY
            history[:, :, pos + 1] = keys[families_index, copy_dst[:, stage, j]][:, None]
            history[:, :, pos + 2] = keys[families_index, copy_src[:, stage, j]][:, None]
            pos += 3  # Poison this source before any other chain's next command.
            history[:, :, pos] = STORE_TOKEN
            history[:, :, pos + 1] = keys[families_index, store_slot[:, stage, j]][:, None]
            history[:, :, pos + 2] = poison[families_index, store_lineage[:, stage, j]][:, None]
    history[:, :, total - SUFFIX_TOKENS:] = FILLER_TOKEN
    truth = np.empty((families, 2, TARGETS), np.int32)
    active = np.zeros((families, TARGETS), bool)
    for f, (pairing, _, choice) in enumerate(plans):
        live, dead = paths[f].states[depth - 1]
        for slot in range(TARGETS):
            if slot in live:
                truth[f, :, slot] = labels[f, :, int(pairing[2 * live[slot]])]
                active[f, slot] = True
            else:
                truth[f, :, slot] = choice[dead[slot]]
    order = np.array([plan[1] for plan in plans], np.int64)
    truth = np.take_along_axis(truth, order[:, None, :], axis=2)
    active = np.take_along_axis(active, order, axis=1)
    return history, truth, active


def versioned_cases(seed=732013, families=32, depths=(2, 24), gaps=(0, 1024)):
    """Yield ``(depth, gap, history, queries, truth, active)`` panels.

    ``history`` is ``int32[F,2,T]`` (two counterfactual worlds per family),
    ``queries`` is ``int32[F,2,8,3]`` in the same shuffled order for both
    worlds, ``truth`` is ``int32[F,2,8]`` and ``active`` marks the four wires
    that still carry a copied original value.
    """
    families = int(families)
    depths = tuple(int(d) for d in depths)
    gaps = tuple(int(g) for g in gaps)
    if families < 1 or not depths or not gaps or min(depths) < 1 or min(gaps) < 0:
        raise ValueError('use positive families, depths and gaps')
    if any(g % BLOCK for g in gaps):
        raise ValueError(f'gaps must be multiples of {BLOCK}')
    prefix, block, queries, labels = histories(int(seed), families, 'other-writes')
    prefix = prefix.reshape(families, 2, PREFIX_TOKENS)
    block = block.reshape(families, 2, BLOCK)
    plans = [_plan(seed, f, labels[f]) for f in range(families)]
    paths = [_paths(seed, f, plans[f][0], max(depths)) for f in range(families)]
    shuffled = np.empty_like(queries)
    for f, plan in enumerate(plans):
        shuffled[f] = queries[f][:, plan[1]]
    for depth in depths:
        for gap in gaps:
            history, truth, active = _build(plans, paths, prefix, queries, labels, block, depth, gap)
            yield depth, gap, history, shuffled, truth, active


def literal_reference(history, question_keys):
    """Read the visible grammar only and answer each question key.

    ``history`` is ``int32[B,T]`` and ``question_keys`` is ``int32[B,N]``;
    the result is ``int32[B,N]`` holding the bound literal, or ``-2`` when the
    key holds nothing.  A copy stores the value the source holds at that
    instant, so later writes to the source never reach the copy, and a copy of
    an unbound source leaves the destination unbound.
    """
    history = np.asarray(history)
    question_keys = np.asarray(question_keys)
    if history.ndim != 2 or question_keys.ndim != 2 or history.shape[0] != question_keys.shape[0]:
        raise ValueError('expected history[B,T] and question_keys[B,N] with equal batch')
    rows = np.arange(history.shape[0])
    # One byte per key: the literal index, or -1 when the key holds nothing.
    table = np.full((history.shape[0], COUNT), -1, np.int8)
    starts = np.flatnonzero(np.any((history == STORE_TOKEN) | (history == COPY), axis=0))
    for start in starts:
        if start + 2 >= history.shape[1]:
            break                      # an incomplete event at the end never commits
        marker = history[:, start]
        target = history[:, start + 1]
        argument = history[:, start + 2]
        key = (target >= KEY_BASE) & (target < KEY_BASE + COUNT)
        literal = (marker == STORE_TOKEN) & key & (argument >= VALUE_BASE) & (argument < VALUE_BASE + COUNT)
        copied = (marker == COPY) & key & (argument >= KEY_BASE) & (argument < KEY_BASE + COUNT)
        if literal.any():
            table[rows[literal], target[literal] - KEY_BASE] = (argument[literal] - VALUE_BASE).astype(np.int8)
        if copied.any():
            table[rows[copied], target[copied] - KEY_BASE] = table[rows[copied], argument[copied] - KEY_BASE]
    asked = (question_keys >= KEY_BASE) & (question_keys < KEY_BASE + COUNT)
    found = np.take_along_axis(table, np.where(asked, question_keys - KEY_BASE, 0), axis=1)
    return np.where(asked & (found >= 0), found.astype(np.int32) + VALUE_BASE, MISSING).astype(np.int32)


__all__ = [
    'ANSWER_MARKER_TOKEN', 'BLOCK', 'CHAINS', 'COPY', 'COUNT', 'FILLER_TOKEN', 'KEY_BASE', 'MISSING',
    'PREFIX_TOKENS', 'QUERY_TOKEN', 'STAGE_TOKENS', 'STORE_TOKEN', 'SUFFIX_TOKENS', 'TARGETS',
    'VALUE_BASE', 'literal_reference', 'versioned_cases',
]
