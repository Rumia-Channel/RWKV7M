"""Counterfactual late-write data for the blind-student distillation trial (§22.16).

The trial fine-tunes an ordinary 504,960-parameter RWKV on histories whose
answer is written *late*, with and without the frozen last-64-token teacher
logits.  This module samples the counterfactual histories, their questions,
their exact answers and the mask of questions a frozen last-64-token reader
could answer.  It returns plain NumPy arrays only.

A distillation history (215 tokens) is a 63-token prefix (the ordinary
shuffled 16-key / 16-value initialization on a stride of four, eight scored
odd slots), a 128-token gap of legal changing writes to the other eight keys
only, and the 24-token §22.15 tail (four stride-four rewrites, eight filler).
The first 88 gap tokens are the nuisance and may differ between the two
nuisance variants, the last 40 are shared, so the final 64 tokens of a
history are the shared gap suffix plus the tail.

A family has four views, ``world = 2 * nuisance + version``: the nuisance
chooses the first 88 gap tokens, the version chooses a cyclic derangement of
the four original target values written by the tail.  Both versions change all
four rewritten keys; the views of one version share their final 64 tokens; the
two versions of one nuisance share their earlier history.  The two questions
are one rewritten key and one untouched original target, in a random order
that the four views share.

``step % 4 == 3`` carries anchor rows instead: it alternates between the
ordinary near 63-token binding and the legacy gap-32 overwrite rows of
:func:`retrieval_frontier.make_examples`.  Each condition has two meaningful
counterfactual worlds and the four views are two distinct counterfactual
pairs, never fabricated mutations of an answer.  Anchor questions are never
teacher eligible.

Family ``f`` of step ``s`` is sampled from ``SeedSequence([seed, s, f, tag])``,
so the data of a family never depends on how many families a batch holds.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from retrieval_frontier import (
    ANSWER_MARKER,
    FILLER,
    KEY_TOKENS,
    PREFIX_TOKENS,
    QUERY,
    QUERY_TOKENS,
    QUERYABLE_SLOTS,
    ROW_TOKENS,
    STORE,
    VALUE_TOKENS,
    make_examples,
)

#: Tokens of the legal changing-write gap between the prefix and the tail.
GAP_TOKENS = 128
#: Gap tokens the two nuisance variants may differ in.
NUISANCE_GAP_TOKENS = 88
#: Tokens of the §22.15 tail: four stride-four rewrites and eight filler.
TAIL_TOKENS = 24
#: Rewrites inside the tail.
TAIL_WRITES = 4
#: Frozen teacher window: the trailing tokens a frozen reader is given.
TEACHER_TOKENS = 64
#: Tokens of a distillation history.
DISTILL_TOKENS = PREFIX_TOKENS + GAP_TOKENS + TAIL_TOKENS
#: Tokens of the ordinary near binding history.
NEAR_TOKENS = PREFIX_TOKENS
#: Tokens of the legacy gap-32 overwrite history (the row without its query).
LEGACY_TOKENS = ROW_TOKENS - QUERY_TOKENS
#: Where the gap and the tail start inside a distillation history.
GAP_START = PREFIX_TOKENS
TAIL_START = GAP_START + GAP_TOKENS
#: Gap write events: one stride-four write each.
GAP_EVENTS = GAP_TOKENS // 4
#: Gap events whose tokens the two nuisance variants share.
GAP_SHARED_EVENTS = NUISANCE_GAP_TOKENS // 4
#: Slots of the prefix that are never interfered with (the eight even slots).
OTHER_SLOTS = np.arange(0, PREFIX_TOKENS // 4, 2)
#: Slots of the prefix that may be queried or rewritten (the eight odd ones).
SCORED_SLOTS = np.asarray(QUERYABLE_SLOTS)

#: Stream tags; they keep the sampled windows of the three conditions apart.
TAG_DISTILL = 0
TAG_NEAR = 1
TAG_LEGACY = 2
#: Index base of the legacy anchor windows; disjoint from the panel (2e6) and
#: from the calibration windows (1e6) of the retrieval-frontier study.
ANCHOR_INDEX_BASE = 3_000_000
#: Width of the anchor index span.
ANCHOR_INDEX_SPAN = 1_000_000

_KEY_SET = frozenset(int(key) for key in KEY_TOKENS)
_VALUE_SET = frozenset(int(value) for value in VALUE_TOKENS)


def _require_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _public_writes(row):
    """Ordered ``(key, value)`` records of every complete legal write."""
    records = []
    for position in np.flatnonzero(row == STORE):
        if position + 2 >= len(row):
            continue
        key, value = int(row[position + 1]), int(row[position + 2])
        if key in _KEY_SET and value in _VALUE_SET:
            records.append((key, value))
    return records


def _latest_values(row):
    """Latest public value of every key written in a row."""
    latest = {}
    for key, value in _public_writes(row):
        latest[key] = value
    return latest


def _teacher_hits(row, key):
    """True when a complete legal write of ``key`` lies in the teacher window."""
    window = row[-TEACHER_TOKENS:]
    for position in np.flatnonzero(window == STORE):
        if position + 2 >= len(window):
            continue
        if int(window[position + 1]) == key and int(window[position + 2]) in _VALUE_SET:
            return True
    return False


def _gap_shift(write_index, variant, shared):
    """Shift of the value pool used by one gap write.

    Shared writes use the same shift in both nuisance variants, varying
    writes differ between the variants, and consecutive writes of a key never
    share a shift.  Every shift is non-zero, so a gap write never repeats the
    value its key currently holds.
    """
    if shared:
        return write_index + 1
    return write_index + variant


def _distill_family(seed, step, family):
    """Sample the facts of one distillation family."""
    root = np.random.SeedSequence([seed, step, family, TAG_DISTILL])
    key_rng, value_rng, target_rng, query_rng = (
        np.random.default_rng(child) for child in root.spawn(4)
    )
    keys = key_rng.permutation(np.asarray(KEY_TOKENS))
    values = value_rng.permutation(np.asarray(VALUE_TOKENS))
    target_slots = np.sort(
        target_rng.choice(SCORED_SLOTS, size=TAIL_WRITES, replace=False)
    )
    # Two distinct rotations of the four target values; both are derangements.
    rotations = target_rng.choice(np.arange(1, TAIL_WRITES), size=2, replace=False)
    base = values[target_slots]
    tail_values = [
        base[(np.arange(TAIL_WRITES) + int(rotation)) % TAIL_WRITES]
        for rotation in rotations
    ]
    # One rewritten target and one untouched original target are queried; the
    # order of the two questions is random but shared by the four views.
    remaining = np.setdiff1d(SCORED_SLOTS, target_slots)
    chosen = [
        int(target_rng.choice(target_slots)),
        int(target_rng.choice(remaining)),
    ]
    order = query_rng.permutation(2)
    question_slots = [chosen[int(index)] for index in order]
    return keys, values, target_slots, tail_values, question_slots


def _distill_rows(seed, step, family):
    """Build the four views of one distillation family."""
    keys, values, target_slots, tail_values, question_slots = _distill_family(
        seed, step, family
    )
    other_keys = keys[OTHER_SLOTS]
    other_values = values[OTHER_SLOTS]
    rows = np.empty((4, DISTILL_TOKENS), np.int32)
    for world in range(4):
        nuisance, version = world // 2, world % 2
        row = np.full(DISTILL_TOKENS, FILLER, np.int32)
        row[0:PREFIX_TOKENS:4] = STORE
        row[1:PREFIX_TOKENS:4] = keys
        row[2:PREFIX_TOKENS:4] = values
        for event in range(GAP_EVENTS):
            slot = event % len(OTHER_SLOTS)
            write_index = event // len(OTHER_SLOTS) + 1
            shared = event >= GAP_SHARED_EVENTS
            shift = _gap_shift(write_index, nuisance, shared)
            position = GAP_START + 4 * event
            row[position] = STORE
            row[position + 1] = other_keys[slot]
            row[position + 2] = other_values[(slot + shift) % len(OTHER_SLOTS)]
        for index in range(TAIL_WRITES):
            position = TAIL_START + 4 * index
            row[position + 1] = STORE
            row[position + 2] = keys[target_slots[index]]
            row[position + 3] = tail_values[version][index]
        rows[world] = row
    return rows, keys, question_slots


def _near_pair(seed, step, family, pair):
    """One counterfactual pair of the ordinary near 63-token binding."""
    root = np.random.SeedSequence([seed, step, family, TAG_NEAR, pair])
    key_rng, value_rng, query_rng = (
        np.random.default_rng(child) for child in root.spawn(3)
    )
    keys = key_rng.permutation(np.asarray(KEY_TOKENS))
    values = value_rng.permutation(np.asarray(VALUE_TOKENS))
    slots = np.sort(query_rng.choice(SCORED_SLOTS, size=2, replace=False))
    prefix = np.full(PREFIX_TOKENS, FILLER, np.int32)
    prefix[0::4] = STORE
    prefix[1::4] = keys
    prefix[2::4] = values
    # The counterfactual world rotates the eight scored values, so every
    # question changes its answer while the token multiset stays fixed.
    counterfactual = prefix.copy()
    counterfactual[6::8] = np.roll(values[1::2], 1)
    return prefix, counterfactual, keys, slots


def _legacy_pair(seed, step, family, pair):
    """One counterfactual pair of the legacy gap-32 overwrite rows."""
    entropy = np.random.SeedSequence(
        [seed, step, family, TAG_LEGACY, pair]
    ).generate_state(2, dtype=np.uint32)
    window_seed = int(entropy[0])
    window_index = ANCHOR_INDEX_BASE + int(entropy[1]) % ANCHOR_INDEX_SPAN
    ids, labels = make_examples(
        window_seed, window_index, 1, gap=32, writes=0, anchor=True
    )
    return ids[0], labels[0]


def _distill_step(seed, step, families, history, queries, labels, eligible):
    """Fill the batch arrays with the distillation views of one step."""
    for family in range(families):
        rows, keys, question_slots = _distill_rows(seed, step, family)
        for world in range(4):
            row = rows[world]
            history[family, world] = row
            latest = _latest_values(row)
            for question, slot in enumerate(question_slots):
                key = int(keys[slot])
                queries[family, world, question] = (QUERY, key, ANSWER_MARKER)
                labels[family, world, question] = latest[key]
                eligible[family, world, question] = _teacher_hits(row, key)


def _anchor_step(seed, step, families, history, queries, labels):
    """Fill the batch arrays with the anchor views of one step."""
    legacy = (step // 4) % 2 == 1
    for family in range(families):
        for pair in range(2):
            for world in range(2):
                view = 2 * pair + world
                if legacy:
                    ids, pair_labels = _legacy_pair(seed, step, family, pair)
                    # Both questions of a view read the same history.
                    history[family, view] = ids[world, 0, :-3]
                    for question in range(2):
                        queries[family, view, question] = ids[world, question, -3:]
                        labels[family, view, question] = pair_labels[world, question]
                else:
                    prefix, counterfactual, keys, slots = _near_pair(
                        seed, step, family, pair
                    )
                    row = prefix if world == 0 else counterfactual
                    history[family, view] = row
                    latest = _latest_values(row)
                    for question, slot in enumerate(slots):
                        key = int(keys[slot])
                        queries[family, view, question] = (QUERY, key, ANSWER_MARKER)
                        labels[family, view, question] = latest[key]


def is_anchor_step(step):
    """True when the step carries anchor rows instead of distillation rows."""
    return _require_int(step, "step") % 4 == 3


def history_tokens(step):
    """Tokens of one history of the step (215, 63 or 253)."""
    step = _require_int(step, "step")
    if not is_anchor_step(step):
        return DISTILL_TOKENS
    return LEGACY_TOKENS if (step // 4) % 2 == 1 else NEAR_TOKENS


def training_batch(seed, step, families=2):
    """Sample one training batch of the §22.16 late-write trial.

    Parameters
    ----------
    seed:
        Training data seed of the run.
    step:
        Step index; ``step % 4 == 3`` selects the anchor rows, every other
        step selects the distillation rows.
    families:
        Number of independent families in the batch.  The data of a family
        never depends on this number, so a batch can be extended or truncated
        without changing the families it already holds.

    Returns
    -------
    (history, queries, labels, eligible):
        ``history`` is ``int32`` with shape ``(families, 4, T)``, ``queries``
        is ``int32`` with shape ``(families, 4, 2, 3)`` and holds
        ``(QUERY, key, ANSWER_MARKER)`` blocks, ``labels`` is ``int32`` with
        shape ``(families, 4, 2)`` and holds the exact latest public write of
        the queried key, and ``eligible`` is ``bool`` with shape
        ``(families, 4, 2)`` and marks the questions a frozen 64-token reader
        may teach: the rewritten question of a distillation family, whose key
        is written inside the teacher window.  Anchor questions are never
        eligible.  ``T`` is
        :func:`history_tokens` of the step.  No JAX object is returned and no
        answer is part of the returned tokens.
    """
    seed = _require_int(seed, "seed")
    step = _require_int(step, "step")
    families = _require_int(families, "families")
    if seed < 0 or step < 0:
        raise ValueError("seed and step must not be negative")
    if families < 1:
        raise ValueError("families must be positive")

    history = np.empty((families, 4, history_tokens(step)), np.int32)
    queries = np.empty((families, 4, 2, 3), np.int32)
    labels = np.empty((families, 4, 2), np.int32)
    eligible = np.zeros((families, 4, 2), bool)
    if is_anchor_step(step):
        _anchor_step(seed, step, families, history, queries, labels)
    else:
        _distill_step(seed, step, families, history, queries, labels, eligible)
    return history, queries, labels, eligible
