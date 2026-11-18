"""Two disjoint public transit cycles: task examples never contain a test station.

The generator labels by positions in its cycles. The optional executor sees only
predicted one-edge reads, three visible example answers, and the final questions.
"""
from typing import NamedTuple

import numpy as np

from addressed_binding_tokens import COUNT, KEY_BASE, VALUE_BASE
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, FILLER_TOKEN, QUERY_TOKEN, STORE_TOKEN

DATA_SEEDS = (732041, 732043, 732047, 732049, 732053)
FAMILIES, EXAMPLES, TARGETS = 8, 3, 7
RULES = ('next', 'fourth', 'previous', 'conditional')
LEGEND, ALIAS_BASE = 6, 64
MAX_STRIDE = 4
NO_RULE = 99  # distinct from -1 (one station backwards)


class Panel(NamedTuple):
    known: np.ndarray       # [family, world, time] through the last graph write
    alias: np.ndarray       # only source spellings differ; same public legend
    examples: np.ndarray    # [rule, family, world, example, 4]
    queries: np.ndarray     # [family, world, station, 3]
    gold: np.ndarray        # [rule, family, world, station]


def questions(nodes):
    nodes = np.asarray(nodes, dtype=np.int32)
    return np.stack((np.full_like(nodes, QUERY_TOKEN), nodes + KEY_BASE,
                     np.full_like(nodes, ANSWER_MARKER_TOKEN)), axis=-1)


def _cycle_labels(cycle, nodes, speed, rule):
    position = {int(node): i for i, node in enumerate(cycle)}
    steps = {'next': 1, 'fourth': 4, 'previous': -1, 'conditional': 1 + (nodes % 2)}[rule]
    start = np.array([position[int(node)] for node in nodes])
    return VALUE_BASE + cycle[(start + speed * steps) % len(cycle)]


def _family(seed, family, gap):
    rng = np.random.default_rng(np.random.SeedSequence((seed, family)))
    nodes = rng.permutation(COUNT).astype(np.int32)
    target, demo = nodes[:TARGETS], nodes[TARGETS:]
    test_keys, displayed = rng.permutation(target), rng.permutation(COUNT)
    # Nine distinct demo names guarantee at least one of each parity.
    starts = [rng.choice(demo[demo % 2 == parity]) for parity in (0, 1)]
    starts.append(rng.choice(demo[~np.isin(demo, starts)]))
    starts = rng.permutation(starts)
    aliases = ALIAS_BASE + rng.permutation(COUNT)
    legend = [[LEGEND, KEY_BASE+i, VALUE_BASE+i, aliases[i]] for i in rng.permutation(COUNT)]
    noise = np.random.default_rng(np.random.SeedSequence((seed, family, 1)))
    distractors = noise.choice(demo, size=(gap // 8, 2))
    examples = np.empty((len(RULES), 2, EXAMPLES, 4), np.int32)
    gold = np.empty((len(RULES), 2, TARGETS), np.int32)
    known, aliased = [], []
    for w, speed in enumerate((1, 2)):
        edges = np.empty(COUNT, np.int32)
        edges[target] = np.roll(target, -speed)
        edges[demo] = np.roll(demo, -1)
        for use_alias, output in ((False, known), (True, aliased)):
            src = aliases if use_alias else KEY_BASE + np.arange(COUNT)
            row = [token for entry in legend for token in entry]
            def store(key, value): row.extend((STORE_TOKEN, src[key], VALUE_BASE + value))
            for k in displayed: store(k, edges[k])
            for k, v in distractors:
                store(k, v); row.extend((FILLER_TOKEN,) * 5)
            for k in demo: store(k, edges[k])
            output.append(row)
        for r, rule in enumerate(RULES):
            examples[r,w,:,:3] = questions(starts)
            examples[r,w,:,3] = _cycle_labels(demo, starts, 1, rule)
            gold[r,w] = _cycle_labels(target, test_keys, speed, rule)
    queries = np.repeat(questions(test_keys)[None], 2, axis=0)
    return (np.asarray(known, np.int32), np.asarray(aliased, np.int32), examples, queries, gold)


def make_panel(seed, gap, families=FAMILIES):
    if gap < 0 or gap % 8 or families < 1: raise ValueError('expected nonnegative gap in 8-token blocks and positive families')
    k, a, e, q, y = zip(*(_family(seed, f, gap) for f in range(families)), strict=True)
    return Panel(np.stack(k), np.stack(a), np.stack(e, axis=1), np.stack(q), np.stack(y, axis=1))


def normalize_sources(history):
    """Translate only alias source names, using the visible per-episode dictionary."""
    rows = np.asarray(history).reshape(-1, history.shape[-1])
    dictionary = rows[:, :4*COUNT].reshape(len(rows), COUNT, 4)
    canon = dictionary[:,:,1] - KEY_BASE
    alias = dictionary[:,:,3] - ALIAS_BASE
    domain = np.arange(COUNT)
    if (not np.all(dictionary[:,:,0] == LEGEND) or not np.all(dictionary[:,:,2] == VALUE_BASE + canon)
            or not np.all(np.sort(canon, axis=1) == domain) or not np.all(np.sort(alias, axis=1) == domain)):
        raise ValueError('not a complete visible bijective legend')
    mapping = np.empty_like(canon)
    mapping[np.arange(len(rows))[:,None], alias] = KEY_BASE + canon
    result = rows.copy()
    name = rows[:,1:] - ALIAS_BASE
    b, t = np.nonzero((rows[:,:-1] == STORE_TOKEN) & (name >= 0) & (name < COUNT))
    result[b,t+1] = mapping[b,name[b,t]]
    return result.reshape(history.shape)


def walk_tables(edges):
    """All signed walks of length <=4 using the supplied 16 predictions only.

    Missing/non-value successors and non-unique predecessors invalidate that path.
    Returns [stride+MAX_STRIDE, lane, station] of node indices (-1 means invalid).
    """
    forward = np.asarray(edges, np.int32) - VALUE_BASE
    forward = np.where((forward >= 0) & (forward < COUNT), forward, -1)
    inverse = np.full_like(forward, -1)
    for lane, row in enumerate(forward):
        src = np.flatnonzero(row >= 0); dest = row[src]
        unique = np.bincount(dest, minlength=COUNT)[dest] == 1
        inverse[lane,dest[unique]] = src[unique]
    walks = np.empty((2*MAX_STRIDE+1, len(edges), COUNT), np.int32)
    walks[MAX_STRIDE] = np.arange(COUNT)
    for sign, mapping in ((1, forward), (-1, inverse)):
        cur = walks[MAX_STRIDE]
        for length in range(1, MAX_STRIDE+1):
            cur = np.where(cur >= 0, np.take_along_axis(mapping, np.maximum(cur, 0), axis=1), -1)
            walks[MAX_STRIDE + sign*length] = cur
    return walks


def infer_walk(walks, examples, query):
    """Infer a constant stride, or abstain. No task label, hidden map or test answer."""
    lanes = np.arange(len(query))[:,None]
    demo = walks[:,lanes,examples[...,1]-KEY_BASE]
    matches = (demo == examples[...,3][None] - VALUE_BASE).all(axis=-1)
    unique = matches.sum(axis=0) == 1
    index = matches.argmax(axis=0)
    pred = walks[index[:,None],lanes,query[...,1]-KEY_BASE]
    return np.where(unique[:,None] & (pred >= 0), VALUE_BASE+pred, -2), np.where(unique, index-MAX_STRIDE, NO_RULE)
