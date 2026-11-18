"""Controlled eight-token scenes for query-independent episodic addressing.

Each live utterance binds an opaque actor/occasion pair to an object.  Two
counterfactual worlds exchange only the object mentions at two queried facts;
all other visible tokens, including background utterances, are identical.
Ordinary ``kept`` clauses bind facts, whereas ``discussed``/``noticed`` clauses
only mention the same domains.  No query or target-role marker enters a history.

``draw`` is the seed.  Separate per-family semantic, object, position, surface,
and background streams make gap extension prefix-consistent.  Surface choices
never depend on answers, target roles, world, or support positions.  The planned
training/final cells are below; the generator itself supports 4 <= K <= 32.
"""
from operator import index
from typing import NamedTuple

import numpy as np


SPLITS = ('train', 'dev', 'seen', 'held')
TRAIN_DRAWS = (752101, 752103, 752107, 752109)
DEV_DRAWS = (753101, 753103)
TEST_DRAWS = (754101, 754103, 754107, 754109, 754113)
TRAIN_CELLS = ((4, 0), (4, 8), (8, 0), (8, 8))
TEST_CELLS = ((4, 0), (4, 128), (16, 0), (16, 128))
_TRAIN_MENTIONS = np.array([(a,o,v) for a in range(8) for o in range(8) if (a+o)%4 for v in range(8)], np.int32)

# Opaque atomic names, not actor*8+occasion addresses or contiguous slot IDs.
# Object mentions also differ from the zero-based output classes 0..7.
ACTOR_TOKENS = (47, 23, 59, 35, 53, 29, 41, 31)
OCCASION_TOKENS = (86, 62, 92, 68, 80, 56, 74, 65)
OBJECT_TOKENS = (115, 94, 124, 103, 121, 97, 109, 100)

(_KEPT, _THE, _AT, _WITH, _CARE, _DISCUSSED, _INTEREST, _NOTICED,
 _WHICH, _OBJECT, _DID, _KEEP, _DURING, _QUESTION) = range(8, 22)
TOKEN_TO_WORD = dict(zip(range(8, 22), (
    'kept', 'the', 'at', 'with', 'care', 'discussed', 'interest', 'noticed',
    'which', 'object', 'did', 'keep', 'during', '?',
)))
TOKEN_TO_WORD.update(zip(ACTOR_TOKENS,
                         ('Aro', 'Bex', 'Cilu', 'Daro', 'Evi', 'Fenn', 'Goro', 'Hali')))
TOKEN_TO_WORD.update(zip(OCCASION_TOKENS,
                         ('Luma', 'Mera', 'Niso', 'Pavo', 'Qira', 'Runi', 'Savo', 'Teli')))
TOKEN_TO_WORD.update(zip(OBJECT_TOKENS,
                         ('shell', 'coin', 'ribbon', 'stone', 'cup', 'feather', 'ring', 'leaf')))


class Panel(NamedTuple):
    blocks: np.ndarray   # int32 [family, world=2, live+gap, token=8]
    queries: np.ndarray  # int32 [family, query=2, token=8]; shared across worlds
    support: np.ndarray  # int32 [family, query=2]; shared positional support
    answers: np.ndarray  # int32 [family, world=2, query=2]; output classes 0..7


def gloss(tokens):
    """Render one controlled clause (or flattened sequence) for human inspection."""
    return ' '.join(TOKEN_TO_WORD[int(token)] for token in np.asarray(tokens).reshape(-1))


def _rng(draw, split_id, family, stream):
    return np.random.default_rng(np.random.SeedSequence((draw, split_id, family, stream)))


def _pairs(split, rng, live):
    a0, a1 = (int(a) for a in rng.choice(8, size=2, replace=False))
    if split == 'held':
        o0 = int(rng.choice([o for o in range(8) if (a0 + o) % 4 == 0]))
        o1 = int(rng.choice([o for o in range(8) if o != o0 and (a1 + o) % 4 == 0]))
    else:
        allowed = [o for o in range(8) if (a0 + o) % 4 != 0 and (a1 + o) % 4 != 0]
        o0, o1 = (int(o) for o in rng.choice(allowed, size=2, replace=False))
    # The first two are the queried diagonal; the other two are both kinds of lure.
    pairs = [(a0, o0), (a1, o1), (a0, o1), (a1, o0)]
    rectangle = set(pairs)
    pool = [(a, o) for a in range(8) for o in range(8)
            if (a, o) not in rectangle and (split == 'held' or (a + o) % 4 != 0)]
    pairs.extend(pool[int(i)] for i in rng.permutation(len(pool))[:live - 4])
    return np.asarray(pairs, dtype=np.int32)


def _clause(actor, occasion, value, order, *, background=False):
    actor, occasion, value = (ACTOR_TOKENS[actor], OCCASION_TOKENS[occasion],
                               OBJECT_TOKENS[value])
    verb = (_DISCUSSED if order == 0 else _NOTICED) if background else _KEPT
    tail = _INTEREST if background else _CARE
    if order == 0:
        return (actor, verb, _THE, value, _AT, occasion, _WITH, tail)
    return (_AT, occasion, actor, verb, _THE, value, _WITH, tail)


def _question(actor, occasion, order):
    actor, occasion = ACTOR_TOKENS[actor], OCCASION_TOKENS[occasion]
    if order == 0:
        return (_WHICH, _OBJECT, _DID, actor, _KEEP, _DURING, occasion, _QUESTION)
    if order == 1:
        return (_DURING, occasion, _WHICH, _OBJECT, _DID, actor, _KEEP, _QUESTION)
    # Final-only rearrangement; every lexeme occurs in both training orders.
    return (_WHICH, _OBJECT, _DURING, occasion, _DID, actor, _KEEP, _QUESTION)


def make_panel(split, draw, families, live, gap, *, shift=False):
    """Return paired scenes, two shared questions, support positions, and labels.

    ``gap`` counts appended eight-token nonbinding utterances, not raw tokens.
    For train/dev/seen every mention (including background) excludes ``(actor+occasion) % 4 == 0``.
    For held, both queried pairs satisfy that predicate; all other live pairs remain
    distinct but are unrestricted.  ``shift=True`` is reserved for seen/held.
    """
    if split not in SPLITS:
        raise ValueError(f'unknown split {split!r}; expected one of {SPLITS}')
    draw, families, live, gap = map(index, (draw, families, live, gap))
    if draw < 0 or families < 1 or not 4 <= live <= 32 or gap < 0:
        raise ValueError('expected nonnegative draw/gap, positive families, and 4 <= live <= 32')
    if shift and split not in ('seen', 'held'):
        raise ValueError('shift is a final-only question order for seen/held')

    blocks = np.empty((families, 2, live + gap, 8), dtype=np.int32)
    queries = np.empty((families, 2, 8), dtype=np.int32)
    support = np.empty((families, 2), dtype=np.int32)
    answers = np.empty((families, 2, 2), dtype=np.int32)
    split_id = SPLITS.index(split)
    for family in range(families):
        pairs = _pairs(split, _rng(draw, split_id, family, 0), live)
        object_rng = _rng(draw, split_id, family, 1)
        values = np.empty(live, dtype=np.int32)
        values[:2] = object_rng.choice(8, size=2, replace=False)
        values[2:] = object_rng.integers(8, size=live - 2, dtype=np.int32)
        positions = _rng(draw, split_id, family, 2).permutation(live)
        support[family] = np.argsort(positions)[:2]

        surface_rng = _rng(draw, split_id, family, 3)
        statement_orders = surface_rng.integers(2, size=(8, 8), dtype=np.int32)
        question_orders = surface_rng.integers(2, size=(8, 8), dtype=np.int32)
        background_orders = surface_rng.integers(2, size=(8, 8, 8), dtype=np.int32)
        for position, pair_index in enumerate(positions):
            actor, occasion = pairs[pair_index]
            blocks[family, 0, position] = _clause(
                actor, occasion, values[pair_index], statement_orders[actor, occasion])

        # One tuple draw per row: changing G cannot change an earlier background row.
        noise = _rng(draw, split_id, family, 4)
        background = (noise.integers(8, size=(gap, 3), dtype=np.int32) if split == 'held' else
                      _TRAIN_MENTIONS[noise.integers(len(_TRAIN_MENTIONS), size=gap, dtype=np.int32)])
        for offset, (actor, occasion, value) in enumerate(background):
            blocks[family, 0, live + offset] = _clause(
                actor, occasion, value, background_orders[actor, occasion, value], background=True)
        blocks[family, 1] = blocks[family, 0]
        answers[family, 0] = values[:2]
        answers[family, 1] = values[1::-1]
        for query, (actor, occasion) in enumerate(pairs[:2]):
            position = support[family, query]
            object_position = 3 if statement_orders[actor, occasion] == 0 else 5
            blocks[family, 1, position, object_position] = OBJECT_TOKENS[values[1 - query]]
            order = 2 if shift else question_orders[actor, occasion]
            queries[family, query] = _question(actor, occasion, order)
    return Panel(blocks, queries, support, answers)
