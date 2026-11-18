"""Eight-token marked-phrase scenes with genuine giver/recipient counterfactuals.

Each row has three two-token phrases (name+GA, name+NI, object+WO), a
predicate, and punctuation. Both directions of every live unordered pair occur.
World 1 exchanges only the two names in both queried statements: every row's
bag, object, phrase order, and position stays fixed, but directed truth changes.

Cycle-distance-two name pairs are withheld from *all* train/dev/seen utterances.
Held scenes query those pairs; ``shift=True`` additionally changes every clause
and question from even to odd phrase permutations, without changing semantics.
Independent per-family streams preserve semantic prefixes across live sizes and
background prefixes across gaps. Raw IDs 0..7 and 125..127 remain unused/reserved.
"""
from operator import index
from typing import NamedTuple

import numpy as np


SPLITS = ('train', 'dev', 'seen', 'held')
TRAIN_DRAWS = (761101, 761103, 761107, 761109)
DEV_DRAWS = (762101, 762103)
TEST_DRAWS = (763101, 763103, 763107, 763109, 763113)
TRAIN_CELLS = ((4, 0), (4, 8), (8, 0), (8, 8))
TEST_CELLS = ((4, 0), (4, 128), (16, 0), (16, 128))

# Names share one lexicon in both roles; objects are not raw output-label IDs.
NAME_TOKENS = (47, 23, 59, 35, 53, 29, 41, 31)
OBJECT_TOKENS = (115, 94, 124, 103, 121, 97, 109, 100)
GA, NI, WO, HAND, CONSIDER, PERIOD, WHAT, ASK = range(8, 16)
SEEN_ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
SHIFT_ORDERS = ((0, 2, 1), (2, 1, 0), (1, 0, 2))
_ORDERS = SEEN_ORDERS + SHIFT_ORDERS

TOKEN_TO_WORD = dict(zip(range(8, 16), (
    'ga', 'ni', 'wo', 'handed', 'considered handing', '.', 'what', '?',
)))
TOKEN_TO_WORD.update(zip(NAME_TOKENS,
                         ('Aro', 'Bex', 'Cilu', 'Daro', 'Evi', 'Fenn', 'Goro', 'Hali')))
TOKEN_TO_WORD.update(zip(OBJECT_TOKENS,
                         ('shell', 'coin', 'ribbon', 'stone', 'cup', 'feather', 'ring', 'leaf')))
TOKEN_TO_JAPANESE = dict(zip(range(8, 16), (
    'が', 'に', 'を', '渡した', '渡そうと考えた', '。', '何', '？',
)))
TOKEN_TO_JAPANESE.update(zip(NAME_TOKENS,
                             ('アロ', 'ベクス', 'シル', 'ダロ', 'エヴィ', 'フェン', 'ゴロ', 'ハリ')))
TOKEN_TO_JAPANESE.update(zip(OBJECT_TOKENS,
                             ('貝', '硬貨', 'リボン', '石', 'カップ', '羽', '指輪', '葉')))

_ALL_PAIRS = tuple((a, b) for a in range(8) for b in range(a + 1, 8))
_HELD_PAIRS = tuple(pair for pair in _ALL_PAIRS if (pair[1] - pair[0]) in (2, 6))
_SEEN_PAIRS = tuple(pair for pair in _ALL_PAIRS if pair not in _HELD_PAIRS)
_ALL_DIRECTIONS = tuple(direction for pair in _ALL_PAIRS for direction in (pair, pair[::-1]))
_SEEN_DIRECTIONS = tuple(direction for pair in _SEEN_PAIRS for direction in (pair, pair[::-1]))


class Panel(NamedTuple):
    blocks: np.ndarray   # int32 [family, world=2, live+gap, token=8]
    queries: np.ndarray  # int32 [family, query=2, token=8]; shared across worlds
    support: np.ndarray  # int32 [family, world=2, query=2]; positions, not training labels
    answers: np.ndarray  # int32 [family, world=2, query=2]; object classes 0..7


def gloss(tokens, *, japanese=False):
    """Render raw elementary tokens with readable case markers or Japanese text."""
    words = TOKEN_TO_JAPANESE if japanese else TOKEN_TO_WORD
    return ' '.join(words[int(token)] for token in np.asarray(tokens).reshape(-1))


def _render(giver, recipient, object_token, order, predicate, ending):
    giver, recipient = map(index, (giver, recipient))
    order = tuple(map(index, order))
    if not 0 <= giver < 8 or not 0 <= recipient < 8:
        raise ValueError('name IDs must be in 0..7')
    if order not in _ORDERS:
        raise ValueError('order must be a permutation of (0, 1, 2)')
    phrases = ((NAME_TOKENS[giver], GA), (NAME_TOKENS[recipient], NI), (object_token, WO))
    return tuple(token for phrase in order for token in phrases[phrase]) + (predicate, ending)


def statement(giver, recipient, obj, order, *, binding=True):
    """Render semantic IDs 0..7 in a phrase permutation of giver/recipient/object."""
    obj = index(obj)
    if not 0 <= obj < 8:
        raise ValueError('object ID must be in 0..7')
    return _render(giver, recipient, OBJECT_TOKENS[obj], order,
                   HAND if binding else CONSIDER, PERIOD)


def question(giver, recipient, order):
    """Render a directed object question using the same public raw grammar."""
    return _render(giver, recipient, WHAT, order, HAND, ASK)


def _rng(draw, kind_id, family, stream):
    return np.random.default_rng(np.random.SeedSequence((draw, kind_id, family, stream)))


def _family_pairs(kind, rng):
    targets = _HELD_PAIRS if kind == 'held' else _SEEN_PAIRS
    pool = _ALL_PAIRS if kind == 'held' else _SEEN_PAIRS
    target = targets[int(rng.integers(len(targets)))]
    foils = tuple(pair for pair in pool if pair != target and
                  (pair[0] in target or pair[1] in target))
    foil = foils[int(rng.integers(len(foils)))]
    remainder = tuple(pair for pair in pool if pair not in (target, foil))
    return (target, foil) + tuple(remainder[int(i)] for i in rng.permutation(len(remainder)))


def make_panel(kind, draw, families, live, gap, *, shift=False):
    """Generate paired raw worlds, shared directed questions, support, and labels.

    ``live`` is even: 4..40 for train/dev/seen, 4..56 for held. Every unordered
    live pair contributes both directions; a nuisance pair shares a target name.
    ``gap`` appends nonbinding rows. Only held accepts ``shift=True``. Neither
    support positions nor role labels form part of the visible utterances.
    """
    if kind not in SPLITS:
        raise ValueError(f'unknown kind {kind!r}; expected one of {SPLITS}')
    draw, families, live, gap = map(index, (draw, families, live, gap))
    maximum = 56 if kind == 'held' else 40
    if draw < 0 or families < 1 or gap < 0 or not 4 <= live <= maximum or live % 2:
        raise ValueError(f'expected nonnegative draw/gap, positive families, and even 4 <= live <= {maximum}')
    if shift and kind != 'held':
        raise ValueError('shift is reserved for held pairs with unseen clause and question orders')

    blocks = np.empty((families, 2, live + gap, 8), dtype=np.int32)
    queries = np.empty((families, 2, 8), dtype=np.int32)
    support = np.empty((families, 2, 2), dtype=np.int32)
    answers = np.empty((families, 2, 2), dtype=np.int32)
    kind_id = SPLITS.index(kind)
    orders = SHIFT_ORDERS if shift else SEEN_ORDERS
    noise_pairs = _ALL_DIRECTIONS if kind == 'held' else _SEEN_DIRECTIONS
    for family in range(families):
        pairs = _family_pairs(kind, _rng(draw, kind_id, family, 0))
        directions = tuple(direction for pair in pairs[:live // 2]
                           for direction in (pair, pair[::-1]))
        orientation = int(_rng(draw, kind_id, family, 1).integers(2))
        target_values = _rng(draw, kind_id, family, 2).choice(8, size=2, replace=False)
        values = _rng(draw, kind_id, family, 3).integers(8, size=(8, 8), dtype=np.int32)
        values[directions[0]], values[directions[1]] = target_values
        positions = _rng(draw, kind_id, family, 4).permutation(live)
        inverse = np.argsort(positions)
        support[family, 0] = (inverse[orientation], inverse[1 - orientation])
        support[family, 1] = support[family, 0, ::-1]
        answers[family, 0] = (target_values[orientation], target_values[1 - orientation])
        answers[family, 1] = answers[family, 0, ::-1]

        clause_orders = _rng(draw, kind_id, family, 5).integers(3, size=(8, 8), dtype=np.int32)
        question_orders = _rng(draw, kind_id, family, 6).integers(3, size=2, dtype=np.int32)
        noise_orders = _rng(draw, kind_id, family, 7).integers(3, size=gap, dtype=np.int32)
        noise = _rng(draw, kind_id, family, 8).integers(len(noise_pairs) * 8, size=gap, dtype=np.int32)
        for position, pair_index in enumerate(positions):
            giver, recipient = directions[int(pair_index)]
            blocks[family, 0, position] = statement(
                giver, recipient, values[giver, recipient], orders[int(clause_orders[giver, recipient])])
        for offset, code in enumerate(noise):
            pair_index, obj = divmod(int(code), 8)
            giver, recipient = noise_pairs[pair_index]
            blocks[family, 0, live + offset] = statement(
                giver, recipient, obj, orders[int(noise_orders[offset])], binding=False)
        blocks[family, 1] = blocks[family, 0]
        name_a, name_b = (NAME_TOKENS[name] for name in pairs[0])
        for position in support[family, 0]:
            row = blocks[family, 1, int(position)]
            for column in (0, 2, 4):
                if row[column] == name_a:
                    row[column] = name_b
                elif row[column] == name_b:
                    row[column] = name_a
        for query, direction in enumerate((orientation, 1 - orientation)):
            giver, recipient = directions[direction]
            queries[family, query] = question(
                giver, recipient, orders[int(question_orders[query])])
    return Panel(blocks, queries, support, answers)
