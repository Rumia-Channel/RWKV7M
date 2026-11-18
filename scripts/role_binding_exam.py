"""Public Japanese logistics and classroom-register problems over marked tokens.

Advanced material uses only elementary-training dyads and three public reads.
The held classroom form uses two reads and cycle-distance-two root/terminal
pairs. Each family is a four-world root-by-terminal intervention, not four
independent examples. Only the two effective records at each intervened dyad
change; every record position keeps exactly the same word bag.

The Japanese rendering is a readable gloss of this controlled token grammar,
not a claim that the policy understands unrestricted Japanese. Reference traces
are annotations: ``public_document`` deliberately exports none of them.
"""
from operator import index
from typing import NamedTuple

import numpy as np

from role_binding_data import (NAME_TOKENS, OBJECT_TOKENS, SEEN_ORDERS,
                               SHIFT_ORDERS, TOKEN_TO_JAPANESE, question,
                               statement)


ADV_DRAWS = (764101, 764103, 764107, 764109)
MID_DRAWS = (766101, 766103, 766107, 766109, 766113)
FORMS = ('advanced', 'midterm')
WORLD_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))

_ALL_PAIRS = tuple((a, b) for a in range(8) for b in range(a + 1, 8))
_HELD_PAIRS = tuple(pair for pair in _ALL_PAIRS if pair[1] - pair[0] in (2, 6))
_TRAIN_PAIRS = tuple(pair for pair in _ALL_PAIRS if pair not in _HELD_PAIRS)


class ExamPanel(NamedTuple):
    blocks: np.ndarray      # int32 [family, world=4, records, token=8]
    queries: np.ndarray     # int32 [family, start=2, token=8], shared by worlds
    cards: np.ndarray       # int32 [family, output class=8, token=8]
    answers: np.ndarray     # int32 [family, world=4, start=2]
    support: np.ndarray     # int32 [family, world=4, start=2, hop=reads]
    values: np.ndarray      # int32 [family, world=4, start=2, hop=reads]
    reads: int
    card_order: np.ndarray  # int32 [family, printed row=8], output classes


def _rng(draw, form, family, stream):
    return np.random.default_rng(np.random.SeedSequence(
        (draw, FORMS.index(form), family, stream)))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _pairs(form, rng):
    """Choose task dyads, then genuine shared-name foils, without reuse."""
    pool = _TRAIN_PAIRS if form == 'advanced' else _ALL_PAIRS
    targets = _TRAIN_PAIRS if form == 'advanced' else _HELD_PAIRS
    count = 3 if form == 'advanced' else 2
    stages = []
    for _ in range(count):
        stages.append(_pick(rng, tuple(pair for pair in targets if pair not in stages)))
    foils = []
    for target in (stages[0], stages[-1]):
        candidates = tuple(pair for pair in pool if pair not in stages + foils
                           and set(pair) & set(target))
        foils.append(_pick(rng, candidates))
    if form == 'advanced':
        foils.append(_pick(rng, tuple(pair for pair in pool if pair not in stages + foils)))
    return tuple(stages), tuple(foils)


def _directions(pair, rng):
    return (pair, pair[::-1]) if int(rng.integers(2)) == 0 else (pair[::-1], pair)


def _records(form, stage_dirs, stage_values, foil_dirs, rng, layout):
    """Return semantic records; the first two are superseded terminal entries."""
    terminal = stage_dirs[-1]
    objects = tuple(value for layer in stage_values for value in layer)
    remaining = tuple(value for value in range(8) if value not in objects)
    provisional = tuple(int(value) for value in rng.permutation(remaining)[:2])
    misleading = (tuple(int(value) for value in rng.permutation(
        tuple(value for value in remaining if value not in provisional))[:2])
        if form == 'midterm' else stage_values[0])

    def record(direction, obj, role, binding=True):
        return (*direction, int(obj), binding, role)

    prefix = [record(direction, obj, 'provisional')
              for direction, obj in zip(terminal, provisional, strict=True)]
    prefix = [prefix[int(i)] for i in layout.permutation(2)]
    stages = [[record(direction, obj, 'root' if hop == 0 else
                      ('terminal' if hop == len(stage_dirs) - 1 else 'intermediate'))
               for direction, obj in zip(directions, values, strict=True)]
              for hop, (directions, values) in enumerate(zip(stage_dirs, stage_values, strict=True))]
    foils = [record(direction, rng.integers(8), 'foil')
             for directions in foil_dirs for direction in directions]
    if form == 'advanced':
        body = [row for layer in stages for row in layer] + foils
        body = [body[int(i)] for i in layout.permutation(len(body))]
    else:
        # The classroom register has two later corrections interleaved with a
        # morning check, rather than the advanced exercise's shuffled table.
        other = stages[0] + foils
        other = [other[int(i)] for i in layout.permutation(len(other))]
        corrections = [stages[-1][int(i)] for i in layout.permutation(2)]
        split = int(layout.integers(1, 4))
        body = other[:split] + corrections[:1] + other[split:5] + corrections[1:] + other[5:]
    tail = [record(direction, obj, 'plan', binding=False)
            for direction, obj in zip(terminal, misleading, strict=True)]
    tail = [tail[int(i)] for i in layout.permutation(2)]
    return prefix + body + tail


def _card_directions(stage_dirs, stage_values, foil_dirs, rng):
    """All eight public labels ask real live questions, including wrong branches."""
    cards = [None] * 8
    for values, following in zip(stage_values[:-1], stage_dirs[1:], strict=True):
        # Independent branch assignments prevent a fixed class/direction rule.
        following = following if int(rng.integers(2)) == 0 else following[::-1]
        for obj, direction in zip(values, following, strict=True):
            cards[obj] = direction
    used = {direction for direction in cards if direction is not None}
    available = tuple(direction for directions in stage_dirs + foil_dirs
                      for direction in directions if direction not in used)
    available = [available[int(i)] for i in rng.permutation(len(available))]
    for obj, direction in zip((i for i, row in enumerate(cards) if row is None), available):
        cards[obj] = direction
    return tuple(cards)


def make_exam(form, draw, families, gap, *, familiar=False):
    """Build an independent advanced-train or held classroom quartet panel.

    There are six/four live bidirectional dyads (12/8 effective facts), plus
    two superseded terminal records and two later, nonbinding terminal plans.
    ``gap`` appends harmless plans. Separate family, layout, phrase, card, and
    background streams preserve prefixes when families or gaps are enlarged.
    ``familiar=True`` only rerenders midterm semantics in seen phrase orders.
    No hidden object-to-question mapping is needed beyond the public card.
    """
    if form not in FORMS:
        raise ValueError(f'unknown exam form {form!r}; expected one of {FORMS}')
    try:
        draw, families, gap = map(index, (draw, families, gap))
    except TypeError as error:
        raise ValueError('draw, families and gap must be integers') from error
    if draw < 0 or families < 1 or gap < 0:
        raise ValueError('expected nonnegative draw/gap and positive families')
    if familiar and form != 'midterm':
        raise ValueError('familiar is a diagnostic rerender of midterm only')

    reads = 3 if form == 'advanced' else 2
    core = (12 if form == 'advanced' else 8) + 4
    orders = SEEN_ORDERS if form == 'advanced' or familiar else SHIFT_ORDERS
    blocks = np.empty((families, 4, core + gap, 8), dtype=np.int32)
    queries = np.empty((families, 2, 8), dtype=np.int32)
    cards = np.empty((families, 8, 8), dtype=np.int32)
    support = np.empty((families, 4, 2, reads), dtype=np.int32)
    values = np.empty_like(support)
    card_order = np.empty((families, 8), dtype=np.int32)
    noise_pairs = _TRAIN_PAIRS if form == 'advanced' else _ALL_PAIRS

    for family in range(families):
        semantic = _rng(draw, form, family, 0)
        stages, foils = _pairs(form, semantic)
        stage_dirs = tuple(_directions(pair, semantic) for pair in stages)
        foil_dirs = tuple(_directions(pair, semantic) for pair in foils)
        objects = semantic.permutation(8)
        stage_values = tuple(tuple(int(value) for value in objects[2 * hop:2 * hop + 2])
                             for hop in range(reads))
        records = _records(form, stage_dirs, stage_values, foil_dirs, semantic,
                           _rng(draw, form, family, 1))
        following = _card_directions(stage_dirs, stage_values, foil_dirs,
                                     _rng(draw, form, family, 2))
        starts = stage_dirs[0] if int(semantic.integers(2)) == 0 else stage_dirs[0][::-1]

        # Phrase/template randomness never changes the selected task semantics.
        text = _rng(draw, form, family, 3)
        record_orders = tuple(orders[int(text.integers(3))] for _ in records)
        for query, direction in enumerate(starts):
            queries[family, query] = question(*direction, orders[int(text.integers(3))])
        for obj, direction in enumerate(following):
            cards[family, obj] = question(*direction, orders[int(text.integers(3))])
        card_order[family] = _rng(draw, form, family, 4).permutation(8)

        background = _rng(draw, form, family, 5)
        background_text = _rng(draw, form, family, 6)
        for _ in range(gap):
            direction = _pick(background, noise_pairs)
            if int(background.integers(2)):
                direction = direction[::-1]
            records.append((*direction, int(background.integers(8)), False, 'background'))
        record_orders += tuple(orders[int(background_text.integers(3))] for _ in range(gap))

        for world, (rootflip, terminalflip) in enumerate(WORLD_ORDER):
            effective = {}
            for position, ((giver, recipient, obj, binding, role), order) in enumerate(
                    zip(records, record_orders, strict=True)):
                if (rootflip and role == 'root') or (terminalflip and role == 'terminal'):
                    giver, recipient = recipient, giver
                blocks[family, world, position] = statement(giver, recipient, obj, order,
                                                            binding=binding)
                if binding:
                    effective[giver, recipient] = (obj, position)
            for query, start in enumerate(starts):
                direction = start
                for hop in range(reads):
                    obj, position = effective[direction]
                    support[family, world, query, hop] = position
                    values[family, world, query, hop] = obj
                    if hop + 1 < reads:
                        direction = following[obj]

    return ExamPanel(blocks, queries, cards, values[..., -1], support, values, reads, card_order)


def public_document(panel, family, world):
    """Export only visible raw records, two starts, eight labeled rows and H."""
    return {
        'blocks': panel.blocks[family, world].tolist(),
        'queries': panel.queries[family].tolist(),
        'cards': [{'on': int(OBJECT_TOKENS[int(obj)]),
                   'ask': panel.cards[family, int(obj)].tolist()}
                  for obj in panel.card_order[family]],
        'reads': int(panel.reads),
    }


def _japanese(tokens):
    return ''.join(TOKEN_TO_JAPANESE[int(token)] for token in tokens)


def render_exam(panel, family, world, form):
    """Print a complete answer-free story, preserving the public token evidence."""
    if form not in FORMS or panel.reads != (3 if form == 'advanced' else 2):
        raise ValueError('form must match the panel read count')
    public = public_document(panel, family, world)
    variant = int(panel.card_order[family, 0]) % 2
    if form == 'advanced':
        titles = ('入試演習：物流センターの配布記録', '入試演習：配送係の受け渡し照合')
        scenes = (
            '物流センターでは、八種類の品物を担当者どうしで配っています。以下は整理前の記録も含む搬送台帳です。',
            '配送係は、八種類の品物を別の担当者へ引き渡しました。混在した配布記録を照合して、指定された照会を処理してください。',
        )
    else:
        titles = ('単元の中テスト：備品係の申し送り', '単元の中テスト：部活動の備品登録')
        scenes = (
            '教室の備品係は、展示に使う八種類の品物を貸し借りしています。当番の短い申し送りを、記録番号の順に読んでください。',
            '部活動では、展示用の八種類の備品を生徒どうしで渡しています。次の連絡メモは、部員が当番へ伝えた内容を記録順に並べたものです。',
        )
    lines = [titles[variant], '', scenes[variant],
             '同じ「渡した人→受け取った人」の記録が複数あるときは、番号がいちばん大きい「渡した」の品物を採用します。後の有効な記録は前の記録の訂正です。',
             '「渡そうと考えた」は予定だけで、受け渡しや訂正として数えません。', '',
             f'【照会の手順：各問とも記録をちょうど{public["reads"]}回照会】',
             'まず、その問の出発の質問について、上の規則で品物を一つ求めます。',
             f'最後の照会でなければ、いま求めた品物を下の8行の索引で探し、その行の質問を次の照会に使います。これを繰り返し、{public["reads"]}回目に得た品物だけを解答欄に書きます。',
             '問1と問2は別々に出発します。索引の行の並び順には意味がありません。', '', '【受け渡しの記録】']
    if form == 'advanced':
        lines.extend(('| 記録番号 | 台帳の記載 |', '| --- | --- |'))
        lines.extend(f'| {position:02d} | {_japanese(row)} |'
                     for position, row in enumerate(public['blocks'], 1))
    else:
        speakers = ('当番', '部員') if variant else ('前の当番', '今日の当番')
        lines.extend(f'記録{position:02d}　{speakers[(position - 1) % 2]}「{_japanese(row)}」'
                     for position, row in enumerate(public['blocks'], 1))
    lines.extend(('', '【品物→次の質問の索引（全8行）】'))
    lines.extend(f'{_japanese((row["on"],))} → {_japanese(row["ask"])}'
                 for row in public['cards'])
    lines.extend(('', '【出発の質問】'))
    lines.extend(f'問{number}　{_japanese(row)}　　解答欄（　　　　　　　）'
                 for number, row in enumerate(public['queries'], 1))
    return '\n'.join(lines)
