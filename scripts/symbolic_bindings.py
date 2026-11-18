"""Deterministic exact-arithmetic families. The gold computation never parses the prompt."""

from hashlib import sha256
from random import Random

from symbolic_arithmetic import format_answer

MODES = ('updates', 'definitions')
QUERIES = ('A', 'ACD')
RULE = {
    'updates': 'Execute assignments from top to bottom. Each right side uses the values at that line. A later update does not change a previously computed variable.',
    'definitions': 'Equations define variables once. They can reference definitions appearing later. Order does not change their values.',
}


def add(x, y): return x[0] + y[0], x[1] + y[1]
def sub(x, y): return x[0] - y[0], x[1] - y[1]
def mul(x, y): return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]


# Every round reads both live registers and therefore extends the longest data-dependency path.
# Multipliers in the chain are bounded leaves; the three-factor query itself multiplies computed results.
OPS = (
    ('{x} + {y}V', lambda x, y, v, b: add(x, mul(y, v))),
    ('{y} - {x}V + B', lambda x, y, v, b: add(sub(y, mul(x, v)), b)),
    ('({x} - {y})V + B', lambda x, y, v, b: add(mul(sub(x, y), v), b)),
    ('{x} + ({y} - B)V', lambda x, y, v, b: add(x, mul(sub(y, b), v))),
    ('{y} + B - {x}', lambda x, y, v, b: sub(add(y, b), x)),
)


def _instance(seed, family, depth):
    rng = Random(f'symbolic-bindings-v1:{seed}:{family}:{depth}')
    while True:
        # Swap two equal-width literals, not names, query positions, or the bag of tokens.
        changed = rng.sample(range(1, 10), 2)
        b = (rng.choice((-4, -3, -2, 2, 3, 4)), 0)
        v = (0, rng.choice((-3, -2, -1, 1, 2, 3, 9)))
        d = (rng.choice((-5, -3, -2, 2, 3, 5)), rng.randrange(-2, 3))
        rounds = [(rng.randrange(2), rng.randrange(len(OPS))) for _ in range(depth - 1)]
        final = rng.randrange(len(OPS))
        gold = []
        for c in changed:
            regs = [(c, 0), d]
            for target, op in rounds:
                regs[target] = OPS[op][1](regs[target], regs[1-target], v, b)
            a = OPS[final][1](*regs, v, b)
            gold.append((a, mul(mul(a, regs[0]), regs[1])))
        if (all(x != y for x, y in zip(*gold)) and
                all(a != acd and (0, 0) not in (a, acd) for a, acd in gold)):
            return changed, d, v, b, rounds, final, gold


def _core(instance, mode, world, order_seed):
    changed, d, v, b, rounds, final, _ = instance
    totals = [sum(t == j for t, _ in rounds) for j in (0, 1)]
    counts = [0, 0]
    def reg(j):
        name = ('C', 'D')[j]
        return name if mode == 'updates' or counts[j] == totals[j] else f'{name}{counts[j]}'
    rows = [f'{reg(0)} = {changed[world]}', f'K = {changed[1-world]}', f'{reg(1)} = {format_answer(d)}',
            f'V = {format_answer(v)}', f'B = {format_answer(b)}']
    rng = Random(order_seed); rng.shuffle(rows)
    for step, (target, op) in enumerate(rounds):
        rhs = OPS[op][0].format(x=reg(target), y=reg(1-target))
        counts[target] += 1
        rows.append(f'{reg(target)} = {rhs}')
        if step % 3 == 1:  # A plausible but dead equation mixed into the supporting evidence.
            rows.append(f'H{step} = ({reg(0)} - K){reg(1)} + B')
    rows.append('A = ' + OPS[final][0].format(x=reg(0), y=reg(1)))
    if mode == 'definitions': rng.shuffle(rows)
    return rows


def _noise(seed, n):
    rng = Random(seed)
    for j in range(n):
        x, y = rng.sample(('A', 'B', 'C', 'D', 'V', 'K'), 2)
        yield f'{rng.choice(("P", "Q", "R"))}{j} = ' + rng.choice(OPS)[0].format(x=x, y=y)


def cases(seed, families, depths, gaps, modes=MODES):
    """Yield (public, private) rows. Gaps are trailing *statements*, not tokenizer-dependent distances."""
    if families < 1 or not depths or any(not 1 <= d <= 256 for d in depths):
        raise ValueError('use at least one family and dependency depths 1..256')
    if 0 not in gaps or any(g < 0 for g in gaps) or not modes or any(m not in MODES for m in modes):
        raise ValueError('include gap zero, nonnegative gaps and known modes')
    if any(len(set(v)) != len(v) for v in (depths, gaps, modes)):
        raise ValueError('duplicate depth, gap or mode')
    for depth in depths:
        for family in range(families):
            instance = _instance(seed, family, depth)
            order_seed = f'symbolic-layout-v1:{seed}:{family}:{depth}'
            noise = tuple(_noise(order_seed + ':noise', max(gaps)))
            for mode in modes:
                intro = (f'Mode: {mode}\n{RULE[mode]}\ni*i = -1. Juxtaposition means multiplication; '
                         'a capital letter with trailing digits is one name. Return just the simplified Gaussian integer '
                         '(for example -3+2i) on one line.\nStatements:\n')
                for world in (0, 1):
                    core = '\n'.join(_core(instance, mode, world, order_seed))
                    for gap in gaps:
                        program = core + (('\n' + '\n'.join(noise[:gap])) if gap else '')
                        for query, answer in zip(QUERIES, instance[-1][world], strict=True):
                            prompt = f'{intro}{program}\nQuery: {query}\nAnswer:'
                            # IDs bind both input and independent family (two identical texts need not be the same draw).
                            uid = sha256(f'{order_seed}:{mode}:{world}:{gap}:{query}\0{prompt}'.encode()).hexdigest()[:32]
                            yield {'id': uid, 'prompt': prompt}, dict(id=uid, seed=seed, family=family, mode=mode,
                                depth=depth, gap=gap, world=world, query=query, answer=format_answer(answer))
