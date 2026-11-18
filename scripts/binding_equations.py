"""Grounded equations: learned operand retrieval followed by an exact calculator.

These are new, ordered arithmetic programs over synthetic memory key IDs, not the
original unconstrained text-LM/generation task and not end-to-end learned arithmetic.
Program construction sees only seed, family, keys, and depth: never world labels,
retrieved values, predictions, gaps, or intervention arms. A public ``X16`` names
key 16; only ``answer_program`` binds it to a retrieved value token.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from random import Random

from symbolic_arithmetic import format_answer, solve_prompt


@dataclass(frozen=True, slots=True)
class Program:
    keys: tuple[int, int, int, int]
    statements: tuple[str, ...]


def make_programs(seed: int, family: int, keys: Sequence[int], depth: int) -> tuple[Program, ...]:
    """Partition distinct keys into deterministic, disjoint four-operand programs.

    Key order is canonicalized before the local shuffle. Grouping and operand roles
    are stable across depths as well as across worlds, gaps, and retrieval arms.
    ``depth`` counts arithmetic update stages, including final A. Operand setup,
    snapshot copies, and the final ACD query product do not add stages.
    """
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (seed, family, depth)):
        raise ValueError("seed, family, and depth must be integers")
    if family < 0 or depth < 1:
        raise ValueError("family must be nonnegative and depth must be positive")
    key_ids = tuple(keys)
    if (not key_ids or len(key_ids) % 4
            or any(isinstance(k, bool) or not isinstance(k, Integral) or not 16 <= k <= 31
                   for k in key_ids)
            or len(set(key_ids)) != len(key_ids)):
        raise ValueError("keys must be distinct IDs 16..31, grouped in fours")
    ordered = sorted(int(k) for k in key_ids)
    Random(f"binding-equations-v1:{int(seed)}:{int(family)}").shuffle(ordered)
    programs = []
    for start in range(0, len(ordered), 4):
        group = tuple(ordered[start:start + 4])
        b, c, d, v = group
        rows = [f"B = X{b}", f"C = X{c}", f"D = X{d}", f"V = i*X{v}"]
        if depth > 1:
            rows.append("C0 = C")
        if depth > 2:
            rows.append("D0 = D")
        # Snapshots retain their original values after C and D are overwritten.
        # The alternating signs preserve C in quadrant I and D in quadrant IV
        # for positive operands. Final A therefore has no cancelling input path.
        for stage in range(depth - 1):
            rows.append("C = C + DV + C0" if stage % 2 == 0 else "D = D - C*V + D0")
        rows.append("A = B + C + DV")
        # C and D do not depend on B, so ACD's B*CD term also retains dependence
        # on all updated operands; CD is nonzero. At depth one V enters via A.
        programs.append(Program(group, tuple(rows)))
    return tuple(programs)


def answer_program(program: Program, predicted: Mapping[int, int]) -> tuple[str | None, str | None]:
    """Return canonical (A, ACD) using only the four retrieved value token IDs.

    Tokens 32..47 map to integers 1..16. Any required missing/noninteger/out-of-range
    value invalidates both answers; unrelated keys are ignored. No gold values or
    generator/reference answers are available to this score-neutral function.
    Malformed equation programs raise the existing interpreter's error rather
    than being disguised as retrieval misses.
    """
    bindings = []
    for key in program.keys:
        value = predicted.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral) or not 32 <= value <= 47:
            return None, None
        bindings.append(f"X{key} = {int(value) - 31}")
    body = "\n".join((
        "Mode: updates",
        "Execute assignments top to bottom; copies keep their values after later updates.",
        "Use exact Gaussian integers; adjacency means multiplication and i*i=-1.",
        "Statements:",
        *bindings,
        *program.statements,
    ))
    return (
        format_answer(solve_prompt(f"{body}\nQuery: A\nAnswer:")),
        format_answer(solve_prompt(f"{body}\nQuery: ACD\nAnswer:")),
    )
