from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def arithmetic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return import_module('symbolic_arithmetic')


def prompt(statements, query='A', mode='updates'):
    return '\n'.join([
        f'Mode: {mode}',
        'Use exact Gaussian integers; adjacency means multiplication and i*i=-1.',
        'Statements:',
        *statements,
        f'Query: {query}',
        'Answer:',
    ])


@pytest.mark.parametrize('query, expected', [
    ('A', (4, 0)), ('B', (3, 0)), ('C', (1, 18)),
    ('D', (2, 0)), ('V', (0, 9)), ('ACD', (8, 144)),
])
def test_hand_derived_update_sample(arithmetic, query, expected):
    statements = ['B=3', 'C=1', 'D=2', 'V=9i', 'A=B+C', 'C=C+DV']
    assert arithmetic.solve_prompt(prompt(statements, query)) == expected


@pytest.mark.parametrize('query, expected', [
    ('A', (4, 18)), ('B', (3, 0)), ('C', (1, 18)),
    ('D', (2, 0)), ('V', (0, 9)), ('ACD', (-640, 180)),
])
def test_hand_derived_unordered_dag_sample(arithmetic, query, expected):
    # Unlike updates, A reads the unique final definition of C: C0 + D*V.
    statements = ['A=B+C', 'C=C0+DV', 'V=9i', 'B=3', 'D=2', 'C0=1']
    assert arithmetic.solve_prompt(prompt(statements, query, 'definitions')) == expected


def test_ordered_reassignment_reads_old_value_and_keeps_snapshots(arithmetic):
    statements = ['A=2+i', 'B=A', 'A=A(1-i)', 'A=A+1', 'C=AB']
    assert arithmetic.solve_prompt(prompt(statements, 'A')) == (4, -1)
    assert arithmetic.solve_prompt(prompt(statements, 'B')) == (2, 1)
    assert arithmetic.solve_prompt(prompt(statements, 'C')) == (9, 2)
    with pytest.raises(ValueError, match='duplicate'):
        arithmetic.solve_prompt(prompt(statements, 'C', 'definitions'))


@pytest.mark.parametrize('expression, expected', [
    ('2+3*4', (14, 0)),
    ('(2+3)4', (20, 0)),
    ('-2(3+4i)+9i', (-6, 1)),
    ('--(i(-i))', (1, 0)),
    ('-+-i', (0, 1)),
    ('2*-3--4', (-2, 0)),
    ('(1+i)(1-i)', (2, 0)),
    ('2i(3-i)-i*i', (3, 6)),
    ('1-2-3', (-4, 0)),
    ('2(3)(4)', (24, 0)),
])
def test_precedence_unary_and_implicit_products(arithmetic, expression, expected):
    assert arithmetic.solve_prompt(prompt([f'A={expression}'])) == expected


def test_indexed_variables_and_query_products(arithmetic):
    statements = ['A1=2', 'B2=3', 'A10=A1B2+2(i+1)']
    assert arithmetic.solve_prompt(prompt(statements, 'A10A1')) == (16, 4)


def test_integer_arithmetic_is_exact_beyond_float_precision(arithmetic):
    n = 9007199254740993
    statements = [f'A={n}+i', 'B=A(A-i)']
    assert arithmetic.solve_prompt(prompt(statements, 'B')) == (n * n, n)


def test_forward_references_only_work_in_definitions_mode(arithmetic):
    statements = ['A=B+2', 'B=3i']
    assert arithmetic.solve_prompt(prompt(statements, mode='definitions')) == (2, 3)
    with pytest.raises(ValueError, match='undefined|initialization'):
        arithmetic.solve_prompt(prompt(statements))


@pytest.mark.parametrize('mode, statements', [
    ('updates', ['A=A+1']),
    ('updates', ['A=B', 'B=2']),
    ('updates', ['A=1', 'B=Z']),
    ('updates', ['B=3', 'D=2', 'V=9i', 'A=B+C', 'C=C+DV']),
    ('definitions', ['A=A']),
    ('definitions', ['A=B', 'B=A']),
    ('definitions', ['A=1', 'B=C', 'C=B']),
    ('definitions', ['A=1', 'B=Z']),
    ('definitions', ['A=1', 'A=2']),
    ('definitions', ['B=3', 'C=C+DV', 'D=2', 'V=9i', 'A=B+C']),
])
def test_reject_undefined_initialization_cycles_and_duplicates(arithmetic, mode, statements):
    with pytest.raises(ValueError):
        arithmetic.solve_prompt(prompt(statements, mode=mode))


@pytest.mark.parametrize('expression', [
    '', ' ', '+', '*2', '2*', '2**2', '2/1', '2^3', '1.0', '1e3',
    '()', '(1+2', '1+2)', '1+()', '2+[1]', 'True', 'I', '\u22121',
    '__import__("os")', 'A.__class__', '1;2', '1 # comment',
])
def test_reject_unknown_or_incomplete_expression_syntax(arithmetic, expression):
    with pytest.raises(ValueError):
        arithmetic.solve_prompt(prompt([f'A={expression}']))


@pytest.mark.parametrize('broken', [
    'Mode: other\nStatements:\nA=1\nQuery: A\nAnswer:',
    ' Mode: updates\nStatements:\nA=1\nQuery: A\nAnswer:',
    'Mode: updates\nA=1\nQuery: A\nAnswer:',
    'Mode: updates\nStatements:\nQuery: A\nAnswer:',
    'Mode: updates\nStatements:\nA=1\nQuery: A',
    'Mode: updates\nStatements:\na=1\nQuery: A\nAnswer:',
    'Mode: updates\nStatements:\nA=1\nQuery: A+1\nAnswer:',
    'Mode: updates\nStatements:\nA=1\nQuery: 1\nAnswer:',
    'Mode: updates\nStatements:\nA=1\nQuery: A\nAnswer: 1',
])
def test_prompt_boundaries_are_explicit(arithmetic, broken):
    with pytest.raises(ValueError):
        arithmetic.solve_prompt(broken)


@pytest.mark.parametrize('mode', ['updates', 'definitions'])
def test_window_keeps_last_assignment_lines_not_instructions_or_query(arithmetic, mode):
    text = prompt(['A=100', 'B=4', 'C=B+1'], 'C', mode)
    assert arithmetic.solve_prompt(text) == (5, 0)
    assert arithmetic.solve_prompt(text, window=2) == (5, 0)
    assert arithmetic.solve_prompt(text, window=100) == (5, 0)
    for window in (0, 1):
        with pytest.raises(ValueError, match='undefined|initialization'):
            arithmetic.solve_prompt(text, window=window)


@pytest.mark.parametrize('mode', ['updates', 'definitions'])
def test_missing_distractor_does_not_hide_a_known_answer(arithmetic, mode):
    text = prompt(['B=8', 'C=B+1', 'A=3i'], 'A', mode)
    assert arithmetic.solve_prompt(text, window=2) == (0, 3)


def test_unknown_overwrite_does_not_fall_back_to_stale_value(arithmetic):
    text = prompt(['B=8', 'A=2', 'A=A+B'], 'A')
    with pytest.raises(ValueError, match='undefined|initialization'):
        arithmetic.solve_prompt(text, window=2)


@pytest.mark.parametrize('window', [-1, 1.5, True, '2'])
def test_invalid_window_is_rejected(arithmetic, window):
    with pytest.raises(ValueError, match='window'):
        arithmetic.solve_prompt(prompt(['A=1']), window=window)


def test_long_unordered_dependency_chain_does_not_recurse(arithmetic):
    count = 2500
    statements = ['A0=1', *(f'A{index}=A{index - 1}+1' for index in range(1, count))]
    assert arithmetic.solve_prompt(
        prompt(list(reversed(statements)), f'A{count - 1}', 'definitions')
    ) == (count, 0)


def test_deep_expression_parentheses_do_not_recurse(arithmetic):
    expression = '(' * 1500 + '1+i' + ')' * 1500
    assert arithmetic.solve_prompt(prompt([f'A={expression}'])) == (1, 1)


@pytest.mark.parametrize('text, expected', [
    ('0', (0, 0)), ('-0', (0, 0)), ('+12', (12, 0)), (' - 12 ', (-12, 0)),
    ('i', (0, 1)), ('+i', (0, 1)), ('-i', (0, -1)), ('\u2212 i', (0, -1)),
    (' 9 i ', (0, 9)), ('-0i', (0, 0)), ('- 12 i', (0, -12)),
    ('3+2i', (3, 2)), ('-3-2i', (-3, -2)), ('+3+i', (3, 1)),
    ('\u22123\u2212i', (-3, -1)), (' \n+ 3 \u2212 4 i\t', (3, -4)),
    ('0003 + 0002 i', (3, 2)), ('3+0i', (3, 0)), ('0-1i', (0, -1)),
])
def test_parse_complete_gaussian_literal_aliases(arithmetic, text, expected):
    assert arithmetic.parse_answer(text) == expected


@pytest.mark.parametrize('text', [
    '', ' ', 'Answer: 2+i', '2+i\nThanks', '(2+i)', 'i+2', '1+2',
    '1 + -2i', '1++2i', '2*i', 'i*i', '1 2', '1 2+3i', '1+2 3i',
    '1.0', '1e2', 'NaN', '2j', 'I', '2ii', '2i+3i', '1/2',
    '1\n2', '1+2i\n3+4i', '\u20142i', '\uff12i', '[2, 3]', '+', '-',
    '1\n+2i',
])
def test_near_miss_answers_are_not_expressions_or_prose(arithmetic, text):
    assert arithmetic.parse_answer(text) is None


@pytest.mark.parametrize('value, expected', [
    ((0, 0), '0'), ((12, 0), '12'), ((-12, 0), '-12'),
    ((0, 1), 'i'), ((0, -1), '-i'), ((0, 9), '9i'), ((0, -9), '-9i'),
    ((3, 1), '3+i'), ((3, -1), '3-i'), ((-3, 2), '-3+2i'), ((-3, -2), '-3-2i'),
])
def test_canonical_format_and_round_trip(arithmetic, value, expected):
    assert arithmetic.format_answer(value) == expected
    assert arithmetic.parse_answer(expected) == value
