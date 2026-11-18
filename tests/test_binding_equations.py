from collections import defaultdict
from importlib import import_module
from pathlib import Path
import re

import pytest


@pytest.fixture
def equations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return import_module('binding_equations')


def predicted(program, operands):
    return {key: value + 31 for key, value in zip(program.keys, operands, strict=True)}


def test_independent_hand_calculated_gaussian_example(equations):
    program, = equations.make_programs(17, 3, [16, 17, 18, 19], 2)
    values = predicted(program, (3, 1, 2, 9))
    before = values.copy()
    # B=3, C0=C=1, D=2, V=9i. The update makes C=2+18i.
    # A=3+(2+18i)+18i=5+36i; ACD=(5+36i)(2+18i)*2=-1276+324i.
    assert equations.answer_program(program, values) == ('5+36i', '-1276+324i')
    assert values == before
    assert tuple(row for row in program.statements if row.startswith('V = ')) == (
        f'V = i*X{program.keys[3]}',
    )
    for name, key in zip(('B', 'C', 'D'), program.keys, strict=False):
        assert f'{name} = X{key}' in program.statements
    assert program.statements[-1] == 'A = B + C + DV'


def test_snapshots_survive_repeated_register_overwrites(equations):
    program, = equations.make_programs(17, 3, [16, 17, 18, 19], 4)
    # C0=1 and D0=2 stay fixed. Successive arithmetic writes give
    # C=2+18i, D=166-18i, C=165+1512i, then A=330+3006i.
    # CD=54606+248022i; multiply by A without floating-point arithmetic.
    assert program.statements.count('C0 = C') == program.statements.count('D0 = D') == 1
    assert program.statements.count('C = C + DV + C0') == 2
    assert equations.answer_program(program, predicted(program, (3, 1, 2, 9))) == (
        '330+3006i', '-727534152+245992896i',
    )


def test_superseded_retrieved_value_does_not_leave_a_stale_binding(equations):
    program, = equations.make_programs(17, 3, [16, 17, 18, 19], 2)
    values = predicted(program, (3, 1, 2, 9))
    assert equations.answer_program(program, values) == ('5+36i', '-1276+324i')
    values[program.keys[1]] = 33  # The current retrieval is 2, not the earlier 1.
    # C=4+18i; A=7+36i; ACD=(7+36i)(4+18i)*2=-1240+540i.
    assert equations.answer_program(program, values) == ('7+36i', '-1240+540i')
    del values[program.keys[1]]
    assert equations.answer_program(program, values) == (None, None)


@pytest.mark.parametrize('value, expected', [
    (32, ('3+2i', '4+7i')),
    (47, ('48+512i', '-2072576+458752i')),
])
def test_token_boundaries_map_to_one_and_sixteen(equations, value, expected):
    program, = equations.make_programs(11, 0, [16, 17, 18, 19], 2)
    assert equations.answer_program(program, dict.fromkeys(program.keys, value)) == expected


@pytest.mark.parametrize('slot', range(4))
@pytest.mark.parametrize('invalid', [None, -1, 0, 31, 48, 96, 32.0, '32', True, 32 + 0j])
def test_any_invalid_required_retrieval_invalidates_both_queries(equations, slot, invalid):
    program, = equations.make_programs(11, 0, [16, 17, 18, 19], 2)
    values = predicted(program, (3, 1, 2, 9))
    values[program.keys[slot]] = invalid
    assert equations.answer_program(program, values) == (None, None)


@pytest.mark.parametrize('slot', range(4))
def test_missing_retrieval_never_uses_an_implicit_mapping_default(equations, slot):
    program, = equations.make_programs(11, 0, [16, 17, 18, 19], 2)
    values = defaultdict(lambda: 32, predicted(program, (3, 1, 2, 9)))
    missing = program.keys[slot]
    del values[missing]
    assert equations.answer_program(program, values) == (None, None)
    assert missing not in values


def test_pairing_is_deterministic_across_a_changed_world(equations):
    keys = tuple(range(16, 24))
    programs = equations.make_programs(731, 2, keys, 8)
    assert len(programs) == 2
    assert all(isinstance(p.keys, tuple) and len(p.keys) == 4 for p in programs)
    assert set(programs[0].keys).isdisjoint(programs[1].keys)
    assert set(programs[0].keys + programs[1].keys) == set(keys)
    assert equations.make_programs(731, 2, tuple(reversed(keys)), 8) == programs
    for depth in (2, 24):
        assert tuple(p.keys for p in equations.make_programs(731, 2, keys, depth)) == (
            programs[0].keys, programs[1].keys,
        )
    world0 = {key: 32 + index for index, key in enumerate(keys)}
    world1 = dict(world0)
    world1[programs[0].keys[0]] = 47
    first0 = equations.answer_program(programs[0], world0)
    first1 = equations.answer_program(programs[0], world1)
    assert all(x != y for x, y in zip(first0, first1, strict=True))
    assert equations.answer_program(programs[1], world0) == equations.answer_program(programs[1], world1)
    # Different worlds/arms supply only retrievals; none can influence construction.
    assert equations.make_programs(731, 2, keys, 8) == programs
    world0[31] = None  # An unrelated invalid prediction cannot poison this program.
    assert equations.answer_program(programs[0], world0) == first0


@pytest.mark.parametrize('family', range(4))
@pytest.mark.parametrize('depth', [2, 8, 24])
def test_true_arithmetic_depth_and_four_operand_dependence(equations, family, depth):
    arithmetic = import_module('symbolic_arithmetic')
    for program in equations.make_programs(731, family, tuple(range(16, 24)), depth):
        # Independently follow the ordered, versioned dataflow. Initial operands
        # and snapshot copies have zero additional arithmetic depth.
        layers = {f'X{key}': 0 for key in program.keys}
        dependencies = {f'X{key}': {key} for key in program.keys}
        stages = []
        for statement in program.statements:
            target, expression = statement.split(' = ')
            names = re.findall(r'[A-Z][0-9]*', expression)
            assert names and set(names) <= layers.keys()
            layer = max(layers[name] for name in names)
            inputs = set().union(*(dependencies[name] for name in names))
            initialization = target in ('B', 'C', 'D', 'V') and target not in layers
            snapshot = re.fullmatch(r'[A-Z][0-9]*', expression) is not None
            if not initialization and not snapshot:
                layer += 1
                stages.append(target)
            layers[target], dependencies[target] = layer, inputs
        assert len(stages) == layers['A'] == depth
        assert stages == [('C', 'D')[stage % 2] for stage in range(depth - 1)] + ['A']
        assert dependencies['A'] == set(program.keys)
        assert dependencies['A'] | dependencies['C'] | dependencies['D'] == set(program.keys)
        if depth > 2:
            assert 'D = D - C*V + D0' in program.statements

        operands = [3, 1, 2, 9]
        answers = equations.answer_program(program, predicted(program, operands))
        assert all(answer is not None for answer in answers)
        for answer in answers:
            exact = arithmetic.parse_answer(answer)
            assert exact is not None and arithmetic.format_answer(exact) == answer
            if depth == 24:
                assert max(abs(part) for part in exact) > 2**53
        for slot in range(4):
            changed = operands.copy()
            changed[slot] += 1
            other = equations.answer_program(program, predicted(program, changed))
            assert all(a != b for a, b in zip(answers, other, strict=True))


def test_depth_one_counts_only_the_final_assignment(equations):
    program, = equations.make_programs(11, 0, [16, 17, 18, 19], 1)
    assert len(program.statements) == 5
    assert equations.answer_program(program, dict.fromkeys(program.keys, 32)) == ('2+i', '2+i')


@pytest.mark.parametrize('keys', [
    (), (16, 17, 18), (16, 17, 18, 18), (15, 17, 18, 19),
    (16, 17, 18, 32), (16.0, 17, 18, 19), (True, 17, 18, 19),
])
def test_reject_incomplete_duplicate_or_invalid_key_groups(equations, keys):
    with pytest.raises(ValueError, match='keys'):
        equations.make_programs(11, 0, keys, 2)


@pytest.mark.parametrize('changes', [
    {'seed': True}, {'seed': 1.0}, {'family': -1}, {'family': False},
    {'depth': 0}, {'depth': -1}, {'depth': True}, {'depth': 2.0},
])
def test_reject_invalid_program_parameters(equations, changes):
    args = dict(seed=11, family=0, keys=[16, 17, 18, 19], depth=2)
    args.update(changes)
    with pytest.raises(ValueError):
        equations.make_programs(**args)


def test_malformed_program_is_not_misreported_as_missing_retrieval(equations):
    program, = equations.make_programs(11, 0, [16, 17, 18, 19], 2)
    malformed = equations.Program(program.keys, program.statements + ('A = Z',))
    with pytest.raises(ValueError, match='undefined'):
        equations.answer_program(malformed, predicted(program, (3, 1, 2, 9)))
