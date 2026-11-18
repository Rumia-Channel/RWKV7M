from collections import Counter
from importlib import import_module
import json
from pathlib import Path
import re

import pytest


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return import_module('benchmark_symbolic_bindings')


def program(row):
    return row['prompt'].split('Statements:\n', 1)[1].split('\nQuery:', 1)[0].splitlines()


def test_generator_has_real_dependency_depth_and_an_independent_executable_key(bench):
    args = 143, 2, [2, 9], [0, 7]
    rows = list(bench.cases(*args)); by = {}
    for public, key in rows:
        assert set(public) == {'id', 'prompt'}
        assert bench.format_answer(bench.solve_prompt(public['prompt'])) == key['answer']
        label = tuple(key[k] for k in ('family','depth','query','world','mode','gap'))
        by[label] = public, key['answer']
        if key['mode'] == 'updates' and key['query'] == 'A' and key['gap'] == 0:
            # Infer the versioned data-flow independently, rather than trusting the generator's `depth` label.
            deps = {}
            for line in program(public):
                target, rhs = line.split(' = '); inputs = re.findall(r'[A-Z][0-9]*', rhs)
                deps[target] = max((deps[x] for x in inputs), default=-1) + 1
            assert deps['A'] == key['depth']
    for (fam,d,q,w,mode,g), (pub, answer) in by.items():
        near, _ = by[fam,d,q,w,mode,0]; peer, other = by[fam,d,q,1-w,mode,g]
        assert program(pub)[:len(program(near))] == program(near) and len(program(pub)) - len(program(near)) == g
        assert answer != other
        assert Counter(''.join(program(pub))) == Counter(''.join(program(peer)))
        assert sum(x != y for x,y in zip(program(pub),program(peer),strict=True)) == 2
        assert answer == by[fam,d,q,w,'definitions' if mode=='updates' else 'updates',g][1]


def as_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return path


def test_near_far_score_requires_all_four_answers_and_counts_missing(bench, tmp_path):
    keys = [k for _,k in bench.cases(91, 1, [2], [0,5], ['updates'])]
    answers = []
    for r in keys:
        if (r['query'],r['gap'],r['world']) == ('A',0,1): continue
        bad = (r['query'],r['gap'],r['world']) == ('ACD',5,0)
        answers.append(dict(id=r['id'],prediction='12 + 3' if bad else r['answer']))
    k = as_jsonl(tmp_path/'key', keys); p = as_jsonl(tmp_path/'pred', reversed(answers))
    report = bench.score(k,p)
    assert (report['answers'],report['correct'],report['missing'],report['invalid']) == (8,6,1,1)
    cell = {(x['query'],x['gap']):x for x in report['cells']}
    assert (cell['A',5]['both_worlds'],cell['A',5]['near_and_far_both_worlds']) == (1,0)
    assert (cell['ACD',5]['both_worlds'],cell['ACD',5]['near_and_far_both_worlds']) == (0,0)
    as_jsonl(p, answers + [answers[0]])
    with pytest.raises(ValueError, match='duplicate prediction'): bench.score(k,p)
    as_jsonl(p, [dict(id='from-another-suite',prediction='1')])
    with pytest.raises(ValueError, match='unknown prediction'): bench.score(k,p)
