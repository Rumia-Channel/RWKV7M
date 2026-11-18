"""Generate, run or score the paired symbolic-bindings benchmark. No model sees the private key."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from symbolic_arithmetic import parse_answer, solve_prompt, format_answer
from symbolic_bindings import MODES, QUERIES, cases


def records(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def emit(stream, obj):
    stream.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')


def make(args):
    args.directory.mkdir(parents=True, exist_ok=False)
    count = 0
    with (args.directory/'questions.jsonl').open('x', encoding='utf-8', newline='\n') as questions, \
         (args.directory/'key.jsonl').open('x', encoding='utf-8', newline='\n') as key:
        for public, private in cases(args.seed, args.families, args.depths, args.gaps, args.modes):
            emit(questions, public); emit(key, private); count += 1
    print(f'{count} questions; pass only questions.jsonl to the model.')


def predict(args):
    if args.command == 'rwkv':
        from symbolic_rwkv_readout import make_reader
        reader = make_reader(args.checkpoint, vocab_file=args.vocab_file, max_new_tokens=args.max_new_tokens)
    else:
        if args.last_statements is not None and args.last_statements < 1:
            raise ValueError('last-statements must be positive')
        def reader(prompt):
            try:
                return {'prediction': format_answer(solve_prompt(prompt, window=args.last_statements))}
            except ValueError:
                if args.last_statements is None: raise
                return {'prediction': '', 'stop': 'unresolved_in_window'}
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        for row in records(args.questions):
            emit(out, {'id': row['id'], **reader(row['prompt'])})


def score(key, predictions):
    """Missing or unparsable answers are misses, including in paired and joint denominators."""
    preds = {}
    for row in records(predictions):
        uid = row['id']
        if uid in preds: raise ValueError(f'duplicate prediction: {uid}')
        preds[uid] = parse_answer(row['prediction']) if isinstance(row['prediction'], str) else None
    # seed/mode/depth/query/family/gap/world -> (correct, missing, invalid), independent of file order.
    families = defaultdict(dict)
    seen = set(); absent = object(); gaps = set()
    for r in records(key):
        uid = r['id']; cohort = (r['seed'], r['mode'], r['depth'], r['query'], r['family']); cell = (r['gap'], r['world'])
        if uid in seen or cell in families[cohort]: raise ValueError(f'duplicate key observation: {uid}')
        if r['mode'] not in MODES or r['query'] not in QUERIES or cell[1] not in (0,1) or cell[0] < 0:
            raise ValueError(f'invalid key observation: {uid}')
        seen.add(uid); gaps.add(cell[0]); gold = parse_answer(r['answer'])
        if gold is None: raise ValueError(f'invalid reference answer: {uid}')
        answer = preds.pop(uid, absent)
        families[cohort][cell] = (answer == gold, answer is absent, answer is None, gold)
    if preds: raise ValueError(f'unknown prediction ID: {next(iter(preds))}')
    if 0 not in gaps: raise ValueError('key needs a matched gap-zero reference')
    buckets = defaultdict(lambda: dict(answers=0, correct=0, missing=0, invalid=0,
                                       families=0, both_worlds=0, near_and_far_both_worlds=0))
    expected = {(g, w) for g in gaps for w in (0, 1)}
    for (seed, mode, depth, query, _), series in families.items():
        if series.keys() != expected: raise ValueError('incomplete paired reference family')
        if series[0,0][3] == series[0,1][3]: raise ValueError('counterfactual answers did not change')
        near = series[0,0][0] and series[0,1][0]
        for gap in gaps:
            result = buckets[seed, mode, depth, query, gap]; pair = [series[gap,w] for w in (0,1)]
            if any(x[3] != series[0,w][3] for w,x in enumerate(pair)): raise ValueError('gap changed the underlying problem')
            for correct, missing, invalid, _ in pair:
                result['answers'] += 1; result['correct'] += int(correct)
                result['missing'] += int(missing); result['invalid'] += int(invalid)
            both = all(x[0] for x in pair); result['families'] += 1
            result['both_worlds'] += int(both); result['near_and_far_both_worlds'] += int(near and both)
    cells = [dict(seed=s, mode=m, depth=d, query=q, gap=g, **buckets[s,m,d,q,g]) for s,m,d,q,g in sorted(buckets)]
    return dict(answers=len(seen), correct=sum(c['correct'] for c in cells), missing=sum(c['missing'] for c in cells),
                invalid=sum(c['invalid'] for c in cells), cells=cells)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('make', help='write public prompts and a separate private key in a new directory')
    gen.add_argument('directory', type=Path); gen.add_argument('--seed', type=int, default=42101)
    gen.add_argument('--families', type=int, default=8); gen.add_argument('--depths', type=int, nargs='+', default=[2, 8, 24])
    gen.add_argument('--gaps', type=int, nargs='+', default=[0, 32]); gen.add_argument('--modes', choices=MODES, nargs='+', default=MODES)
    ref = sub.add_parser('baseline', help='blind symbolic interpreter, optionally restricted to a suffix of the prompt')
    ref.add_argument('--last-statements', type=int, help='ignore all but this many trailing assignments')
    model = sub.add_parser('rwkv', help='greedy answer generation from a tokenizer-compatible text checkpoint')
    model.add_argument('--checkpoint', type=Path, required=True); model.add_argument('--vocab-file', type=Path)
    model.add_argument('--max-new-tokens', type=int, default=64)
    for cmd in (ref, model):
        cmd.add_argument('questions', type=Path); cmd.add_argument('output', type=Path)
    report = sub.add_parser('score', help='grade JSONL {id,prediction} rows from any runtime')
    report.add_argument('key', type=Path); report.add_argument('predictions', type=Path)
    args = p.parse_args()
    if args.command == 'make': make(args)
    elif args.command == 'score': print(json.dumps(score(args.key, args.predictions), indent=2))
    else: predict(args)


if __name__ == '__main__':
    main()
