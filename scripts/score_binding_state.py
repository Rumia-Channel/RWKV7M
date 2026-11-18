"""Analyze per-world frozen readouts; no JAX or checkpoint required to reproduce the scores."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from pathlib import Path

from binding_equations import answer_program, make_programs


ARMS = ('live', 'fifo', 'keyed')
GUARD_ARMS = (*ARMS, 'guarded')
MIXED = {('mixed',0), ('mixed',1024)}
DIRECT = {('near',0), ('repeat',128), ('change',128), ('repeat',1024), ('change',1024), ('filler',1024)}
CONDITIONS = DIRECT | {('overwrite',32)}
DEPTHS = (2, 8, 24)


def observations(path):
    data = {}; hashes = {}; families = {}
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        guarded = 'guarded0' in (reader.fieldnames or ())
        arms = GUARD_ARMS if guarded else ARMS
        expected = CONDITIONS | MIXED if guarded else CONDITIONS
        for line in reader:
            seed, family, count, gap, draw = (int(line[k]) for k in ('weight_seed','family','families','gap','data_seed'))
            task = line['task']; condition = task, gap; key = seed, family, task, gap
            if condition not in expected or key in data or not 0 <= family < count or families.setdefault(seed,count) != count:
                raise ValueError(f'unexpected, duplicate or inconsistent observation {key}')
            signature = (draw, line['input_sha256'], line['keys'], line['truth0'], line['truth1'], line.get('recent'))
            if hashes.setdefault((family,task,gap), signature) != signature:
                raise ValueError(f'weight seeds did not receive identical inputs: {key}')
            def integers(field): return tuple(map(int,line[field].split()))
            keys = integers('keys'); n = 2 if task == 'overwrite' else 8
            if len(set(keys)) != n or len(keys) != n or not all(16 <= k < 32 for k in keys):
                raise ValueError(f'invalid query keys for {key}')
            values = {arm: tuple(integers(f'{arm}{w}') for w in (0,1)) for arm in ('truth', *arms)}
            if any(len(world) != n for pair in values.values() for world in pair): raise ValueError(f'incomplete answers for {key}')
            if any(not 32 <= y < 48 for world in values['truth'] for y in world): raise ValueError(f'invalid truth for {key}')
            if guarded:
                recent = integers('recent')
                if len(recent) != n or any(flag not in (0,1) for flag in recent):
                    raise ValueError(f'invalid visible recency mask for {key}')
                values['recent'] = recent
            data[key] = (draw, keys, values)
    if not families or len(data) != len(expected)*sum(families.values()): raise ValueError('missing paired panel observations')
    return data


def key_hits(pred, truth):
    return tuple(p0 == g0 and p1 == g1 for p0,p1,g0,g1 in zip(*pred,*truth,strict=True))


@lru_cache(maxsize=512)
def _programs(draw, family, keys, depth):
    return tuple(make_programs(draw, family, keys, depth))


@lru_cache(maxsize=2048)
def _answers(program, keys, worlds):
    return tuple(answer_program(program, dict(zip(keys, world, strict=True))) for world in worlds)


def summarize(data):
    recall = defaultdict(Counter); arithmetic = defaultdict(Counter)
    for (seed,family,task,gap), (draw,keys,values) in data.items():
        arms = GUARD_ARMS if 'guarded' in values else ARMS
        direct = (task,gap) in DIRECT or (task,gap) in MIXED
        if direct:
            nd, nk, near = data[seed,family,'mixed' if task=='mixed' else 'near',0]
            if (draw, keys, values['truth']) != (nd, nk, near['truth']): raise ValueError('far case changed operands or questions')
        for arm in arms:
            pred = values[arm]; truth = values['truth']; hits = key_hits(pred,truth)
            r = recall[seed,task,gap,arm]
            r.update(families=1, key_pairs=len(keys), both_worlds=sum(hits), questions=2*len(keys),
                     correct=sum(p==y for pp,yy in zip(pred,truth,strict=True) for p,y in zip(pp,yy,strict=True)),
                     nonfinite=sum(p==-1 for pp in pred for p in pp))
            if direct:
                r['near_and_far'] += sum(a and b for a,b in zip(hits,key_hits(near[arm],truth),strict=True))
        if not direct: continue  # The overwrite control has two operands, not four.
        for depth in DEPTHS:
            for program in _programs(draw, family, keys, depth):
                gold = _answers(program, keys, values['truth'])
                for arm in arms:
                    a = arithmetic[seed,task,gap,depth,arm]
                    got = _answers(program, keys, values[arm]); close = got if task=='near' else _answers(program, keys, near[arm])
                    paired = got == gold
                    a.update(program_pairs=1, a_both_worlds=int(all(p[0]==g[0] for p,g in zip(got,gold,strict=True))),
                             acd_both_worlds=int(all(p[1]==g[1] for p,g in zip(got,gold,strict=True))),
                             both_outputs_both_worlds=int(paired), near_and_far=int(paired and close==gold),
                             both_targets_changed=int(all(left!=right for left,right in zip(*gold,strict=True))),
                             invalid_worlds=sum(None in w for w in got))
    return recall, arithmetic


def mixed_age_scores(data):
    """Stratify by the key's visible age in this history, not the correctness of either arm."""
    counts = defaultdict(Counter)
    for (seed,family,task,gap), (_,keys,v) in data.items():
        if task != 'mixed':
            continue
        _, _, near = data[seed,family,'mixed',0]
        for arm in GUARD_ARMS:
            hit = key_hits(v[arm],v['truth']); close = key_hits(near[arm],v['truth'])
            for k in range(len(keys)):
                c = counts[seed,gap,'recent' if v['recent'][k] else 'old',arm]
                c.update(key_pairs=1, both_worlds=int(hit[k]), near_and_far=int(hit[k] and close[k]),
                         different_from_live=sum(v[arm][w][k] != v['live'][w][k] for w in (0,1)))
    return counts


def _csv(path, key_columns, metrics, counts):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n'); writer.writerow((*key_columns,*metrics))
        for key, row in sorted(counts.items()):
            writer.writerow((*key, *(row.get(field, '') for field in metrics)))


def report(path, directory):
    data = observations(path)
    recall, arithmetic = summarize(data)
    mixed = mixed_age_scores(data)
    arms = GUARD_ARMS if mixed else ARMS
    candidate = 'guarded' if mixed else 'keyed'
    directory.mkdir(parents=True, exist_ok=False)
    _csv(directory/'retrieval.csv', ('weight_seed','task','gap','arm'),
         ('families','questions','correct','nonfinite','key_pairs','both_worlds','near_and_far'), recall)
    _csv(directory/'equations.csv', ('weight_seed','task','gap','depth','arm'),
         ('program_pairs','both_targets_changed','a_both_worlds','acd_both_worlds','both_outputs_both_worlds','near_and_far','invalid_worlds'), arithmetic)
    if mixed:
        _csv(directory/'mixed-age.csv', ('weight_seed','gap','recency','arm'),
             ('key_pairs','both_worlds','near_and_far','different_from_live'), mixed)
    def rate(seed, task, gap, arm, field):
        r = recall[seed,task,gap,arm]; return r[field]/r['key_pairs']
    passed = True
    for seed in sorted({k[0] for k in recall}):
        joint = {arm: rate(seed,'change',1024,arm,'near_and_far') for arm in arms}
        gaps = [rate(seed,t,d,candidate,'both_worlds')-rate(seed,t,d,'live','both_worlds') for t,d in (('near',0),('overwrite',32))]
        nonfinite = sum(r['nonfinite'] for (s,_,__,arm),r in recall.items() if s==seed and arm==candidate)
        ok = min(joint[candidate]-joint[arm] for arm in ('live','fifo')) >= .10 and min(gaps)>=-.02 and nonfinite==0
        if mixed:
            old = mixed[seed,1024,'old',candidate]; reference = mixed[seed,1024,'old','live']
            recent = mixed[seed,1024,'recent',candidate]
            gain = (old['both_worlds']-reference['both_worlds']) / old['key_pairs'] if old['key_pairs'] else 0
            ok &= gain >= .10 and recent['key_pairs'] > 0 and recent['different_from_live']==0
            print(f'parent {seed}: mixed old-key gain {gain:.4f}, recent {recent["both_worlds"]}/{recent["key_pairs"]}, changed recent predictions {recent["different_from_live"]}')
        print(f'parent {seed}: near and changing-1024 {joint}; near/overwrite delta {gaps}; {candidate} nonfinite {nonfinite}; gate {ok}')
        passed &= ok
    print(f'All fixed model gates passed: {passed}')


def _literal_read(history, keys, store_token):
    """Domain-specific non-neural comparator: 16 addresses with a visible byte value each."""
    table = bytearray(16)
    for op, key, value in zip(history, history[1:], history[2:]):
        if op == store_token and 16 <= key < 32 and 32 <= value < 48:
            table[int(key)-16] = int(value)
    return tuple(table[k-16] for k in keys)


def literal_control(path, output):
    from benchmark_binding_state import panel
    from retrieval_frontier import STORE
    data = observations(path)
    seed = min(k[0] for k in data)
    near = data[seed,0,'near',0]
    pairs = sum(s==seed and t=='near' for s,_,t,_ in data)
    counts = defaultdict(Counter)
    cases = panel(pairs, near[0], data[seed,0,'overwrite',32][0], guard_study='guarded' in near[2])
    for task, gap, draw, histories, queries, _ in cases:
        for family, worlds in enumerate(histories):
            keys = tuple(map(int,queries[family,0,:,1]))
            read = tuple(_literal_read(h, keys, STORE) for h in worlds)
            original_draw, original_keys, outputs = data[seed,family,task,gap]
            if (original_draw,original_keys) != (draw,keys):
                raise ValueError('reference query mismatch in regenerated public panel')
            y = outputs['truth']
            counts[task,gap].update(key_pairs=len(keys), both_worlds=sum(key_hits(read,y)),
                                    questions=2*len(keys), correct=sum(p==t for w in (0,1) for p,t in zip(read[w],y[w],strict=True)))
    _csv(output, ('task','gap'), ('key_pairs','both_worlds','questions','correct'), counts)


def example(data, seed, family, depth):
    draw, keys, values = data[seed,family,'change',1024]
    p = _programs(draw,family,keys,depth)[0]
    print('Public ordered program (X<key> = value token ID minus 31):'); print('\n'.join(p.statements))
    print('world,readout,retrieved IDs in public operand order,A,ACD')
    for w in (0,1):
        for arm in ('truth',*(GUARD_ARMS if 'guarded' in values else ARMS)):
            vals = dict(zip(keys,values[arm][w],strict=True)); a,acd = answer_program(p,vals)
            print(f'{w},{arm},{" ".join(str(vals[k]) for k in p.keys)},{a},{acd}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    run = sub.add_parser('report'); run.add_argument('readouts',type=Path); run.add_argument('directory',type=Path)
    table = sub.add_parser('table', help='score a literal 16-byte key/value table on the public protocol, once per family')
    table.add_argument('readouts',type=Path); table.add_argument('output',type=Path)
    show = sub.add_parser('example'); show.add_argument('readouts',type=Path); show.add_argument('--seed',type=int,default=42)
    show.add_argument('--family',type=int,default=0); show.add_argument('--depth',type=int,choices=DEPTHS,default=2)
    args = p.parse_args()
    if args.action=='report': report(args.readouts,args.directory)
    elif args.action=='table': literal_control(args.readouts,args.output)
    else: example(observations(args.readouts),args.seed,args.family,args.depth)
