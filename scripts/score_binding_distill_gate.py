"""Apply §22.16's predeclared gates to complete, full-history-only held-out results."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path

SEEDS = (42,43,44)
FAMILIES = 32
CASES = (('mixed',1024), ('mixed',0), ('near',0), ('overwrite',32), ('filler',1024), ('change',1024))


def gate(rows):
    totals = {}
    for row in rows:
        key = int(row['weight_seed']), row['arm'], row['task'], int(row['gap']), row['subset']
        seed, arm, task, gap, subset = key
        n, hits, bad = (int(row[k]) for k in ('key_pairs','both_worlds','nonfinite'))
        if (seed not in SEEDS or arm not in ('ce','distill','parent') or (task,gap) not in CASES or
                subset not in (('all','old','rewritten') if task == 'mixed' else ('all',))):
            raise ValueError(f'unregistered cell: {key}')
        expected = FAMILIES * (2 if task == 'overwrite' else 4 if subset != 'all' else 8)
        if key in totals or n != expected or not 0 <= hits <= n or not 0 <= bad <= 2*n:
            raise ValueError(f'duplicate or invalid cell: {key}, n={n}, hits={hits}, nonfinite={bad}')
        totals[key] = hits, n, bad
    expected_cells = {(s,a,t,g,k) for s in SEEDS for a in ('parent','ce','distill') for t,g in CASES
                      for k in (('all','old','rewritten') if t == 'mixed' else ('all',))}
    if totals.keys() != expected_cells: raise ValueError(f'missing registered cells: {sorted(expected_cells-totals.keys())}')
    for candidate, ref, gain, min_votes in (('ce','parent',Fraction(1,10),0),('distill','ce',Fraction(1,20),2)):
        records = []; dh = denom = votes = 0; protected = True
        for s in SEEDS:
            def change(t,g,k):
                a,n,_ = totals[s,candidate,t,g,k]; b,_,_ = totals[s,ref,t,g,k]; return a-b,n
            d,n = change('mixed',1024,'rewritten'); dh += d; denom += n; votes += d > 0
            saved = [Fraction(*change(t,g,k)) for t,g,k in (('mixed',1024,'old'),('near',0,'all'),('overwrite',32,'all'))]
            nonfinite = sum(totals[s,candidate,t,g,'all'][2] for t,g in CASES)
            safe = min(saved) >= -Fraction(1,50) and nonfinite == 0; protected &= safe
            records.append((s,d,n,*map(float,saved),nonfinite,safe))
        passed = Fraction(dh,denom) >= gain and votes >= min_votes and protected
        for r in records:
            yield (candidate,ref,*r,dh,denom,votes,passed)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('retrieval',type=Path); p.add_argument('output',type=Path)
    a = p.parse_args()
    with a.retrieval.open(encoding='utf8',newline='') as f: results = list(gate(csv.DictReader(f)))
    with a.output.open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n'); wr.writerow(('candidate','reference','weight_seed','rewritten_gain_pairs','rewritten_pairs','old_delta','near_delta','overwrite_delta','nonfinite','guardrails_pass','pooled_gain_pairs','pooled_pairs','positive_seeds','adopt')); wr.writerows(results)
    for r in results[::len(SEEDS)]: print(f'{r[0]} vs {r[1]}: {r[-4]:+d}/{r[-3]} rewritten pairs, {r[-2]} positive seeds; adopt = {r[-1]}')
