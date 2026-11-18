"""CPU-only rescore of the registered iterative-reader panel; does not load a model."""
import argparse
import csv
from pathlib import Path

import numpy as np

from iterative_read_tasks import SPLITS, TEST_RULES, route_labels

DRAWS, FAMILIES, GAPS = SPLITS['test']
SEEDS = (42,43,44)
ARMS = ('isolated','reset','ordinary')
NBOOT = 10_000


def summarize(cache,panel):
    dims = (len(ARMS),len(TEST_RULES),len(GAPS),len(DRAWS),FAMILIES,len(SEEDS))
    hits = np.full(dims,-1,np.int8)
    with (panel/'readouts.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['arm'] not in ARMS: continue
            i = (ARMS.index(row['arm']),tuple(TEST_RULES).index(row['task']),GAPS.index(int(row['gap'])),
                 DRAWS.index(int(row['draw'])),int(row['family']),SEEDS.index(int(row['weight_seed'])))
            arr = np.array([list(map(int,row[name].split())) for name in ('pred0','answer0','pred1','answer1')])
            if arr.shape!=(4,7) or hits[i] != -1: raise ValueError('unexpected or duplicate family predictions')
            hits[i] = np.sum((arr[0]==arr[1]) & (arr[2]==arr[3]))
    if np.any(hits<0): raise ValueError('incomplete registered panel')
    # Keep worlds, questions, and repeats with the same weights paired. Resample
    # graph draw, then families within each selected draw; weights are fixed.
    rng = np.random.default_rng(744019)
    d = rng.integers(len(DRAWS),size=(NBOOT,len(DRAWS),1))
    f = rng.integers(FAMILIES,size=(NBOOT,len(DRAWS),FAMILIES))
    with (panel/'paired-deltas.csv').open('x',encoding='utf8',newline='') as stream:
        wr=csv.writer(stream,lineterminator='\n'); wr.writerow(('task','gap','control','draws','families','fixed_weight_seeds','paired_keys','isolated_hits','control_hits','delta_pp','lower95_pp','upper95_pp'))
        for task,rule in enumerate(TEST_RULES):
            for g,gap in enumerate(GAPS):
                a=hits[0,task,g]
                for arm in ('reset','ordinary'):
                    b=hits[ARMS.index(arm),task,g]; diff=(a-b).mean(-1)*(100/7)
                    lower,upper=np.quantile(diff[d,f].mean(axis=(1,2)),[.025,.975])
                    wr.writerow((rule,gap,arm,len(DRAWS),len(DRAWS)*FAMILIES,len(SEEDS),int(a.size*7),int(a.sum()),int(b.sum()),float(diff.mean()),float(lower),float(upper)))
    with (panel/'direct-read.csv').open('x',encoding='utf8',newline='') as stream:
        wr=csv.writer(stream,lineterminator='\n'); wr.writerow(('weight_seed','weight_sha','frontend','task','gap','primitive_correct','primitive_total','direct_both_worlds','direct_pairs'))
        for seed in SEEDS:
            with np.load(cache/f'test-s{seed}.npz',allow_pickle=False) as z:
                for frontend in ('ordinary','isolated'):
                    edges,truth,q = z[frontend],z['truth'],z['query']
                    pred = np.take_along_axis(edges,q,axis=-1)  # Never access labels for the direct read.
                    for rule,pair in TEST_RULES.items():
                        paired=(pred==route_labels(truth,q,pair)).all(1)
                        for gap in GAPS:
                            take = z['gap']==gap
                            wr.writerow((seed,z['weight_sha'].item(),frontend,rule,gap,int((edges[take]==truth[take]).sum()),truth[take].size,int(paired[take].sum()),paired[take].size))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('cache',type=Path); ap.add_argument('panel',type=Path)
    args=ap.parse_args(); summarize(args.cache,args.panel)
