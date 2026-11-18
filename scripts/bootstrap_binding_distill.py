"""Paired uncertainty for the rewritten-key endpoint, conditional on the three published parents."""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from probe_binding_distill import FAMILIES, WEIGHT_SEEDS, observations
from score_binding_state import key_hits


def intervals(observed):
    # Each draw retains both worlds, all four keys and every arm/seed on that family.
    draws = np.random.default_rng(424216).integers(FAMILIES, size=(20_000,FAMILIES))
    for candidate,reference in (('ce','parent'),('distill','ce')):
        d = np.empty((len(WEIGHT_SEEDS),FAMILIES),np.int16)
        for i,seed in enumerate(WEIGHT_SEEDS):
            for family in range(FAMILIES):
                left,right=(observed[a,seed,family,'mixed',1024] for a in (candidate,reference))
                recent=left['flags'][0]
                if sum(recent)!=4 or recent!=right['flags'][0]: raise ValueError('unmatched rewritten questions')
                d[i,family]=sum(int(x)-int(y) for x,y,k in zip(key_hits(left['pred'],left['truth']),key_hits(right['pred'],right['truth']),recent,strict=True) if k)
        family_delta=d.sum(axis=0); by_seed=d.sum(axis=1); reps=family_delta[draws].mean(axis=1)/(4*len(by_seed))
        lo,hi=np.quantile(reps,[.025,.975]); obs=abs(by_seed.sum())
        signs=np.asarray(tuple(product((-1,1), repeat=len(by_seed))))
        p=(np.abs(signs@by_seed)>=obs).mean()
        yield candidate,reference,int(d.sum()),d.size*4,float(d.mean()/4),float(by_seed.std(ddof=1)/(FAMILIES*4)),float(lo),float(hi),float(p)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('readouts',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    rows=list(intervals(observations(a.readouts)))
    with a.output.open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(('candidate','reference','gain_pairs','key_pairs','rate_difference','across_parent_sd','family_ci95_low','family_ci95_high','two_sided_parent_sign_flip_p'));wr.writerows(rows)
    for row in rows: print(f'{row[0]} vs {row[1]}: {row[4]:+.4f}, paired family 95% interval [{row[6]:+.4f}, {row[7]:+.4f}], parent sign-flip p={row[8]:.3f}')
