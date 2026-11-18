"""Matched mixed-minus-basic-only changes on exactly the same sealed midterm papers."""
import argparse
import csv
from pathlib import Path

import numpy as np

from role_binding_exam import MID_DRAWS


def outcomes(directory,curriculum):
    rows={}
    with (directory/'families.csv').open(encoding='utf8',newline='') as file:
        for r in csv.DictReader(file):
            if r['form']!='midterm' or r['arm'] not in ('marked-count','marked-rwkv'): continue
            if r['curriculum']!=curriculum or r['worlds']!='4' or r['reads']!='2': raise ValueError('unexpected curriculum or test form')
            key=(r['arm'],r['surface'],int(r['gap']),int(r['seed']),int(r['draw']),int(r['family']),int(r['question']))
            if key in rows: raise ValueError('duplicate paired observation')
            rows[key]=r
    return rows


def compare(mixed,basic,dest):
    a=outcomes(mixed,'mixed'); b=outcomes(basic,'basic-only')
    if a.keys()!=b.keys() or len(a)!=2*2*2*3*5*8*2: raise ValueError('incomplete matched midterm data')
    if any(a[k]['input_sha']!=b[k]['input_sha'] for k in a): raise ValueError('different public histories or cards')
    rng=np.random.default_rng(766919); dr=rng.integers(5,size=(10000,5,1)); fa=rng.integers(8,size=(10000,5,8))
    with dest.open('x',encoding='utf8',newline='') as file:
        w=csv.writer(file,lineterminator='\n'); w.writerow(('arm','surface','gap','metric','mixed','basic_only','denominator','delta_pp','lower95_pp','upper95_pp','draws','families_per_draw','resamples'))
        for condition in sorted({key[:3] for key in a}):
            for metric in ('answer_success','joint_success'):
                keys=[condition+(s,d,f,q) for s in (42,43,44) for d in MID_DRAWS for f in range(8) for q in (0,1)]
                x,y=(np.asarray([int(rows[k][metric]) for k in keys]) for rows in (a,b))
                delta=(x-y).reshape(3,5,8,2).mean(axis=(0,3))*100.
                lo,hi=np.quantile(delta[dr,fa].mean(axis=(1,2)),[.025,.975])
                w.writerow((*condition,metric,int(x.sum()),int(y.sum()),len(keys),float(delta.mean()),float(lo),float(hi),5,8,10000))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('mixed','basic','output'): ap.add_argument(key,type=Path)
    args=ap.parse_args(); compare(args.mixed,args.basic,args.output)
