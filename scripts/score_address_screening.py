"""Score frozen §22.17 reads without JAX or a checkpoint; never infer missing responses."""
import argparse
from collections import Counter,defaultdict
import csv
from fractions import Fraction
from pathlib import Path

from score_binding_state import _answers, _csv, _programs, key_hits

SEEDS=(42,43,44)
REF=('ordinary','replay')
MEM=('coupled','isolated','register','literal')
CASES=(('near',0,0),('overwrite',0,32),('mixed',0,0),('mixed',0,1024),('change',0,1024),('filler',0,1024),
       *(('version',d,g) for d in (2,24) for g in (0,1024)))


def load(directory):
    data={}; inputs={}; weights={}
    for filename,arms in (('references.csv',REF),('routes.csv',MEM)):
        with (directory/filename).open(encoding='utf8',newline='') as f:
            for r in csv.DictReader(f):
                s,arm,c,d,g,draw,fam=int(r['seed']),r['arm'],r['case'],int(r['depth']),int(r['gap']),int(r['data_seed']),int(r['family'])
                k=s,arm,c,d,g,fam
                if (s not in SEEDS or arm not in arms or (c,d,g) not in CASES or k in data or not 0<=fam<32 or
                    (c=='version' and arm in REF) or draw!=({'version':732013,'overwrite':732008}.get(c,732007))): raise ValueError(f'unregistered or duplicate observation {k}')
                if weights.setdefault(s,r['checkpoint_sha256'])!=r['checkpoint_sha256']: raise ValueError(f'changed frozen checkpoint: {s}')
                def vec(field): return tuple(map(int,r[field].split()))
                keys,focus=vec('keys'),vec('focus'); gold=(vec('truth0'),vec('truth1')); pred=(vec('pred0'),vec('pred1'))
                n=2 if c=='overwrite' else 8
                if (len(set(keys))!=n or any(len(v)!=n for v in (keys,focus,*gold,*pred)) or any(not 16<=i<32 for i in keys)
                    or any(not 32<=i<48 for w in gold for i in w) or any(not -2<=i<128 for w in pred for i in w)
                    or any(i not in (0,1) for i in focus) or sum(focus)!=(4 if c in ('version','mixed') else 0)): raise ValueError(f'invalid key, mask or response {k}')
                signature=draw,keys,focus,gold,r['input_sha256']; case=c,d,g,fam
                if inputs.setdefault(case,signature)!=signature: raise ValueError(f'unequal inputs across weights or arms: {case}')
                data[k]=(keys,focus,gold,pred,draw)
    for c,d,g in CASES:
        for s in SEEDS:
            for a in (MEM if c=='version' else REF+MEM):
                for f in range(32):
                    if (s,a,c,d,g,f) not in data: raise ValueError(f'missing {s}/{a}/{c}/{d}/{g}/{f}')
    return data


def scores(data):
    recall=defaultdict(Counter);eq=defaultdict(Counter)
    for (s,a,c,d,g,f),(keys,focus,gold,pred,draw) in data.items():
        close=None
        if c!='overwrite':
            nc=c if c in ('version','mixed') else 'near'; nk,nf,ng,close,_=data[s,a,nc,d,0,f]
            if (nk,nf,ng)!=(keys,focus,gold): raise ValueError('different operands at the matched near distance')
        hits=key_hits(pred,gold); both=key_hits(close,gold) if close is not None else [False]*len(keys)
        sets={'all':tuple(range(len(keys)))}
        if c in ('version','mixed'):
            sets['copied' if c=='version' else 'rewritten']=tuple(i for i,b in enumerate(focus) if b)
            sets['poison' if c=='version' else 'old']=tuple(i for i,b in enumerate(focus) if not b)
        for sub,idx in sets.items():
            got=[pred[w][i] for i in idx for w in range(2)]
            recall[s,a,c,d,g,sub].update(key_pairs=len(idx),both_worlds=sum(hits[i] for i in idx),near_and_far=sum(hits[i] and both[i] for i in idx),
                                         paired_key_pairs=len(idx) if close is not None else 0,nonfinite=got.count(-1),unbound=got.count(-2))
        if c not in ('mixed','version'): continue
        operands=tuple(k for k,flag in zip(keys,focus,strict=True) if flag) if c=='version' else keys
        for program in _programs(draw,f,operands,24):
            expected,result,near=(_answers(program,keys,x) for x in (gold,pred,close))
            eq[s,a,c,d,g].update(program_pairs=1,both_outputs_both_worlds=int(result==expected),near_and_far=int(near==expected and result==expected),
                a_both_worlds=int(all(p[0]==t[0] for p,t in zip(result,expected,strict=True))),acd_both_worlds=int(all(p[1]==t[1] for p,t in zip(result,expected,strict=True))),
                invalid_worlds=sum(None in world for world in result))
    return recall,eq


def gates(recall,eq):
    checks={}
    for s in SEEDS:
        def rate(arm,c,d,g,sub):
            r=recall[s,arm,c,d,g,sub];return Fraction(r['both_worlds'],r['key_pairs'])
        acc=[rate('isolated',*case) for case in (('mixed',0,1024,'rewritten'),('mixed',0,1024,'old'),('version',24,1024,'copied'))]
        protected=[rate('isolated',c,0,g,'all')-rate('ordinary',c,0,g,'all') for c,g in (('near',0),('overwrite',32))]
        nonfinite=sum(x['nonfinite'] for (ws,a,_,__,___,sub),x in recall.items() if (ws,a,sub)==(s,'isolated','all'))
        checks[s]=(*acc,*protected,nonfinite,min(acc)>=Fraction(9,10) and min(protected)>=-Fraction(2,100) and nonfinite==0)
    def gain(table,key,field,total):
        isolated=[table[(s,'isolated',*key)] for s in SEEDS]; coupled=[table[(s,'coupled',*key)] for s in SEEDS]
        return Fraction(sum(a[field]-b[field] for a,b in zip(isolated,coupled,strict=True)),sum(a[total] for a in isolated))
    old_gain=gain(recall,('mixed',0,1024,'old'),'both_worlds','key_pairs'); eq_gain=gain(eq,('version',24,1024),'both_outputs_both_worlds','program_pairs')
    adopt=all(x[-1] for x in checks.values()) and min(old_gain,eq_gain)>=Fraction(1,10)
    for s in SEEDS: yield (s,*map(float,checks[s][:5]),*checks[s][5:],float(old_gain),float(eq_gain),adopt)


def pooled_scores(data):
    """Keep all seeds, keys and both worlds together when resampling a history family."""
    import numpy as np

    by_family=defaultdict(dict)
    for key,value in data.items():
        by_family[key[-1]][key]=value
    pooled_r,pooled_e=defaultdict(Counter),defaultdict(Counter)
    paired=defaultdict(list)
    for family in sorted(by_family):
        r,e=scores(by_family[family])
        for parts,total in ((r,pooled_r),(e,pooled_e)):
            for key,counts in parts.items():
                total[key].update(counts)
        for endpoint,table,key,correct,trials in (
            ('old',r,('mixed',0,1024,'old'),'both_worlds','key_pairs'),
            ('versioned_equation',e,('version',24,1024),'both_outputs_both_worlds','program_pairs'),
        ):
            a=[table[s,'isolated',*key] for s in SEEDS]
            b=[table[s,'coupled',*key] for s in SEEDS]
            paired[endpoint].append((sum(x[correct]-y[correct] for x,y in zip(a,b,strict=True)),sum(x[trials] for x in a)))
    # Same cluster bootstrap convention as §22.16. These intervals do not alter the gate.
    n=len(by_family)
    draws=np.random.default_rng(424216).integers(n,size=(20_000,n))
    intervals=[]
    for endpoint,series in paired.items():
        gain,total=np.asarray(series,dtype=np.int32).T
        lo,hi=np.quantile(gain[draws].sum(1)/total[draws].sum(1),[0.025,0.975])
        intervals.append((endpoint,int(gain.sum()),int(total.sum()),gain.sum()/total.sum(),lo,hi))
    return pooled_r,pooled_e,intervals


def report(directory,output):
    r,e,intervals=pooled_scores(load(directory));decisions=list(gates(r,e));output.mkdir(parents=True,exist_ok=True)
    _csv(output/'retrieval.csv',('seed','arm','case','depth','gap','subset'),('key_pairs','both_worlds','near_and_far','paired_key_pairs','nonfinite','unbound'),r)
    _csv(output/'equations.csv',('seed','arm','case','depth','gap'),('program_pairs','both_outputs_both_worlds','a_both_worlds','acd_both_worlds','near_and_far','invalid_worlds'),e)
    with (output/'gates.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(('seed','rewritten_rate','old_rate','copied_rate','near_delta','overwrite_delta','nonfinite','per_seed_pass','pooled_old_gain','pooled_equation_gain','advance'));w.writerows(decisions)
    with (output/'intervals.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(('endpoint','gain_pairs','pairs','rate_difference','family_ci95_low','family_ci95_high'));w.writerows(intervals)
    print('isolated vs coupled, pooled old gain / versioned equation gain / gate:',decisions[0][-3:])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();report(a.directory,a.output)
