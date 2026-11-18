"""Sealed paired §22.20 assessment, or label-free prediction on raw eight-token windows."""
import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from benchmark_address_screening import checkpoints
from episodic_experiment import load_cell
from episodic_scene_data import TEST_CELLS, TEST_DRAWS
from rwkv7m.io.episodic_address import PRIMARY_ARMS, load_head
from rwkv7m.model.episodic_address import BITS, empty_state, hard_episode, normalize, project, read_queries, write_window

COUNT_FIELDS=('paired_answer','paired_source','paired_swapped_query','paired_oracle_window','paired_survived_write_address',
              'paired_queries','individual_queries','selected','correct_if_selected','no_write','wrong_address','overwritten','bad_decode',
              'fully_correct','value_changes_write_address','same_query_pairs','fact_writes','fact_opportunities','background_writes','background_opportunities')
RAW=('arm','weight_seed','weight_sha','head_sha','surface','live','gap_windows','draw','family','input_sha','question','support',
     'gold0','gold1','pred0','pred1','swapped0','swapped1','source0','source1','read_address','support_address0','support_address1',
     'failure0','failure1','oracle0','oracle1','own_slot_owner0','own_slot_owner1')
FAIL=('no_write','wrong_address','overwritten','bad_decode','fully_correct')


def digest(path):
    with path.open('rb') as f: return file_digest(f,'sha256').hexdigest()


def _last(w,active,r):
    # Scoring-only source tracker from observable runtime writes. Never goes back into the model.
    times=np.arange(w.shape[1],dtype=np.int32)
    return np.where(active[:,None,:] & (w[:,None,:]==r[:,:,None]),times,-1).max(-1)


def diagnose(w,enabled,r,pred,support,gold):
    own_address=np.take_along_axis(w,support,axis=1); stored=np.take_along_axis(enabled,support,axis=1)
    owner=_last(w,enabled,r); own_owner=_last(w,enabled,own_address)
    code=np.where(~stored,0,np.where(r!=own_address,1,np.where(owner!=support,2,np.where(pred!=gold,3,4))))
    return owner,own_owner,own_address,code


def _evaluate(d,arm,p,mean,scale,k,g):
    f=len(d['draw']); b=2*f; gold=d['answers'].reshape(b,2); support=np.repeat(d['support'],2,axis=0)
    if arm=='ordinary':
        x=normalize(d['ordinary'].reshape(b,2,-1),mean,scale)
        allpred=np.asarray(project(p,jnp.concatenate((x,x[:,::-1]),axis=1)).argmax(-1))
    else:
        table=normalize(d['features'],mean,scale); x=table[d['blocks'].reshape(b,k+g)]
        q=jnp.repeat(table[d['queries']],2,axis=0)
        out=hard_episode(p,x,jnp.ones((b,k+g),bool),jnp.concatenate((q,q[:,::-1]),axis=1))
        allpred=np.asarray(out.answers)
    pred,swapped=np.split(allpred,2,axis=1)
    def paired(flag): return flag.reshape(f,2,2).all(axis=1)
    hit=paired(pred==gold)
    counters={key:'' for key in COUNT_FIELDS}; counters.update(paired_answer=int(hit.sum()),paired_swapped_query=int(paired(swapped==gold).sum()),
                                                                       paired_queries=hit.size,individual_queries=gold.size)
    extra={}
    if arm!='ordinary':
        w,enabled,r=np.asarray(out.write_address),np.asarray(out.wrote),np.asarray(out.read_address)[:,:2]
        owner,own_owner,own_address,fail=diagnose(w,enabled,r,pred,support,gold)
        oracle=np.asarray(project(p,x[np.arange(b)[:,None],support])[...,2*BITS+1:].argmax(-1))
        located=owner==support
        counters.update(paired_source=int(paired(located).sum()),paired_oracle_window=int(paired(oracle==gold).sum()),
            paired_survived_write_address=int(paired(own_owner==support).sum()),selected=int(located.sum()),
            correct_if_selected=int((located & (pred==gold)).sum()),value_changes_write_address=int((own_address[::2]!=own_address[1::2]).sum()),
            same_query_pairs=hit.size,fact_writes=int(enabled[:,:k].sum()),fact_opportunities=b*k,
            background_writes=int(enabled[:,k:].sum()),background_opportunities=b*g)
        counters.update(dict(zip(FAIL,map(int,np.bincount(fail.ravel(),minlength=5)),strict=True)))
        extra.update(source=owner,support_address=own_address,failure=np.asarray(FAIL)[fail],oracle=oracle,own_slot_owner=own_owner)
        if not np.array_equal(r[::2],r[1::2]): raise RuntimeError('query routing changed with the hidden world')
    for i in range(f):
        for qi in range(2):
            row={'draw':int(d['draw'][i]),'family':int(d['family'][i]),'input_sha':d['input_sha'][i],
                 'question':qi,'support':int(d['support'][i,qi])}
            if arm!='ordinary': row['read_address']=int(r[2*i,qi])
            for label,array in dict(gold=gold,pred=pred,swapped=swapped,**extra).items():
                for world in (0,1): row[label+str(world)]=array[2*i+world,qi]
            yield row,None,None
    yield None,counters,hit.sum(1)


def _effects(directory,family_hits):
    rng=np.random.default_rng(754919); nd=len(TEST_DRAWS); draws=rng.integers(nd,size=(10000,nd,1)); fam=rng.integers(8,size=(10000,nd,8))
    with (directory/'paired-effects.csv').open('x',encoding='utf8',newline='') as fp:
        wr=csv.writer(fp,lineterminator='\n'); wr.writerow(('surface','live','gap_windows','treatment','reference','delta_pp','lower95_pp','upper95_pp','maps','weight_seeds'))
        for surf in ('seen','held','shift'):
            for k,g in TEST_CELLS:
                for a,c in (('source','endpoint'),('source','ordinary'),('endpoint','ordinary')):
                    diff=np.stack([family_hits[s,surf,k,g,a]-family_hits[s,surf,k,g,c] for s in (42,43,44)]).mean(0)*50.
                    lo,hi=np.quantile(diff[draws,fam].mean(axis=(1,2)),[.025,.975]); wr.writerow((surf,k,g,a,c,diff.mean(),lo,hi,nd*8,3))


def assess(cache,heads,output,*,factors_only=False):
    arms=('factors',) if factors_only else PRIMARY_ARMS
    output.mkdir(parents=True,exist_ok=True); summaries=[]; family_hits={}; identical={}
    with (output/'readouts.csv').open('x',encoding='utf8',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=RAW,lineterminator='\n'); wr.writeheader()
        for seed,_,sha in checkpoints():
            fitted={}
            for arm in arms:
                path=heads/f'{arm}-s{seed}.safetensors'; p,m,z,meta=load_head(path)
                if (meta['arm'],meta['weight_seed'],meta['weight_sha'],meta['updates'])!=(arm,str(seed),sha,'960'):
                    raise ValueError(f'wrong registered fit: {path}')
                fitted[arm]=(p,m,z,digest(path))
            for surf in ('seen','held','shift'):
                for k,g in TEST_CELLS:
                    d=load_cell(cache,'test',seed,sha,surf,k,g)
                    ids=tuple(zip(d['draw'],d['family'],d['input_sha'],strict=True)); key=(surf,k,g)
                    if identical.setdefault(key,ids)!=ids: raise ValueError('raw histories/questions change between weights')
                    order=np.lexsort((d['family'],d['draw']))
                    for arm,(p,m,z,head_sha) in fitted.items():
                        base=dict(arm=arm,weight_seed=seed,weight_sha=sha,head_sha=head_sha,surface=surf,live=k,gap_windows=g)
                        for row,counts,h in _evaluate(d,arm,p,m,z,k,g):
                            if row is not None: wr.writerow({**base,**row})
                            else:
                                summaries.append({**base,**counts}); family_hits[seed,surf,k,g,arm]=h[order].reshape(len(TEST_DRAWS),8)
            fp.flush()
    with (output/'scores.csv').open('x',encoding='utf8',newline='') as fp:
        wr=csv.DictWriter(fp,tuple(summaries[0]),lineterminator='\n'); wr.writeheader(); wr.writerows(summaries)
    with (output/'gates.csv').open('x',encoding='utf8',newline='') as fp:
        wr=csv.writer(fp,lineterminator='\n'); wr.writerow(('arm','weight_seed','surface','conditions','answer_gate','source_gate','joint_gate'))
        for seed in (42,43,44):
            for arm in arms:
                for surf in ('seen','held','shift'):
                    rows=[r for r in summaries if (r['weight_seed'],r['arm'],r['surface'])==(seed,arm,surf)]
                    def passes(key): return int(all(10*r[key]>=9*r['paired_queries'] for r in rows))
                    aa=passes('paired_answer'); ss='' if arm=='ordinary' else passes('paired_source')
                    wr.writerow((arm,seed,surf,len(rows),aa,ss,'' if ss=='' else int(aa and ss)))
    if not factors_only: _effects(output,family_hits)


def predict(head,source):
    from episodic_scene_features import WindowEncoder
    p,mean,scale,meta=load_head(head); seed=int(meta['weight_seed'])
    _,path,sha=next(t for t in checkpoints() if t[0]==seed)
    if sha!=meta['weight_sha']: raise ValueError('wrong frozen RWKV for this head')
    if source=='-': data=json.load(sys.stdin)
    else:
        with open(source,encoding='utf8') as f: data=json.load(f)
    if not isinstance(data,dict) or set(data)!= {'blocks','queries'}: raise ValueError('only raw blocks and queries are accepted')
    blocks,q=(np.asarray(data[name]) for name in ('blocks','queries'))
    if (blocks.ndim!=2 or not len(blocks) or blocks.shape[1]!=8 or q.shape!=(2,8) or
        any(x.dtype.kind not in 'iu' or np.any((x<0)|(x>=128)) for x in (blocks,q))): raise ValueError('expected 8-token windows and two queries')
    enc=WindowEncoder(path); blocks=blocks.astype(np.int32); q=q.astype(np.int32)
    if meta['arm']=='ordinary':
        x=normalize(enc.ordinary(blocks[None],q[None]),mean,scale); ans=np.asarray(project(p,x).argmax(-1))[0]; addr=None
    else:
        s=empty_state(1)
        for i in range(0,len(blocks),32):
            chunk=normalize(enc.local(blocks[i:i+32]),mean,scale)
            for j in range(len(chunk)): s,_,_=write_window(p,s,chunk[j:j+1])
        ans,addr=read_queries(p,s,normalize(enc.local(q)[None],mean,scale)); ans=np.asarray(ans)[0]; addr=np.asarray(addr)[0].tolist()
    print(json.dumps({'answers':ans.tolist(),'read_addresses':addr},allow_nan=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='mode',required=True)
    e=sub.add_parser('assess'); e.add_argument('cache',type=Path); e.add_argument('heads',type=Path); e.add_argument('output',type=Path)
    e.add_argument('--factors-only',action='store_true',help='score the conditional cue-supervised head only')
    e=sub.add_parser('predict'); e.add_argument('head',type=Path); e.add_argument('input'); a=ap.parse_args()
    if a.mode=='assess': assess(a.cache,a.heads,a.output,factors_only=a.factors_only)
    else: predict(a.head,a.input)
