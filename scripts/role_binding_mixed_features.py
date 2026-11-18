"""Raw public cards and frozen observations for the advanced training and sealed midterm forms."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_address_screening import checkpoints
from episodic_scene_features import WindowEncoder, _atomic_cost, _atomic_npz
from role_binding_exam import ADV_DRAWS, MID_DRAWS, make_exam


def conditions(form):
    if form=='advanced': return ADV_DRAWS,16,('advanced',),(0,8)
    if form=='midterm': return MID_DRAWS,8,('midterm','familiar'),(0,128)
    raise ValueError(f'unknown examination form: {form}')


def public_digest(blocks,queries,cards,reads):
    # Cards are in public object-class order, independent of their order on the page.
    h=sha256(b'rwkv7m.role-binding.card.v1\0')
    for x in ((reads,),blocks,queries,cards): h.update(np.asarray(x,dtype='<i4').tobytes())
    return h.hexdigest()


def raw_exam(form,surface,gap):
    draws,size,surfaces,gaps=conditions(form)
    if surface not in surfaces or gap not in gaps: raise ValueError('unregistered exam condition')
    groups=[make_exam(form,d,size,gap,familiar=surface=='familiar') for d in draws]
    b,q,c,s,y,v=(np.concatenate([getattr(x,k) for x in groups]) for k in ('blocks','queries','cards','support','answers','values'))
    reads=groups[0].reads; f,w,n,l=b.shape
    if (l!=8 or w!=4 or s.shape!=(f,4,2,reads) or y.shape!=(f,4,2) or not np.array_equal(v[...,-1],y)
        or np.any(s<0) or np.any(s>=n) or np.any(y<0) or np.any(y>=8)): raise ValueError('invalid answer or source layout')
    tokens,ids=np.unique(np.concatenate([a.reshape(-1,8) for a in (b,q,c)]),axis=0,return_inverse=True)
    ids=ids.astype(np.int32,copy=False).reshape(-1); ib,iq,ic=np.split(ids,[f*w*n,f*w*n+f*2])
    hashes=np.asarray([public_digest(bi,qi,ci,reads) for bi,qi,ci in zip(b,q,c,strict=True)])
    return dict(feature_tokens=tokens,blocks=ib.reshape(f,w,n),queries=iq.reshape(f,2),cards=ic.reshape(f,8),
                answers=y,support=s,values=v,reads=reads,input_sha=hashes,
                draw=np.repeat(np.asarray(draws,np.int32),size),family=np.tile(np.arange(size,dtype=np.int32),len(draws)))


def _path(cache,form,seed,surface,gap): return Path(cache)/f'role-{form}-{surface}-s{seed}-g{gap}.npz'


def exam_cell(cache,form,seed,sha,surface,gap,*,neural):
    d=raw_exam(form,surface,gap)
    if not neural: return d
    path=_path(cache,form,seed,surface,gap)
    with np.load(path,allow_pickle=False) as a:
        if (a['format'].item()!='role-card-observation-v1' or a['weight_sha'].item()!=sha or a['seed'].item()!=seed or
            not np.array_equal(a['input_sha'],d['input_sha']) or not np.array_equal(a['feature_tokens'],d['feature_tokens'])):
            raise ValueError(f'wrong frozen model, vocabulary or public problem: {path}')
        f,o=a['features'],a['ordinary']
    if (f.shape!=(len(d['feature_tokens']),256) or o.shape!=(len(d['blocks']),4,10,256) or
        any(x.dtype!=np.float32 or not np.isfinite(x).all() for x in (f,o))): raise ValueError(f'invalid neural observations: {path}')
    return dict(d,features=f,ordinary=o)


def build(cache,form,chosen=None):
    _,_,surfaces,gaps=conditions(form); cache.mkdir(parents=True,exist_ok=True)
    for seed,path,sha in checkpoints():
        if chosen is not None and seed!=chosen: continue
        start=perf_counter(); enc=WindowEncoder(path); setup=perf_counter()-start
        for surface in surfaces:
            for gap in gaps:
                dest=_path(cache,form,seed,surface,gap); cost=dest.with_suffix('.cost.csv')
                if dest.exists() or cost.exists(): raise FileExistsError(dest)
                d=raw_exam(form,surface,gap); f,w,n=d['blocks'].shape; t=d['feature_tokens']; before=dict(enc.cost)
                start=perf_counter(); local=enc.local(t); ls=perf_counter()-start
                root_cards=t[np.concatenate((d['queries'],d['cards']),axis=1)]
                # Ten independent forks from the completed prefix: two roots + eight public card queries.
                q=np.repeat(root_cards,w,axis=0).reshape(f*w,5,2,8).transpose(1,0,2,3)
                start=perf_counter(); cont=enc.ordinary_many(t[d['blocks'].reshape(f*w,n)],q).transpose(1,0,2,3).reshape(f,w,10,256); cs=perf_counter()-start
                _atomic_npz(dest,dict(format='role-card-observation-v1',seed=seed,weight_sha=sha,input_sha=d['input_sha'],
                                      feature_tokens=t,features=local,ordinary=cont))
                row=dict(form=form,surface=surface,gap=gap,seed=seed,weight_sha=sha,families=f,unique_windows=len(t),
                         weight_setup_seconds=setup,local_seconds=ls,ordinary_seconds=cs,
                         **{k:enc.cost[k]-before[k] for k in before}); setup=0.
                _atomic_cost(cost,row,list(row)); print(f'{dest.name}: {f} quartets, {len(t)} distinct utterances',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('cache',type=Path)
    ap.add_argument('--form',choices=('advanced','midterm'),required=True); ap.add_argument('--seed',type=int,choices=(42,43,44))
    args=ap.parse_args(); build(args.cache,args.form,args.seed)
