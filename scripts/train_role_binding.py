"""Train on one-read elementary problems AND three-read entrance-style problems. No midterm access."""
import argparse
import csv
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from benchmark_address_screening import checkpoints
from role_binding_data import TRAIN_CELLS
from role_binding_observations import load_cell, raw_cell
from rwkv7m.io.role_binding import save_policy
from rwkv7m.model.episodic_address import project, soft_episode
from rwkv7m.model.episodic_lexical import lexical_features, word_counts
from rwkv7m.model.role_binding import ARMS, induce_router, init_policy, route, soft_marked

STEPS, BATCH, INTERVAL = 960, 32, 240
OPT = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.003))

def training_cells(cache, seed, weight_sha, neural, basic_only):
    basic=[load_cell(cache,'train',seed,weight_sha,'seen',k,g) if neural else raw_cell('train','seen',k,g) for k,g in TRAIN_CELLS]
    if basic_only: return basic,[]
    from role_binding_mixed_features import exam_cell
    advanced=[exam_cell(cache,'advanced',seed,weight_sha,'advanced',g,neural=neural) for g in (0,8)]
    return basic,advanced


def public_queries(d):
    return np.concatenate((d['queries'],d['cards']),axis=1) if 'cards' in d else d['queries']


def observations(d,arm):
    if arm=='ordinary': return d['ordinary']
    t=d['feature_tokens']
    return word_counts(t) if arm in ('bag','marked-count') else lexical_features(d['features'],t)


def training_identity(groups):
    h=sha256()
    for d in groups:
        for public in d['input_sha']: h.update(bytes.fromhex(str(public)))
        h.update(d['answers'].astype('<i4',copy=False).tobytes())
    return h.hexdigest()


def pack(groups, matrices, arm, router, mean, scale):
    # Supervision boundary: only final object labels are packed. No source, value
    # path, counterfactual ID or desired address is present in an optimizer batch.
    width=max(d['blocks'].shape[2] for d in groups); parts=[]
    for d,obs in zip(groups,matrices,strict=True):
        f,w,n=d['blocks'].shape; b=f*w
        row={'gold':d['answers'].reshape(b,2)}
        z=np.clip((obs-mean)*scale,-12,12)
        if arm=='ordinary': row['ordinary']=z.reshape(b,-1,z.shape[-1])
        else:
            idx=d['blocks'].reshape(b,n); iq=np.repeat(public_queries(d),w,axis=0)
            x=np.zeros((b,width,z.shape[-1]),np.float32); x[:,:n]=z[idx]
            row.update(x=x,valid=np.broadcast_to(np.arange(width)<n,(b,width)))
            if router is not None:
                addresses=route(router,d['feature_tokens']); wr=np.full((b,width),-1,np.int32); wr[:,:n]=addresses[idx]
                row.update(wa=wr,ra=addresses[iq])
            else: row['q']=z[iq]
        parts.append(row)
    return {k:jnp.asarray(np.concatenate([r[k] for r in parts])) for k in parts[0]}


def follow_card(prob,reads):
    """Differentiable training surrogate: only a *predicted* class selects a card row."""
    answer=prob[:,:2]
    if reads!=1:
        table=prob[:,2:]
        for _ in range(reads-1): answer=answer @ table
    return answer


def loss(params,row,step,arm,reads):
    beta=1.+3.*jnp.minimum(step/720.,1.)
    if arm=='ordinary': prob=jax.nn.softmax(project(params,row['ordinary']),axis=-1)
    elif arm.startswith('marked-'):
        prob=soft_marked(params,row['x'],row['valid'],row['wa'],row['ra'],beta).answers
    else: prob=soft_episode(params,row['x'],row['valid'],row['q'],beta).answers
    p=jnp.take_along_axis(follow_card(prob,reads),row['gold'][...,None],axis=-1)
    return -jnp.log(jnp.maximum(p,1e-10)).mean()


@partial(jax.jit,static_argnames=('arm','second_reads'))
def update(p,opt,basic,second,ix1,ix2,starts,*,arm,second_reads):
    def one(p,opt,data,i,t,reads):
        sample=jax.tree.map(lambda a:a[i],data)
        val,grad=jax.value_and_grad(loss)(p,sample,t,arm,reads)
        delta,opt=OPT.update(grad,opt,p)
        return optax.apply_updates(p,delta),opt,val
    def pair(carry,rows):
        par,state=carry; i,j,t=rows
        par,state,l1=one(par,state,basic,i,t,1)
        par,state,l3=one(par,state,second,j,t+1,second_reads)
        return (par,state),jnp.stack((l1,l3))
    (p,opt),values=jax.lax.scan(pair,(p,opt),(ix1,ix2,starts))
    return p,opt,values.mean(axis=0)


def fit(heads,seed,sha,arm,basic,advanced):
    path=heads/f'{arm}-s{seed}.safetensors'; log=heads/f'learning-{arm}-s{seed}.csv'
    if path.exists() or log.exists(): raise FileExistsError(f'refusing to replace {path} or {log}')
    cells=basic+advanced; matrices=[observations(d,arm) for d in cells]
    scale_data=np.concatenate([x.reshape(-1,x.shape[-1]) for x in matrices])
    mean=scale_data.mean(axis=0,dtype=np.float64).astype(np.float32)
    scale=(1./np.maximum(scale_data.std(axis=0,dtype=np.float64),.03)).astype(np.float32)
    del scale_data
    router=None
    if arm.startswith('marked-'):
        words=np.concatenate([d['feature_tokens'][public_queries(d)].reshape(-1,8) for d in cells])
        router=induce_router(words)
    base=pack(basic,matrices[:len(basic)],arm,router,mean,scale)
    other=pack(advanced,matrices[len(basic):],arm,router,mean,scale) if advanced else base
    p=init_policy(seed,arm); opt=OPT.init(p); size=sum(v.size for v in p.values())
    rng=np.random.default_rng(767091)
    ix,jx=(rng.integers(len(d['gold']),size=(STEPS//2,BATCH),dtype=np.int32) for d in (base,other))
    bh=sha256(ix.astype('<i4',copy=False).tobytes()+jx.astype('<i4',copy=False).tobytes()).hexdigest()
    inputs=(training_identity(basic),training_identity(advanced) if advanced else ''); cumulative=0.
    with log.open('x',encoding='utf8',newline='') as file:
        writer=csv.writer(file,lineterminator='\n'); writer.writerow(('arm','seed','updates','basic_updates','advanced_updates','batch_worlds','trainable_params',
            'basic_nll','advanced_nll','update_seconds','basic_sha','advanced_sha','batch_sha'))
        for start in range(0,STEPS,INTERVAL):
            end=start+INTERVAL; sl=slice(start//2,end//2); tick=perf_counter()
            p,opt,val=update(p,opt,base,other,ix[sl],jx[sl],jnp.arange(start,end,2),arm=arm,second_reads=3 if advanced else 1)
            result=np.asarray(val); cumulative+=perf_counter()-tick
            if not np.isfinite(result).all(): raise FloatingPointError(f'nonfinite {arm} seed {seed} at {end}')
            writer.writerow((arm,seed,end,end//2 if advanced else end,end//2 if advanced else 0,BATCH,size,
                             float(result[0]) if advanced else float(result.mean()),float(result[1]) if advanced else '',cumulative,*inputs,bh)); file.flush()
    curriculum='mixed' if advanced else 'basic-only'
    save_policy(path,p,mean,scale,arm=arm,seed=seed,weight_sha=sha,updates=STEPS,router=router,curriculum=curriculum)
    print(f'{arm} s{seed}: {curriculum}, final interval NLL {result.round(3).tolist()}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('cache',type=Path); ap.add_argument('heads',type=Path)
    ap.add_argument('--arm',action='append',choices=ARMS); ap.add_argument('--seed',type=int,choices=(42,43,44))
    ap.add_argument('--basic-only',action='store_true',help='matched 960-update basic-only control for the two induced routers')
    a=ap.parse_args(); a.heads.mkdir(parents=True,exist_ok=True); chosen=a.arm or (('marked-count','marked-rwkv') if a.basic_only else ARMS)
    if a.basic_only and any(not arm.startswith('marked-') for arm in chosen): ap.error('--basic-only is registered for marked arms')
    for neural in (False,True):
        arms=[arm for arm in chosen if (arm not in ('bag','marked-count'))==neural]
        if not arms: continue
        models=checkpoints() if neural else ((s,None,'none') for s in (42,43,44))
        for s,_,sha in models:
            if a.seed is not None and a.seed!=s: continue
            basic,advanced=training_cells(a.cache,s,sha,neural,a.basic_only)
            for arm in arms: fit(a.heads,s,sha,arm,basic,advanced)


if __name__=='__main__': main()
