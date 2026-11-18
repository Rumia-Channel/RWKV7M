"""Fit the §22.20 one-probe memory and a slightly larger ordinary readout. No test loader call."""
import argparse
import csv
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from benchmark_address_screening import checkpoints
from episodic_experiment import load_cell
from episodic_scene_data import ACTOR_TOKENS, OCCASION_TOKENS, TRAIN_CELLS
from rwkv7m.io.episodic_address import ARMS, PRIMARY_ARMS, save_head
from rwkv7m.model.episodic_address import BITS, hard_episode, init_head, normalize, project, soft_episode

STEPS,BATCH,INTERVAL=960,32,240
OPT=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.003))


class Samples(NamedTuple):
    blocks: np.ndarray
    valid: np.ndarray
    queries: np.ndarray
    ordinary: np.ndarray | None
    support: np.ndarray
    answers: np.ndarray


def pool(cache,split,seed,sha,*,lexical=False,counts_only=False):
    if split not in ('train','dev'): raise ValueError('fitting only accepts train/dev')
    groups=[]; width=max(k+g for k,g in TRAIN_CELLS)
    for k,g in TRAIN_CELLS:
        d=load_cell(cache,split,seed,sha,'seen',k,g,lexical=lexical,counts_only=counts_only)
        n=2*len(d['draw']); table=d['features']; dims=table.shape[-1]
        blocks=np.zeros((n,width,dims),np.float32); blocks[:,:k+g]=table[d['blocks'].reshape(n,-1)]
        valid=np.broadcast_to(np.arange(width)<k+g,(n,width))
        ordinary=None if counts_only else d['ordinary'].reshape(n,2,dims)
        groups.append(Samples(blocks,valid,np.repeat(table[d['queries']],2,axis=0),ordinary,
                              np.repeat(d['support'],2,axis=0),d['answers'].reshape(n,2)))
    return Samples(*(None if parts[0] is None else np.concatenate(parts,axis=0) for parts in zip(*groups,strict=True)))


def scaling(train,ordinary):
    dims=train.blocks.shape[-1]
    x=train.ordinary.reshape(-1,dims) if ordinary else np.concatenate((train.blocks[train.valid],train.queries.reshape(-1,dims)))
    mean=x.mean(axis=0,dtype=np.float64); std=x.std(axis=0,dtype=np.float64)
    return mean.astype(np.float32),(1./np.maximum(std,.03)).astype(np.float32)


def prepared(pool,arm,mean,scale):
    def norm(x): return normalize(jnp.asarray(x),mean,scale)
    if arm=='ordinary': return norm(pool.ordinary),jnp.asarray(pool.answers)
    return norm(pool.blocks),jnp.asarray(pool.valid),norm(pool.queries),jnp.asarray(pool.support),jnp.asarray(pool.answers)


def training_cue_bits(cache,seed,sha,*,counts_only=False):
    """Extra supervision for the registered diagnostic only; never used by the predictor or dev/test."""
    labels=[]
    for k,g in TRAIN_CELLS:
        d=load_cell(cache,'train',seed,sha,'seen',k,g,counts_only=counts_only); queries=d['feature_tokens'][d['queries']]
        axes=[]
        for tokens in (ACTOR_TOKENS,OCCASION_TOKENS):
            matches=(queries[...,None]==np.asarray(tokens)).any(axis=-2)
            if not np.all(matches.sum(-1)==1): raise ValueError('training query lacks an unambiguous cue')
            axes.append(matches.argmax(-1))
        codes=axes[0]+8*axes[1]
        labels.append(np.repeat(((codes[...,None]>>np.arange(BITS)) & 1).astype(np.float32),2,axis=0))
    return jnp.asarray(np.concatenate(labels))


def loss(params,sample,step,arm):
    if arm=='ordinary':
        features,gold=sample; nll=optax.softmax_cross_entropy_with_integer_labels(project(params,features),gold).mean()
        return nll,(nll,jnp.float32(0))
    blocks,valid,q,support,gold=sample[:5]
    beta=1.+3.*jnp.minimum(step/720.,1.)
    result=soft_episode(params,blocks,valid,q,beta)
    answer=jnp.take_along_axis(result.answers,gold[...,None],axis=-1)[...,0]
    nll=-jnp.log(jnp.maximum(answer,1e-10)).mean()
    if arm in ('source','factors'):
        located=jnp.take_along_axis(result.owners,support[...,None],axis=-1)[...,0]
        aux=-jnp.log(jnp.maximum(located,1e-10)).mean()
        if arm=='factors':
            # These identical projections are shared with soft_episode by XLA. Only the two
            # matched training windows and questions receive bit labels, never a hard test key.
            w=project(params,blocks)[jnp.arange(len(q))[:,None],support,:BITS]
            r=project(params,q)[...,BITS:2*BITS]; bits=sample[5]
            cue=(optax.sigmoid_binary_cross_entropy(w,bits)+optax.sigmoid_binary_cross_entropy(r,bits)).sum(-1).mean()
            return nll+aux+cue,(nll,aux,cue)
        return nll+aux,(nll,aux)
    return nll,(nll,jnp.float32(0))


@partial(jax.jit,static_argnames=('arm',))
def update(params,state,data,indices,steps,*,arm):
    def step(carry,row):
        p,opt=carry; ix,t=row; sample=tuple(a[ix] for a in data)
        (_,log),grad=jax.value_and_grad(loss,has_aux=True)(p,sample,t,arm)
        delta,opt=OPT.update(grad,opt,p)
        return (optax.apply_updates(p,delta),opt),jnp.stack(log)
    (params,state),metrics=jax.lax.scan(step,(params,state),(indices,steps))
    return params,state,metrics.mean(0)


def dev_pair(params,data,arm):
    if arm=='ordinary': pred=project(params,data[0]).argmax(-1)
    else: pred=hard_episode(params,*data[:3]).answers
    hit=np.asarray(pred)==np.asarray(data[-1]); return float(hit.reshape(-1,2,2).all(axis=1).mean())


def fit(cache,heads,seed,sha,arms,*,lexical=False,counts_only=False):
    view=dict(lexical=lexical,counts_only=counts_only)
    train=pool(cache,'train',seed,sha,**view); dev=pool(cache,'dev',seed,sha,**view)
    rng=np.random.default_rng(751099); batches=jnp.asarray(rng.integers(len(train.valid),size=(STEPS,BATCH),dtype=np.int32))
    for arm in arms:
        out=heads/f'{arm}-s{seed}.safetensors'; log=heads/f'learning-{arm}-s{seed}.csv'
        if out.exists() or log.exists(): raise FileExistsError(f'refusing to replace {out} or {log}')
        ordinary=arm=='ordinary'; params=init_head(seed,ordinary=ordinary,counts_only=counts_only); state=OPT.init(params)
        mean,scale=scaling(train,ordinary); inputs=prepared(train,arm,mean,scale); valid=prepared(dev,arm,mean,scale)
        if arm=='factors': inputs+= (training_cue_bits(cache,seed,sha,counts_only=counts_only),)
        elapsed=0.
        with log.open('x',encoding='utf8',newline='') as f:
            wr=csv.writer(f,lineterminator='\n'); wr.writerow(('arm','weight_seed','step','trainable_params','answer_nll','source_nll',*(('cue_nll',) if arm=='factors' else ()),'dev_paired_answer','update_seconds'))
            for start in range(0,STEPS,INTERVAL):
                end=start+INTERVAL; tick=perf_counter()
                params,state,metrics=update(params,state,inputs,batches[start:end],jnp.arange(start,end),arm=arm)
                nll,aux,*cue=map(float,np.asarray(metrics)); elapsed+=perf_counter()-tick
                if not np.isfinite(nll+aux+sum(cue)): raise FloatingPointError(f'nonfinite {arm} at {end}')
                paired=dev_pair(params,valid,arm)
                wr.writerow((arm,seed,end,sum(x.size for x in params.values()),nll,aux if arm in ('source','factors') else '',*cue,paired,elapsed)); f.flush()
        save_head(out,params,mean,scale,arm=arm,weight_seed=seed,weight_sha=sha,updates=STEPS,**view)
        print(f'{seed} {arm}, final seen-combination dev both-worlds {paired:.3f}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('cache',type=Path); ap.add_argument('heads',type=Path)
    ap.add_argument('--seed',type=int,choices=(42,43,44)); ap.add_argument('--arm',choices=ARMS)
    view=ap.add_mutually_exclusive_group()
    view.add_argument('--lexical',action='store_true',help='registered hybrid word-count view, including the factors arm')
    view.add_argument('--counts-only',action='store_true',help='train the capacity-matched factors head on token counts; cache unused')
    a=ap.parse_args()
    if a.counts_only and a.arm not in (None,'factors'): ap.error('--counts-only trains only the factors control')
    a.heads.mkdir(parents=True,exist_ok=True)
    models=((s,None,'none') for s in (42,43,44)) if a.counts_only else checkpoints()
    arms=('factors',) if a.counts_only else ((a.arm,) if a.arm else (ARMS if a.lexical else PRIMARY_ARMS))
    for s,_,sha in models:
        if a.seed is None or s==a.seed: fit(a.cache,a.heads,s,sha,arms,lexical=a.lexical,counts_only=a.counts_only)
