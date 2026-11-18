"""One-probe episodic state for §22.20. No passage IDs or supervision at inference.

Each observed window predicts a gate, a binary write address and an eight-way
payload; each question predicts only a binary read address. The finite-code
marginal used for differentiable training is not used in the hard runtime.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

FEATURES, BITS, CLASSES = 256, 6, 8
SLOTS = 1 << BITS


def init_head(seed, *, ordinary=False, counts_only=False):
    if ordinary and counts_only: raise ValueError('count-only control is a memory head')
    hidden, out = (68, CLASSES) if ordinary else (118 if counts_only else 64, 2*BITS+1+CLASSES)
    rng = np.random.default_rng(seed)
    def matrix(d,n):
        return jnp.asarray(rng.normal(0.,d**-.5,size=(d,n)).astype(np.float32))
    return {'input':matrix(128 if counts_only else FEATURES,hidden),'bias':jnp.zeros(hidden,jnp.float32),
            'out':matrix(hidden,out),'out_bias':jnp.zeros(out,jnp.float32)}


def project(params, x):
    return jax.nn.gelu(x @ params['input'] + params['bias']) @ params['out'] + params['out_bias']


def normalize(x, mean, scale):
    return jnp.clip((x-mean)*scale,-12.,12.)


def _address(logits):
    return jnp.sum((logits>0).astype(jnp.int32) << jnp.arange(BITS,dtype=jnp.int32),axis=-1)


class State(NamedTuple):
    bank: jax.Array   # [lane,64,8], neither raw tokens nor annotated source positions
    occupied: jax.Array


def empty_state(lanes):
    return State(jnp.zeros((lanes,SLOTS,CLASSES),jnp.float32),jnp.zeros((lanes,SLOTS),bool))


def _store(state, address, enabled, value):
    lanes=jnp.arange(address.shape[0]); cell=state.bank[lanes,address]
    return State(state.bank.at[lanes,address].set(jnp.where(enabled[:,None],value,cell)),
                 state.occupied.at[lanes,address].set(enabled | state.occupied[lanes,address]))


@jax.jit
def write_window(params,state,x):
    """Consume one normalized window per lane. No query argument, no slot search."""
    z=project(params,x); address=_address(z[:,:BITS]); enabled=z[:,2*BITS]>0
    return _store(state,address,enabled,z[:,2*BITS+1:]),address,enabled


@jax.jit
def read_queries(params,state,q):
    """One indexed cell per question, -1 for an unwritten address; no state mutation."""
    address=_address(project(params,q)[...,BITS:2*BITS]); lanes=jnp.arange(len(q))[:,None]
    pred=state.bank[lanes,address].argmax(-1).astype(jnp.int32)
    return jnp.where(state.occupied[lanes,address],pred,-1),address


class Decisions(NamedTuple):
    answers: jax.Array
    read_address: jax.Array
    write_address: jax.Array
    wrote: jax.Array


@jax.jit
def hard_episode(params,blocks,valid,queries):
    """Batched replay for evaluation. Write traces are outputs to the scorer only."""
    def step(state,row):
        x,present=row; z=project(params,x); address=_address(z[:,:BITS]); enabled=(z[:,2*BITS]>0) & present
        return _store(state,address,enabled,z[:,2*BITS+1:]),(address,enabled)
    state,(w,enabled)=jax.lax.scan(step,empty_state(len(blocks)),(blocks.swapaxes(0,1),valid.T))
    pred,r=read_queries(params,state,queries)
    return Decisions(pred,r,w.T,enabled.T)


def _prob_codes(logits,beta):
    # Training only: independent Bernoulli bits, represented as a categorical over 64 fixed codes.
    code=((jnp.arange(SLOTS)[:,None] >> jnp.arange(BITS)) & 1).astype(jnp.float32)
    return jax.nn.softmax((logits @ code.T)*beta,axis=-1)


class SoftResult(NamedTuple):
    answers: jax.Array       # Includes uniform eight-way prediction on unoccupied mass.
    owners: jax.Array        # [B,Q,T]: probability the last write at the selected address was window t.


def soft_episode(params,blocks,valid,queries,beta):
    b=project(params,blocks); q=project(params,queries)
    write=_prob_codes(b[...,:BITS],beta)*(jax.nn.sigmoid(beta*b[...,2*BITS])*valid)[...,None]
    read=_prob_codes(q[...,BITS:2*BITS],beta)
    # Strict suffix product: a later write to the same physical address replaces this one.
    after=jnp.cumprod(1.-write[:,1:][:,::-1],axis=1)[:,::-1]
    after=jnp.concatenate((after,jnp.ones_like(write[:,:1])),axis=1)
    owners=jnp.einsum('bqs,bts->bqt',read,write*after)
    values=jax.nn.softmax(b[...,2*BITS+1:],axis=-1)
    miss=jnp.maximum(1.-owners.sum(axis=-1,keepdims=True),0.)
    return SoftResult(jnp.einsum('bqt,btc->bqc',owners,values)+miss/CLASSES,owners)
