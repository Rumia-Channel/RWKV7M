"""Blind addressed RWKV reader for the explicit STORE/COPY/QUERY research grammar.

The persistent matrix contains RWKV hidden vectors, not token IDs or answers.
A finite 16×16 literal alphabet permits caching the local neural write encoder.
There is no ordinary-trunk prefill or computed expression solver in this route.
"""
from dataclasses import replace
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from addressed_binding_tokens import COUNT, KEY_BASE, VALUE_BASE, parse_chunk
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, FILLER_TOKEN, QUERY_TOKEN, STORE_TOKEN
from rwkv7m.io import load_model_safetensors
from rwkv7m.model.address_screening import apply_events_jit, initial_memory, make_address_codes, read_memory_jit
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


@nnx.jit
def _encode(model, tokens, state):
    hidden, *_ = model.compute_recurrent_hidden(tokens,*state,phase='read_write',deterministic=True)
    return hidden[:,-1,:].astype(jnp.float32)


@nnx.jit
def _predict(model, hidden, bound):
    logits = model.compute_logits(hidden).astype(jnp.float32)
    return jnp.where(~bound,-2,jnp.where(jnp.isfinite(logits).all(-1),logits.argmax(-1),-1)).astype(jnp.int32)


class FrozenBindingCodec:
    """Materialize the exact hidden write features of a frozen synthetic checkpoint."""
    def __init__(self,checkpoint):
        params,cfg,_ = load_model_safetensors(checkpoint)
        if cfg is None or cfg.use_screening or cfg.vocab_size!=128:
            raise ValueError('expected an ordinary 128-ID RWKV checkpoint')
        cfg=replace(cfg,remat_blocks=False)
        self.model=load_linen_params_into_nnx(initialize_nnx_model(jax.random.key(0),cfg),params)
        self.embeddings=np.asarray(params['token_embedding']['embedding'][KEY_BASE:KEY_BASE+COUNT],dtype=np.float32)
        keys=np.repeat(np.arange(COUNT,dtype=np.int32)+KEY_BASE,COUNT)
        values=np.tile(np.arange(COUNT,dtype=np.int32)+VALUE_BASE,COUNT)
        rows=np.full((COUNT**2,14),FILLER_TOKEN,np.int32)
        rows[:,0]=STORE_TOKEN; rows[:,1]=keys; rows[:,2]=values; rows[:,-3]=QUERY_TOKEN;rows[:,-2]=keys;rows[:,-1]=ANSWER_MARKER_TOKEN
        batch=32; zero=init_rwkv_state(batch,cfg),init_screen_state(batch,cfg.screening)
        self.payloads=jnp.concatenate([_encode(self.model,jnp.asarray(rows[i:i+batch]),zero) for i in range(0,len(rows),batch)]).reshape(COUNT,COUNT,cfg.d_model)
        if not bool(jnp.isfinite(self.payloads).all()): raise FloatingPointError('nonfinite local RWKV write representation')

    def local_predictions(self):
        return np.asarray(_predict(self.model,self.payloads,jnp.ones((COUNT,COUNT),jnp.bool_)))


class StreamState(NamedTuple):
    memory: object
    previous: np.ndarray


class AddressedReader:
    def __init__(self,codec,mode):
        self.codec=codec
        self.q,self.u=map(jnp.asarray,make_address_codes(codec.embeddings,mode))

    def start(self,lanes):
        return StreamState(initial_memory(lanes,COUNT,self.codec.payloads.shape[-1]),np.full((lanes,2),-1,np.int32))

    def advance(self,state,tokens):
        previous,events=parse_chunk(state.previous,tokens)
        memory=state.memory
        if events.shape[1]: memory=apply_events_jit(memory,self.q,self.u,self.codec.payloads,jnp.asarray(events))
        return StreamState(memory,previous)

    def query(self,state,questions):
        """Read independent full QUERY/key/ANSWER triples without changing the carry."""
        q=np.asarray(questions)
        if (q.ndim!=3 or q.shape[0]!=state.previous.shape[0] or q.shape[-1]!=3 or q.dtype!=np.int32 or
            np.any(q[...,0]!=QUERY_TOKEN) or np.any(q[...,2]!=ANSWER_MARKER_TOKEN) or
            np.any((q[...,1]<KEY_BASE)|(q[...,1]>=KEY_BASE+COUNT))):
            raise ValueError('expected visible int32 queries [B,N,3] in the binding vocabulary')
        hidden,valid=read_memory_jit(state.memory,self.q,jnp.asarray(q[...,1]-KEY_BASE))
        return np.asarray(_predict(self.codec.model,hidden,valid))

    def predict(self,h,q):
        h=np.asarray(h);q=np.asarray(q)
        if h.ndim!=3 or q.ndim!=4 or h.shape[:2]!=q.shape[:2]: raise ValueError('unmatched family/world dimensions')
        lanes=h.shape[0]*h.shape[1]
        return self.query(self.advance(self.start(lanes),h.reshape(lanes,-1)),q.reshape(lanes,-1,3)).reshape(q.shape[:-1])
