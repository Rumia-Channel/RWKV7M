"""Streaming lexical adapter for the explicit, 16-address research vocabulary."""
import numpy as np

from rwkv7m.cli.prepare_retrieval import STORE_TOKEN

COPY = 5
KEY_BASE, VALUE_BASE, COUNT = 16, 32, 16


def parse_chunk(previous, tokens):
    """Emit [lane,event,(kind,dst,arg)] only when a visible 3-token event completes.

    A lane with no event at another lane's position emits NOP. The two-token
    carry is owned (not a view retaining the previous chunk). No query or label input.
    """
    tokens = np.asarray(tokens); previous = np.asarray(previous)
    if tokens.ndim != 2 or tokens.dtype != np.int32 or previous.shape != (tokens.shape[0],2) or previous.dtype != np.int32:
        raise ValueError('expected int32 tokens[B,T] and previous[B,2]')
    window = np.concatenate((previous,tokens),axis=1)
    marker,dst,arg = window[:,:-2],window[:,1:-1],window[:,2:]
    literal = (marker == STORE_TOKEN) & (arg >= VALUE_BASE) & (arg < VALUE_BASE+COUNT)
    copy = (marker == COPY) & (arg >= KEY_BASE) & (arg < KEY_BASE+COUNT)
    legal = (dst >= KEY_BASE) & (dst < KEY_BASE+COUNT) & (literal | copy)
    columns = legal.any(0)
    kind = np.where(legal[:,columns],1+copy[:,columns],0)
    events = np.empty((*kind.shape,3),dtype=np.int32)
    events[:,:,0] = kind
    events[:,:,1] = np.where(kind,dst[:,columns]-KEY_BASE,0)
    events[:,:,2] = np.where(kind,arg[:,columns]-np.where(kind==2,KEY_BASE,VALUE_BASE),0)
    return window[:,-2:].copy(),events
