"""Controlled 128-ID task: replace the last-state channels by a local word-count skip.

No grammar, object dictionary, cue class, history lookup or trained projection.
The first 128 channels remain the actual mean recurrent RWKV hidden state.
"""
import numpy as np


def word_counts(tokens):
    tokens=np.asarray(tokens)
    if tokens.shape[-1:]!=(8,) or tokens.dtype.kind not in 'iu' or np.any(tokens<0) or np.any(tokens>=128):
        raise ValueError('expected 8 public vocabulary tokens')
    flat=tokens.reshape(-1,8); counts=np.zeros((len(flat),128),np.float32)
    np.add.at(counts,(np.arange(len(flat))[:,None],flat),np.float32(.125))
    return counts.reshape(*tokens.shape[:-1],128)


def lexical_features(hidden, tokens):
    hidden=np.asarray(hidden)
    if hidden.dtype!=np.float32 or hidden.shape[-1:]!=(256,): raise ValueError('expected 256 FP32 neural features')
    result=np.empty_like(hidden); result[...,:128]=hidden[...,:128]
    result[...,128:]=word_counts(tokens)
    return result
