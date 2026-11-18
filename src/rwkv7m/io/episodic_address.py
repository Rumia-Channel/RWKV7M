"""Portable trained heads for the episode-address experiment, separate from frozen RWKV weights."""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from ..model.episodic_address import BITS, CLASSES, FEATURES, init_head

PRIMARY_ARMS=('endpoint','source','ordinary')
ARMS=(*PRIMARY_ARMS,'factors')
FORMAT='rwkv7m.episodic_address'
LEXICAL_FORMAT='rwkv7m.episodic_address.lexical'
COUNT_FORMAT='rwkv7m.episodic_address.counts'


@lru_cache(maxsize=3)
def _shapes(ordinary,counts_only):
    width=128 if counts_only else FEATURES
    return {**{k:v.shape for k,v in init_head(0,ordinary=ordinary,counts_only=counts_only).items()},'mean':(width,),'scale':(width,)}


def _validate(tensors,meta):
    arm=meta.get('arm')
    counts_only=meta.get('format')==COUNT_FORMAT
    if (meta.get('format') not in (FORMAT,LEXICAL_FORMAT,COUNT_FORMAT) or meta.get('version')!='1' or arm not in ARMS or
        (counts_only and arm!='factors') or
        meta.get('bits')!=str(BITS) or meta.get('classes')!=str(CLASSES) or
        meta.get('weight_seed') not in ('42','43','44') or
        not (meta.get('weight_sha')=='none' if counts_only else re.fullmatch('[0-9a-f]{64}',meta.get('weight_sha',''))) or
        not re.fullmatch('[1-9][0-9]*',meta.get('updates',''))): raise ValueError('invalid episodic head metadata')
    shapes=_shapes(arm=='ordinary',meta['format']==COUNT_FORMAT)
    if set(tensors)!=set(shapes): raise ValueError('unexpected episodic head tensor names')
    for name,x in tensors.items():
        if x.shape!=shapes[name] or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError(f'invalid shape, dtype or nonfinite tensor: {name}')
    if np.any(tensors['scale']<=0): raise ValueError('normalization scale must be positive')


def save_head(path, params, mean, scale, *, arm, weight_seed, weight_sha, updates, lexical=False, counts_only=False):
    if lexical and counts_only: raise ValueError('feature views are mutually exclusive')
    tensors={k:np.ascontiguousarray(x) for k,x in dict(params,mean=mean,scale=scale).items()}
    fmt=COUNT_FORMAT if counts_only else LEXICAL_FORMAT if lexical else FORMAT
    meta={'format':fmt,'version':'1','arm':arm,'weight_seed':str(weight_seed),'weight_sha':weight_sha,
          'updates':str(updates),'bits':str(BITS),'classes':str(CLASSES)}
    _validate(tensors,meta); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    save_file(tensors,str(path),metadata=meta)


def load_head(path):
    with safe_open(str(path),framework='np') as f:
        meta=dict(f.metadata() or {}); tensors={k:f.get_tensor(k) for k in f.keys()}
    _validate(tensors,meta); mean=tensors.pop('mean'); scale=tensors.pop('scale')
    return tensors,mean,scale,meta
