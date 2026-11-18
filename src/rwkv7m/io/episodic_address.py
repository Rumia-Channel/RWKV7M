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


@lru_cache(maxsize=2)
def _shapes(ordinary):
    return {**{k:v.shape for k,v in init_head(0,ordinary=ordinary).items()},'mean':(FEATURES,),'scale':(FEATURES,)}


def _validate(tensors,meta):
    arm=meta.get('arm')
    if (meta.get('format')!=FORMAT or meta.get('version')!='1' or arm not in ARMS or
        meta.get('bits')!=str(BITS) or meta.get('classes')!=str(CLASSES) or
        meta.get('weight_seed') not in ('42','43','44') or not re.fullmatch('[0-9a-f]{64}',meta.get('weight_sha','')) or
        not re.fullmatch('[1-9][0-9]*',meta.get('updates',''))): raise ValueError('invalid episodic head metadata')
    shapes=_shapes(arm=='ordinary')
    if set(tensors)!=set(shapes): raise ValueError('unexpected episodic head tensor names')
    for name,x in tensors.items():
        if x.shape!=shapes[name] or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError(f'invalid shape, dtype or nonfinite tensor: {name}')
    if np.any(tensors['scale']<=0): raise ValueError('normalization scale must be positive')


def save_head(path, params, mean, scale, *, arm, weight_seed, weight_sha, updates):
    tensors={k:np.ascontiguousarray(x) for k,x in dict(params,mean=mean,scale=scale).items()}
    meta={'format':FORMAT,'version':'1','arm':arm,'weight_seed':str(weight_seed),'weight_sha':weight_sha,
          'updates':str(updates),'bits':str(BITS),'classes':str(CLASSES)}
    _validate(tensors,meta); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    save_file(tensors,str(path),metadata=meta)


def load_head(path):
    with safe_open(str(path),framework='np') as f:
        meta=dict(f.metadata() or {}); tensors={k:f.get_tensor(k) for k in f.keys()}
    _validate(tensors,meta); mean=tensors.pop('mean'); scale=tensors.pop('scale')
    return tensors,mean,scale,meta
