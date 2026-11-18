"""Strict portable policy artifacts, independent of the frozen RWKV weights."""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save

from ..model.role_binding import ARMS, Router, _validate_router, init_policy

FORMAT = 'rwkv7m.role_binding'
VERSION = '1'
_PARAMETERS = ('input', 'bias', 'out', 'out_bias')
_ROUTER_TENSORS = ('router_markers', 'router_lexicon')
_METADATA = {'format', 'version', 'arm', 'seed', 'weight_sha', 'updates', 'curriculum'}
_COUNT_ARMS = ('bag', 'marked-count')
_MARKED_ARMS = ('marked-count', 'marked-rwkv')


@lru_cache(maxsize=len(ARMS))
def _shapes(arm):
    params = init_policy(0, arm)
    width = params['input'].shape[0]
    return {**{name: value.shape for name, value in params.items()}, 'mean': (width,), 'scale': (width,)}


def _validate(tensors, meta):
    if (set(meta) != _METADATA or not all(isinstance(value, str) for value in meta.values()) or
            meta.get('format') != FORMAT or meta.get('version') != VERSION or meta.get('arm') not in ARMS or
            meta.get('curriculum') not in ('mixed', 'basic-only') or
            not re.fullmatch(r'0|[1-9][0-9]*', meta.get('seed', '')) or
            not re.fullmatch(r'0|[1-9][0-9]*', meta.get('updates', ''))):
        raise ValueError('invalid role-binding policy metadata')
    arm = meta['arm']
    if not (meta['weight_sha'] == 'none' if arm in _COUNT_ARMS else
            re.fullmatch('[0-9a-f]{64}', meta['weight_sha'])):
        raise ValueError('weight_sha must be none exactly for the no-RWKV arms, otherwise a SHA-256')
    shapes = _shapes(arm)
    marked = arm in _MARKED_ARMS
    expected = set(shapes) | (set(_ROUTER_TENSORS) if marked else set())
    if set(tensors) != expected:
        raise ValueError('unexpected or missing policy tensors/router')
    for name, shape in shapes.items():
        value = tensors[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'invalid shape, dtype or nonfinite policy tensor: {name}')
    if np.any(tensors['scale'] <= 0):
        raise ValueError('normalization scale must be positive')
    router = Router(*(tensors[name] for name in _ROUTER_TENSORS)) if marked else None
    if router is not None:
        _validate_router(router)
    return router


def save_policy(path, params, mean, scale, *, arm, seed, weight_sha, updates, router=None, curriculum='mixed'):
    """Save without overwriting; FP32 parameters and int32 router IDs stay exact."""
    if arm not in ARMS:
        raise ValueError(f'unknown role-binding arm: {arm}')
    if set(params) != set(_PARAMETERS):
        raise ValueError('unexpected or missing policy parameter names')
    if (arm in _MARKED_ARMS) != (router is not None):
        raise ValueError('a router is required exactly for marked arms')
    tensors = {name: np.ascontiguousarray(value) for name, value in params.items()}
    tensors.update(mean=np.ascontiguousarray(mean), scale=np.ascontiguousarray(scale))
    if router is not None:
        _validate_router(router)
        tensors.update({name: np.ascontiguousarray(value) for name, value in zip(_ROUTER_TENSORS, router)})
    meta = {'format': FORMAT, 'version': VERSION, 'arm': arm, 'seed': str(seed),
            'updates': str(updates), 'weight_sha': weight_sha, 'curriculum': curriculum}
    _validate(tensors, meta)
    payload = save(tensors, metadata=meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(payload)


def load_policy(path):
    """Return params, normalization, string metadata, and a router or None."""
    with safe_open(str(path), framework='np') as source:
        meta = dict(source.metadata() or {})
        tensors = {name: source.get_tensor(name) for name in source.keys()}
    router = _validate(tensors, meta)
    mean, scale = tensors.pop('mean'), tensors.pop('scale')
    for name in _ROUTER_TENSORS:
        tensors.pop(name, None)
    return tensors, mean, scale, meta, router
