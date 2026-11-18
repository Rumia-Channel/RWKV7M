"""Materialize one ordinary RWKV at the FP32 midpoint of a fixed fine-tune and its own parent."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from rwkv7m.io.config import model_config_from_dict


def midpoint(parent, fixed, output):
    if output.exists(): raise FileExistsError(output)
    with parent.open('rb') as f: parent_sha = file_digest(f, 'sha256').hexdigest()
    with fixed.open('rb') as f: fixed_sha = file_digest(f, 'sha256').hexdigest()
    tensors = {}
    with safe_open(parent, framework='numpy') as p, safe_open(fixed, framework='numpy') as t:
        meta = t.metadata(); pm = p.metadata()
        # Older format-v1 exports omit defaulted config fields. Compare the configs the runtime actually loads.
        parent_cfg = model_config_from_dict(json.loads(pm['rwkv7m_config_json']))
        tuned_cfg = model_config_from_dict(json.loads(meta['rwkv7m_config_json']))
        if (meta['parent_weights_sha256'] != parent_sha or meta['curriculum_arm'] != 'fixed' or
                parent_cfg.use_screening or parent_cfg != tuned_cfg or p.keys() != t.keys()):
            raise ValueError('the fixed fine-tune and its exact registered parent are required')
        for k in p.keys():
            a, b = p.get_tensor(k), t.get_tensor(k)
            if a.shape != b.shape or a.dtype != b.dtype or a.dtype != np.float32:
                raise ValueError(f'unmatched FP32 weights: {k}')
            out = np.subtract(b, a); out *= np.float32(0.5); out += a
            if not np.isfinite(out).all(): raise FloatingPointError(f'nonfinite midpoint: {k}')
            tensors[k] = out
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata=meta | {'curriculum_arm': 'midpoint', 'interpolation_alpha': '0.5',
                                                     'fixed_source_sha256': fixed_sha})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'fixed', 'output'): parser.add_argument(name, type=Path)
    args = parser.parse_args(); midpoint(args.parent, args.fixed, args.output)
