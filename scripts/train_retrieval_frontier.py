"""Matched-budget warm starts on normal RWKV. The held-out panel is a separate command."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from retrieval_frontier import CELLS, Frontier, calibration_index, fixed_cell, make_examples, to_next_token_batch
from retrieval_paired_readout import paired_final_logits
from rwkv7m import create_train_runtime, load_model_safetensors, load_runtime_params, train_batch
from rwkv7m.io.flax_checkpoint import load_train_checkpoint, save_train_checkpoint
from rwkv7m.io.safetensors import save_model_safetensors
from rwkv7m.model.state import init_rwkv_state, init_screen_state


SEEDS = (42, 43, 44)
ARMS = ('adaptive', 'fixed', 'yoked')
STEPS, WINDOW, TRAIN_PAIRS, CAL_PAIRS = 96, 24, 2, 32
CAL_SEED = 58731


def read_latest(directory):
    name = (directory / 'latest').read_text().strip()
    if name not in {f'step-{s:03d}' for s in range(0, STEPS + 1, WINDOW)}:
        raise ValueError(f'invalid checkpoint name: {name}')
    path = directory / name
    return path, json.loads((path / 'study.json').read_text())


def calibrate(model, cfg, step):
    """One independent window per boundary, shared across arms and seeds. No training or final test labels."""
    state = init_rwkv_state(16, cfg), init_screen_state(16, cfg.screening)
    counts = {}
    anchor_hits = None
    for gap, writes, anchor in [(*c, False) for c in CELLS] + [(32, 0, True)]:
        ids, target = make_examples(CAL_SEED, calibration_index(step // WINDOW), CAL_PAIRS,
                                    gap=gap, writes=writes, anchor=anchor)
        hits = 0
        for start in range(0, CAL_PAIRS, 8):
            logits = np.asarray(paired_final_logits(model, ids[start:start + 8], *state))
            if not np.isfinite(logits).all():
                raise FloatingPointError('nonfinite calibration logits')
            hits += int((logits.argmax(-1) == target[start:start + 8]).all(axis=(1, 2)).sum())
        if anchor:
            anchor_hits = hits
        else:
            counts[gap, writes] = (hits, CAL_PAIRS)
    return counts, anchor_hits


def save_boundary(directory, state, cfg, source_meta, record, policy, previous):
    step = int(state.step); name = f'step-{step:03d}'
    tmp = directory / (name + '.partial'); path = directory / name
    tmp.mkdir(exist_ok=True)
    meta = dict(checkpoint_step=step, parent_weights_sha256=record['protocol']['parent_sha256'],
                curriculum_arm=record['protocol']['arm'], weight_seed=record['protocol']['weight_seed'])
    # The native resume API also writes a default weight export. Replace it inside the unpublished directory:
    # these 128 symbolic IDs are not the repository's natural-language tokenizer.
    save_train_checkpoint(tmp, state, cfg, dataset_position=step, metadata=meta)
    tokenizer = {k: v for k, v in source_meta.items() if k.startswith(('tokenizer_', 'retrieval_'))}
    save_model_safetensors(tmp / 'model.safetensors', state.params, cfg, tokenizer_metadata=tokenizer, metadata=meta)
    record['step'] = step
    record['policy'] = None if policy is None else policy.state_dict()
    (tmp / 'study.json').write_text(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
    if path.exists():  # Completed directory not yet pointed to when an earlier process was interrupted.
        if (directory / 'latest').exists() and read_latest(directory)[0] == path:
            raise RuntimeError(f'refusing to replace committed checkpoint {path}')
        shutil.rmtree(path)
    tmp.replace(path)
    ptr = directory / 'latest.tmp'; ptr.write_text(name + '\n'); os.replace(ptr, directory / 'latest')
    if previous is not None and previous != path:
        shutil.rmtree(previous)
    return path


def train_one(root, parents, arm, seed, until):
    directory = root / f'{arm}-s{seed}'; directory.mkdir(parents=True, exist_ok=True)
    parent = parents / f'long-s{seed}.safetensors'
    params, cfg, meta = load_model_safetensors(parent)
    if cfg.use_screening or cfg.vocab_size != 128:
        raise ValueError('expected a no-screening synthetic parent')
    donor = SEEDS[(SEEDS.index(seed) + 1) % len(SEEDS)] if arm == 'yoked' else None
    replay = None
    if donor is not None:
        _, donated = read_latest(root / f'adaptive-s{donor}')
        if donated['step'] != STEPS:
            raise ValueError('yoked control needs the completed adaptive donor')
        replay = donated['schedule']
    protocol = dict(version=1, arm=arm, weight_seed=seed, data_seed=51000 + seed, cal_seed=CAL_SEED,
                    steps=STEPS, window=WINDOW, cal_pairs=CAL_PAIRS, train_pairs=TRAIN_PAIRS,
                    parent_sha256=sha256(parent.read_bytes()).hexdigest(), donor=donor,
                    replay_sha256=sha256(json.dumps(replay).encode()).hexdigest() if replay is not None else None)
    runtime, state = create_train_runtime(jax.random.key(seed), cfg, batch_size=8, total_steps=STEPS)
    policy = Frontier() if arm == 'adaptive' else None
    previous = None
    if (directory / 'latest').exists():
        previous, record = read_latest(directory)
        if record['protocol'] != protocol: raise ValueError('resume protocol mismatch')
        state, _, _ = load_train_checkpoint(previous, state)
        if int(state.step) != record['step'] or len(record['schedule']) != record['step']:
            raise ValueError('resume state does not match the recorded next data index')
        if policy is not None: policy = Frontier.from_state_dict(record['policy'])
    else:
        load_runtime_params(runtime, params)
        record = dict(protocol=protocol, schedule=[], updates=[], calibrations=[], train_seconds=0.0, calibration_seconds=0.0)
    first = int(state.step)
    for step in range(first, until + 1):
        if step % WINDOW == 0 and (previous is None or step != first):
            start = time.perf_counter(); counts, anchor_hits = calibrate(state.model, cfg, step)
            summary = policy.observe(counts, anchor_hits / CAL_PAIRS) if policy is not None else None
            record['calibrations'].append(dict(step=step, hits={f'{g},{w}': n for (g,w),(n,_) in counts.items()},
                                               anchor_hits=anchor_hits, mastered=None if summary is None else summary['mastered']))
            record['calibration_seconds'] += time.perf_counter() - start
            previous = save_boundary(directory, state, cfg, meta, record, policy, previous)
            print(f'{arm} seed={seed} step={step} train={record["train_seconds"]:.2f}s calibration={record["calibration_seconds"]:.2f}s', flush=True)
        if step == until: break
        start = time.perf_counter()
        cell = policy.cell(step) if policy is not None else fixed_cell(step) if arm == 'fixed' else replay[step]
        anchor = cell is None; gap, writes = (32, 0) if anchor else cell
        if anchor != (step % 4 == 3): raise ValueError('incorrect anchor timing')
        ids, labels = make_examples(protocol['data_seed'], step, TRAIN_PAIRS, gap=gap, writes=writes, anchor=anchor)
        batch = {k: jnp.asarray(v.reshape(8, 256)) for k,v in to_next_token_batch(ids, labels).items()}
        state, metrics = train_batch(state, batch, runtime, phase='read_write', carry_state=False)
        loss, applied, finite_grad, finite_params = (float(metrics[k]) for k in
            ('total_loss', 'gradient_update_applied', 'gradient_all_finite', 'parameter_all_finite'))
        if not np.isfinite(loss) or (applied, finite_grad, finite_params) != (1., 1., 1.) or int(state.step) != step + 1:
            raise FloatingPointError(f'failed update {arm}/{seed}/{step}: loss={loss}')
        record['train_seconds'] += time.perf_counter() - start
        record['schedule'].append(None if anchor else [gap, writes]); record['updates'].append(dict(loss=loss, tokens=ids.size, answers=labels.size))
    return previous


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--parents', type=Path, default=Path('experiments/retention/parents'))
    p.add_argument('--arm', choices=ARMS, action='append', help='default: all three, adaptive donors first')
    p.add_argument('--seed', choices=SEEDS, type=int, action='append', help='default: 42, 43, 44')
    p.add_argument('--until-step', choices=tuple(range(0, STEPS + 1, WINDOW)), type=int, default=STEPS,
                   help='pause after saving a boundary; invocation on the same output resumes')
    args = p.parse_args()
    for arm in ARMS:
        if args.arm and arm not in args.arm: continue
        for seed in args.seed or SEEDS:
            train_one(args.output, args.parents, arm, seed, args.until_step)
