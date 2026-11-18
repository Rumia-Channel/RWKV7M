"""Fine-tune matched ordinary RWKVs; only one arm learns from its frozen suffix teacher."""
import argparse
import csv
from dataclasses import replace
from hashlib import file_digest, sha256
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from binding_distill_data import training_batch
from retrieval_paired_readout import branched_final_logits, compiled_branched_logits
from rwkv7m.io import load_model_safetensors, save_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ARMS = ('ce', 'distill')
STEPS, FAMILIES, SUFFIX = 64, 2, 64
LR, TEMP, KD_WEIGHT = 3e-5, 2., .5


def digest(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def distill_loss(logits, labels, teacher, write_position, weight):
    """The same teacher distribution supervises both nuisance variants, never the old key.

    Worlds are nuisance-major, then final-write version. Teacher [family, version, vocab]
    reads only two suffixes. The independent hard-label loss covers ALL worlds and questions.
    """
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    if teacher is None:  # Near/overwrite anchors: no suffix exists for this objective.
        return ce, (ce, jnp.float32(0), jnp.int32(0))
    families, worlds, _, vocab = logits.shape
    f, w = jnp.arange(families)[:, None], jnp.arange(worlds)[None, :]
    chosen = logits[f, w, write_position[:, None], :].reshape(families, 2, 2, vocab)
    truth = labels[f, w, write_position[:, None]].reshape(families, 2, 2)
    teacher = jax.lax.stop_gradient(teacher)
    logp = jax.nn.log_softmax(teacher / TEMP)[:, None, :, :]
    logq = jax.nn.log_softmax(chosen / TEMP)
    mask = teacher.argmax(-1)[:, None, :] == truth
    count = jnp.sum(mask, dtype=jnp.int32)
    kl = jnp.sum(jnp.where(mask, (jnp.exp(logp) * (logp - logq)).sum(-1), 0)) / jnp.maximum(count, 1)
    return ce + weight * TEMP**2 * kl, (ce, kl, count)


@nnx.jit
def _step(model, optimizer, history, queries, labels, initial, teacher, positions, weight):
    def objective(m):
        logits = branched_final_logits(m, history, queries, *initial)
        return distill_loss(logits, labels, teacher, positions, weight)

    (loss, (ce, kl, count)), grads = nnx.value_and_grad(objective, has_aux=True)(model)
    norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return jnp.stack((loss, ce, kl, count, norm))


def train_pair(directory, parents, seed):
    parent = parents / f'long-s{seed}.safetensors'
    source_sha = digest(parent)
    with (ROOT / 'experiments/binding-state/guarded/weights.csv').open(encoding='utf8') as f:
        registered = next(r for r in csv.DictReader(f) if int(r['weight_seed']) == seed)
    if registered['checkpoint_sha256'] != source_sha:
        raise ValueError(f'parent checksum does not match registered seed {seed}')
    params, cfg, source = load_model_safetensors(parent)
    parameter_count = sum(x.size for x in jax.tree.leaves(params))
    if cfg.use_screening or cfg.vocab_size != 128 or parameter_count != int(registered['parameters']):
        raise ValueError('expected an ordinary 128-token RWKV matching the registered parameter count')
    directory.mkdir(parents=True, exist_ok=True)
    paths = {a: directory / f'{a}-s{seed}.safetensors' for a in ARMS}
    updates, cost = (directory / f'{kind}-s{seed}.csv' for kind in ('updates', 'costs'))
    if any(p.exists() for p in (*paths.values(), updates, cost)):
        raise FileExistsError(f'seed {seed} already has results in {directory}')
    use_cfg = replace(cfg, remat_blocks=False)

    def model():
        return load_linen_params_into_nnx(initialize_nnx_model(jax.random.key(seed), use_cfg), params)

    def initial(rows):
        return init_rwkv_state(rows, use_cfg), init_screen_state(rows, use_cfg.screening)

    teacher, students = model(), {arm: model() for arm in ARMS}
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(LR))
    optim = {a: nnx.Optimizer(m, tx, wrt=nnx.Param) for a, m in students.items()}
    full_initial, local_initial = initial(FAMILIES * 4), initial(FAMILIES * 2)
    data_seed = 64100 + seed
    all_tokens = all_answers = suffix_tokens = accepted = 0
    started = perf_counter(); elapsed = dict.fromkeys(ARMS, 0.0); teacher_seconds = 0.
    with updates.open('x', encoding='utf8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n'); wr.writerow(('weight_seed','data_seed','step','arm','input_sha256','student_tokens','answers','shared_teacher_tokens','teacher_correct_rows','loss','ce','teacher_kl','grad_norm','seconds'))
        for step in range(STEPS):
            h, q, y, eligible = training_batch(data_seed, step, FAMILIES)
            data_hash = sha256()
            for x in (h,q,y): data_hash.update(x.astype('<i4', copy=False).tobytes())
            data_hash.update(eligible.tobytes()); signature = data_hash.hexdigest()
            raw_cost = h.size + q.size; all_tokens += raw_cost; all_answers += y.size
            teach = positions = None; local_cost = 0
            if eligible.any():
                if not np.all(eligible.sum(-1) == 1) or not np.all(eligible == eligible[:, :1]):
                    raise ValueError('mixed family requires one common updated query position')
                pos = eligible[:, 0].argmax(-1)
                q_local = q[np.arange(FAMILIES)[:, None], np.arange(2)[None, :], pos[:, None], :][:, :, None, :]
                # The nuisance variants have identical public suffixes for each version.
                local = h[:, :2, -SUFFIX:]
                local_cost = local.size + q_local.size; suffix_tokens += local_cost
                t = perf_counter()
                teach = compiled_branched_logits(teacher, jnp.asarray(local), jnp.asarray(q_local), *local_initial)[:, :, 0, :]
                if not bool(jnp.isfinite(teach).all()): raise FloatingPointError('nonfinite teacher logits')
                teacher_seconds += perf_counter() - t; positions = jnp.asarray(pos)
            history, questions, labels = map(jnp.asarray, (h, q, y))
            for arm in ARMS:
                t = perf_counter()
                values = np.asarray(_step(students[arm], optim[arm], history, questions, labels, full_initial, teach, positions,
                                          jnp.float32(KD_WEIGHT if arm == 'distill' else 0.)))
                seconds = perf_counter() - t; elapsed[arm] += seconds
                loss, ce, kl, correct, norm = map(float, values)
                if not np.isfinite(values).all(): raise FloatingPointError(f'nonfinite update: {arm}/s{seed}/{step+1}')
                wr.writerow((seed,data_seed,step+1,arm,signature,raw_cost,y.size,local_cost,int(correct),loss,ce,kl,norm,seconds))
                if arm == 'distill': accepted += int(correct)
            if (step+1) % 16 == 0:
                f.flush(); print(f'seed {seed}: completed {step+1}/{STEPS} paired updates',flush=True)
    tokenizer = {k:v for k,v in source.items() if k.startswith(('tokenizer_', 'retrieval_'))}
    with cost.open('x', encoding='utf8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n'); wr.writerow(('weight_seed','arm','parent_sha256','checkpoint_sha256','parameters','updates','student_tokens','answers','shared_teacher_tokens','teacher_correct_rows','student_seconds','shared_teacher_seconds','pair_elapsed_seconds','device','jax'))
        duration = perf_counter() - started
        for arm in ARMS:
            state = nnx.state(students[arm], nnx.Param)
            if int(optim[arm].step[...]) != STEPS or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state)):
                raise FloatingPointError(f'incomplete/nonfinite student: {arm}/s{seed}')
            save_model_safetensors(paths[arm], state, cfg, tokenizer_metadata=tokenizer, metadata={
                'binding_arm':arm, 'weight_seed':seed, 'parent_weights_sha256':source_sha, 'binding_steps':STEPS,
                'binding_data_seed':data_seed, 'binding_teacher_weight':KD_WEIGHT if arm=='distill' else 0,
            })
            wr.writerow((seed,arm,source_sha,digest(paths[arm]),parameter_count,STEPS,all_tokens,all_answers,suffix_tokens,
                         accepted,elapsed[arm],teacher_seconds,duration,jax.devices()[0].device_kind,jax.__version__))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--parents',type=Path,default=ROOT/'experiments/retention/parents')
    p.add_argument('--seed',type=int,choices=SEEDS,action='append')
    args = p.parse_args()
    print(f'JAX {jax.__version__}, device {jax.devices()[0].device_kind}',flush=True)
    for seed in args.seed or SEEDS: train_pair(args.output,args.parents,seed)
