"""Train §22.19 controllers; this program never opens the test split."""
import argparse
import csv
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from benchmark_address_screening import checkpoints
from iterative_read_tasks import SPLITS, TRAIN_PAIRS, route_labels
from rwkv7m.io.iterative_reader import save_controller
from rwkv7m.model.iterative_reader import hard_read, init_controller, soft_read

STEPS, BATCH, LOG_EVERY = 640, 32, 160
LR, READ_COST = 0.03, 0.002
ARMS = {'isolated': ('isolated', True), 'reset': ('isolated', False), 'ordinary': ('ordinary', True)}
OPT = optax.chain(optax.clip_by_global_norm(1.), optax.adam(LR))


class Pool(NamedTuple):
    edges: dict
    demo_src: jax.Array
    demo_dst: jax.Array       # [training operation, world, demo]
    query: jax.Array
    answer: jax.Array         # [training operation, world, question]


def load_pool(cache_dir, split, seed, sha):
    if split not in ('train','dev'): raise ValueError('trainer accepts only train/dev')
    draws, families, gaps = SPLITS[split]
    with np.load(cache_dir / f'{split}-s{seed}.npz', allow_pickle=False) as src:
        if str(src['weight_sha'].item()) != sha or int(src['weight_seed'].item()) != seed:
            raise ValueError('primitive RWKV digest or weight seed mismatch')
        actual = list(zip(src['draw'].tolist(),src['family'].tolist(),src['gap'].tolist(),strict=True))
        expected = {(d,f,g) for d in draws for f in range(families) for g in gaps}
        if set(actual) != expected or len(actual) != len(expected): raise ValueError(f'incomplete {split} cache')
        def flat(name):
            a = src[name]
            if a.dtype != np.int32 or a.ndim != 3 or a.shape[:2] != (len(expected),2):
                raise ValueError(f'bad {name} dimensions or dtype')
            return a.reshape(-1,a.shape[-1])
        # Keep the literal map on the label side. It is not a member of Pool.
        truth = flat('truth'); demo = flat('demo'); query = flat('query')
        if not np.all(np.sort(truth,axis=-1) == np.arange(16)): raise ValueError('broken route generator')
        demo_dst = np.stack([route_labels(truth,demo,r) for r in TRAIN_PAIRS])
        gold = np.stack([route_labels(truth,query,r) for r in TRAIN_PAIRS])
        edges = {name: jnp.asarray(flat(name)) for name in ('ordinary','isolated')}
    return Pool(edges,jnp.asarray(demo),jnp.asarray(demo_dst),jnp.asarray(query),jnp.asarray(gold))


def objective(params, edges, demo_src, demo_dst, query, answer, feedback):
    result = soft_read(params,edges,demo_src,demo_dst,query,feedback=feedback)
    hit = jnp.take_along_axis(result.mass,answer[...,None],axis=-1)[...,0]
    # A small uniform floor keeps the matched no-feedback arm finite even on unreachable targets.
    nll = -jnp.mean(jnp.log((1.-1e-6)*hit + 1e-6 / 16))
    return nll + READ_COST*result.expected_reads.mean(), (nll,result.expected_reads.mean())


@partial(jax.jit, static_argnames=('feedback',))
def update_block(params, opt_state, edges, demo, demos_out, query, gold, worlds, ops, *, feedback):
    def step(carry, indices):
        p, state = carry; w, op = indices
        (_, aux), grad = jax.value_and_grad(objective,has_aux=True)(
            p,edges[w],demo[w],demos_out[op,w],query[w],gold[op,w],feedback)
        updates, state = OPT.update(grad,state,p)
        return (optax.apply_updates(p,updates),state), jnp.stack(aux)
    (params,opt_state), metrics = jax.lax.scan(step,(params,opt_state),(worlds,ops))
    return params,opt_state,metrics.mean(0)


def dev_score(params, pool, frontend, feedback):
    per_op = []; timeouts = 0
    for i in range(len(TRAIN_PAIRS)):
        out = hard_read(params,pool.edges[frontend],pool.demo_src,pool.demo_dst[i],pool.query,feedback=feedback)
        hit = np.asarray(out.answer == pool.answer[i]); per_op.append(hit.reshape(-1,2,hit.shape[-1]).all(1).mean())
        timeouts += int(out.timed_out.sum())
    return float(np.mean(per_op)),float(np.min(per_op)),timeouts


def train_weight(cache_dir, directory, seed, sha, arms):
    train = load_pool(cache_dir,'train',seed,sha); dev = load_pool(cache_dir,'dev',seed,sha)
    rng = np.random.default_rng(744017)
    worlds = jnp.asarray(rng.integers(len(train.query), size=(STEPS,BATCH), dtype=np.int32))
    ops = jnp.asarray(rng.integers(len(TRAIN_PAIRS), size=(STEPS,BATCH), dtype=np.int32))
    for arm in arms:
        frontend, feedback = ARMS[arm]
        params = init_controller(seed); opt_state = OPT.init(params)
        count = sum(int(v.size) for v in params.values())
        trained = 0.
        with (directory/f'learning-{arm}-s{seed}.csv').open('x',encoding='utf8',newline='') as f:
            wr = csv.writer(f,lineterminator='\n'); wr.writerow(('arm','weight_seed','step','head_parameters','endpoint_nll','mean_soft_reads','dev_seen_rate','dev_seen_worst_rule','dev_timeouts','training_seconds'))
            for start in range(0,STEPS,LOG_EVERY):
                stop = start + LOG_EVERY; clock = perf_counter()
                params,opt_state,metrics = update_block(params,opt_state,train.edges[frontend],train.demo_src,train.demo_dst,
                    train.query,train.answer,worlds[start:stop],ops[start:stop],feedback=feedback)
                nll,reads = map(float,np.asarray(metrics)); trained += perf_counter()-clock
                if not np.isfinite(nll+reads): raise FloatingPointError(f'nonfinite training objective in {arm}')
                dev_rate,dev_min,dev_timeouts = dev_score(params,dev,frontend,feedback)
                wr.writerow((arm,seed,stop,count,nll,reads,dev_rate,dev_min,dev_timeouts,trained)); f.flush()
        save_controller(directory/f'{arm}-s{seed}.safetensors',params,frontend=frontend,feedback=feedback,
                        weight_seed=seed,weight_sha=sha,steps=STEPS,cap=6)
        print(f'weight {seed} {arm}: final seen-rule dev {dev_rate:.3f}, worst {dev_min:.3f}, {count} trainable parameters',flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('cache',type=Path); ap.add_argument('directory',type=Path)
    ap.add_argument('--seed',type=int,choices=(42,43,44)); ap.add_argument('--arm',choices=tuple(ARMS)); args = ap.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    for s,_,digest in checkpoints():
        if args.seed is None or s == args.seed: train_weight(args.cache,args.directory,s,digest,(args.arm,) if args.arm else ARMS)
