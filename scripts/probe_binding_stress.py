"""One registered counterexample search: never skip a cycle, reset state, or feed a label."""
import argparse
import csv
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from benchmark_binding_state import _mixed_history
from probe_binding_distill import ARMS, DEFAULT_PARENTS, _published_parents, resolve_checkpoint
from probe_retrieval_distance import BLOCK, histories
from retrieval_paired_readout import compiled_branched_logits
from rwkv7m.io import load_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state

DATA_SEED, WEIGHT_SEED = 731987, 44
GAPS = (0, 16_384, 1_000_000)


def pattern():
    prefix, block, q, original = histories(DATA_SEED,1,'other-writes')
    period = np.tile(block[:,None,:], (1,8,1))
    for t in range(8): period[:,t,2::8] = np.roll(block[:,2::8], (t+1)%8, axis=1)
    mixed, truth = _mixed_history(prefix[None],q,DATA_SEED)
    rewrites = truth != original
    if not np.array_equal(rewrites[0,0],rewrites[0,1]) or rewrites[0,0].sum() != 4:
        raise ValueError('stress probe requires four genuinely rewritten keys in both worlds')
    return prefix, period.reshape(2,8*BLOCK), mixed[:,:,prefix.shape[1]:], q, truth, rewrites[0,0]


def _advance(model,tokens,state):
    _,rwkv,screen,_=model.compute_recurrent_hidden(tokens,*state,phase='read_write',deterministic=True)
    return rwkv,screen


_advance_once = nnx.jit(_advance)


@nnx.jit
def _periods(model,tokens,state,count):
    return jax.lax.fori_loop(0,count,lambda _,s:_advance(model,tokens,s),state)


def run(checkpoints,output,arm,parents):
    path, weights, _, _ = resolve_checkpoint(arm,WEIGHT_SEED,checkpoints,parents,_published_parents())
    params,cfg,_ = load_model_safetensors(path); cfg=replace(cfg,remat_blocks=False)
    model=load_linen_params_into_nnx(initialize_nnx_model(jax.random.key(0),cfg),params)
    prefix,period,tail,q,truth,rewritten=pattern(); basis=sha256()
    for x in (prefix,period,tail,q): basis.update(x.astype('<i4',copy=False).tobytes())
    initial=(init_rwkv_state(2,cfg),init_screen_state(2,cfg.screening)); prefix,period,tail,q=map(jnp.asarray,(prefix,period,tail,q))
    now=0; start=perf_counter(); state=_advance_once(model,prefix,initial)
    output.mkdir(parents=True,exist_ok=True)
    with (output/f'stress-{arm}.csv').open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n'); wr.writerow(('arm','weight_seed','checkpoint_sha256','data_seed','basis_sha256','physical_gap_tokens','key','rewritten','truth0','truth1','pred0','pred1','elapsed_seconds'))
        for gap in GAPS:
            # All intermediate milestones before the last must end at a period boundary.
            if now % period.shape[1]: raise ValueError('unaligned previous milestone')
            count,rest=divmod(gap-now,period.shape[1]); now=gap
            if count: state=_periods(model,period,state,jnp.int32(count))
            if rest: state=_advance_once(model,period[:,:rest],state)
            # The tail and independent questions are disposable branches, not part of the gap carry.
            logits=np.asarray(compiled_branched_logits(model,tail,q,*state))[0]
            pred=np.where(np.isfinite(logits).all(-1),logits.argmax(-1),-1); elapsed=perf_counter()-start
            for k,key in enumerate(np.asarray(q[0,0,:,1])):
                wr.writerow((arm,WEIGHT_SEED,weights,DATA_SEED,basis.hexdigest(),gap,int(key),int(rewritten[k]),*truth[0,:,k],*pred[:,k],elapsed))
            f.flush(); hit=(pred==truth[0]).all(0)
            print(f'{arm} gap={gap}: rewritten {hit[rewritten].sum()}/4, old {hit[~rewritten].sum()}/4; nonfinite {(pred==-1).sum()}/16; {elapsed:.2f}s',flush=True)


def literal_control(output):
    """Check answerability on the same physical streams; the explicit-key reader stores 16 bytes."""
    from retrieval_frontier import STORE
    from score_binding_state import _literal_read

    prefix,period,tail,q,truth,rewritten=pattern()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'stress-literal.csv').open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n'); wr.writerow(('data_seed','physical_gap_tokens','key','rewritten','truth0','truth1','pred0','pred1'))
        for gap in GAPS:
            count,rest=divmod(gap,period.shape[-1]); n=prefix.shape[-1]
            history=np.empty(n+gap+tail.shape[-1],np.int32); pred=[]
            for world in (0,1):
                history[:n]=prefix[world]
                history[n:n+gap-rest].reshape(count,period.shape[-1])[:]=period[world]
                history[n+gap-rest:n+gap]=period[world,:rest]
                history[n+gap:]=tail[0,world]
                pred.append(_literal_read(history,q[0,world,:,1],STORE))
            pred=np.asarray(pred); hit=(pred==truth[0]).all(0)
            for k,key in enumerate(q[0,0,:,1]):
                wr.writerow((DATA_SEED,gap,int(key),int(rewritten[k]),*truth[0,:,k],*pred[:,k]))
            print(f'literal gap={gap}: rewritten {hit[rewritten].sum()}/4, old {hit[~rewritten].sum()}/4',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoints',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--parents',type=Path,default=DEFAULT_PARENTS);p.add_argument('--arm',choices=(*ARMS,'literal'),action='append')
    args=p.parse_args()
    for a in args.arm or ARMS:
        if a=='literal': literal_control(args.output)
        else: run(args.checkpoints,args.output,a,args.parents)
