"""Probe recent overwrites with the same frozen RWKV, with and without its long-prefix state."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np

from benchmark_binding_state import GUARD_DIRECT_SEED, GUARD_OVERWRITE_SEED, ROOT, _recent_flags, panel
from binding_state_memory import SnapshotReader


SPLITS = {'published': (GUARD_DIRECT_SEED, GUARD_OVERWRITE_SEED), 'independent': (731947, 731948)}
CASES = {('near',0), ('overwrite',32), ('mixed',0), ('mixed',1024), ('change',1024), ('filler',1024)}
PUBLISHED = ROOT / 'experiments/binding-state/guarded'
CAPACITY = 16
FAMILIES = 16
WEIGHTS = (42,43,44)
ARMS = ('truth','guard','replay','fresh','tail')


def _digest(h, q):
    digest = sha256(h.astype('<i4',copy=False).tobytes()); digest.update(q.astype('<i4',copy=False).tobytes())
    return digest.hexdigest()


def _pack(x):
    return ' '.join(map(str, x))


def _prior(archive, seed, task, gap, first, history, queries):
    """Archived *predictions*, not labels; bind each row to its complete paired public input."""
    preds = np.empty(queries.shape[:-1], np.int32)
    for i in range(len(history)):
        row = archive[seed,task,gap,first+i]
        if row['input_sha256'] != _digest(history[i],queries[i]):
            raise ValueError('published guard readout belongs to a different input')
        for w in (0,1):
            preds[i,w,:] = tuple(map(int,row[f'guarded{w}'].split()))
    return preds


def _readers():
    with (PUBLISHED/'weights.csv').open(encoding='utf-8',newline='') as f:
        hashes = {int(r['weight_seed']):r['checkpoint_sha256'] for r in csv.DictReader(f)}
    for seed in WEIGHTS:
        path = ROOT/f'experiments/retention/parents/long-s{seed}.safetensors'
        if sha256(path.read_bytes()).hexdigest() != hashes[seed]:
            raise ValueError(f'changed frozen checkpoint: {path}')
        yield seed, SnapshotReader(path,capacity=CAPACITY)


def _archive():
    with (PUBLISHED/'readouts.csv').open(encoding='utf-8',newline='') as f:
        return {(int(r['weight_seed']),r['task'],int(r['gap']),int(r['family'])):r for r in csv.DictReader(f)}


def matched_length_control(output,batch):
    """Keep elapsed tokens, original bindings, late writes, questions and labels; replace only interference."""
    cases = {(t,g):(h,q,y) for t,g,_,h,q,y in panel(FAMILIES,*SPLITS['published'],guard_study=True)
             if t=='mixed' or (t,g)==('filler',1024)}
    far,q,truth = cases['mixed',1024]
    close,cq,cy = cases['mixed',0]
    fill,fq,_ = cases['filler',1024]
    replaced = np.concatenate((fill,far[...,-24:]),axis=-1)
    prefix_len = close.shape[-1]-24
    if (far.shape != replaced.shape or far.shape[-1]-close.shape[-1] != 1024
            or not np.array_equal(far[...,:prefix_len],replaced[...,:prefix_len])
            or not np.array_equal(close[...,-24:],far[...,-24:]) or not np.array_equal(cy,truth)
            or not np.array_equal(q,cq) or not np.array_equal(q,fq)):
        raise ValueError('unmatched public histories in the length control')
    changed = _recent_flags(far,q,CAPACITY).astype(bool)
    archive = _archive()
    output.parent.mkdir(exist_ok=True,parents=True)
    with output.open('x',encoding='utf-8',newline='') as f:
        w = csv.writer(f,lineterminator='\n')
        w.writerow(('weight_seed','data_seed','family','key','truth0','truth1','near0','near1','filler0','filler1','interference0','interference1','filler_input_sha256'))
        for seed,reader in _readers():
            near = _prior(archive,seed,'mixed',0,0,close,q)
            interference = _prior(archive,seed,'mixed',1024,0,far,q)
            totals = np.zeros(3,np.int32)
            for first in range(0,len(far),batch):
                predictions = reader.predict_guarded(replaced[first:first+batch],q[first:first+batch])
                for i,new in enumerate(predictions):
                    k = first+i
                    fingerprint = _digest(replaced[k],q[k])
                    for j in np.flatnonzero(changed[k]):
                        target = truth[k,:,j]; arms = near[k,:,j], new[:,j], interference[k,:,j]
                        totals += [np.array_equal(p,target) for p in arms]
                        w.writerow((seed,GUARD_DIRECT_SEED,k,q[k,0,j,1],*target,*arms[0],*arms[1],*arms[2],fingerprint))
            print(f'{seed}: no gap/filler/changed writes = {totals.tolist()} / {changed.sum()} pairs',flush=True)


def measure(output, split, batch):
    from binding_state_memory import REPLAY_TOKENS, replay_mask
    if batch < 1:
        raise ValueError('batch must be positive')
    direct, overwrite = SPLITS[split]
    archive = _archive() if split=='published' else None
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(('weight_seed','task','gap','data_seed','family','keys','recent','hit',
                         *(f'{arm}{w}' for arm in ARMS for w in (0,1)), 'input_sha256'))
        for seed,reader in _readers():
            tails = {}; native_queries = 0
            for task,gap,draw,h,q,truth in panel(FAMILIES,direct,overwrite,guard_study=True):
                if (task,gap) not in CASES:
                    continue
                recent = _recent_flags(h,q,CAPACITY)
                for first in range(0,len(h),batch):
                    hx = h[first:first+batch]; qx = q[first:first+batch]
                    # No expected value or condition label reaches the two inference routes.
                    guard = (reader.predict_guarded(hx,qx) if archive is None
                             else _prior(archive,seed,task,gap,first,hx,qx))
                    hit = replay_mask(hx,qx,CAPACITY)
                    if not np.array_equal(hit[:,0],hit[:,1]):
                        raise ValueError('paired worlds have different local write chronology')
                    fresh = reader.predict_fresh(hx[..., -REPLAY_TOKENS:],qx)
                    replay = np.where(hit,fresh,guard)
                    if archive is None and ((hit.any() and not hit.all()) or (first==0 and task in ('overwrite','near'))):
                        # Compare every mixed-route batch and a short/no-hit & all-local control
                        # with the deployable one-head implementation, not just offline rerouting.
                        deployed = reader.predict_replayed(hx,qx)
                        if not np.array_equal(deployed,replay):
                            raise ValueError('standalone replay disagrees with independently evaluated branches')
                        replay = deployed; native_queries += replay.size
                    tail = None
                    if task == 'mixed':
                        suffix = hx[...,-24:]
                        sig = _digest(suffix,qx)
                        if gap == 0:
                            tail = reader.predict_fresh(suffix,qx)
                            tails[first] = sig, tail
                        else:
                            expected, tail = tails[first]
                            if sig != expected: raise ValueError('the compared mixed write suffixes or queries differ')
                    for i in range(len(hx)):
                        k = first+i
                        preds = (truth[k],guard[i],replay[i],fresh[i])
                        writer.writerow((seed,task,gap,draw,k,_pack(qx[i,0,:,1]),_pack(recent[k]),_pack(hit[i,0].astype(int)),
                                         *(_pack(world) for arm in preds for world in arm),
                                         *((_pack(tail[i,0]),_pack(tail[i,1])) if tail is not None else ('','')),
                                         _digest(hx[i],qx[i])))
                f.flush()
            print(f'{split}: parent {seed} complete; {native_queries} standalone answers cross-checked',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path,help='new raw readout CSV')
    p.add_argument('--split',choices=(*SPLITS,'length-control'),default='independent')
    p.add_argument('--batch',type=int,default=4)
    a = p.parse_args()
    if a.batch < 1: p.error('batch must be positive')
    if a.split=='length-control': matched_length_control(a.output,a.batch)
    else: measure(a.output,a.split,a.batch)
