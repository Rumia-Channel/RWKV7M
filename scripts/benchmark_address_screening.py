"""Run §22.17 frozen RWKV comparisons; run the scorer separately on their raw answers."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_binding_state import panel
from probe_binding_distill import _digest
from probe_binding_stress import DATA_SEED as STRESS_SEED, pattern

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/'experiments/binding-state/distill'
SEEDS=(42,43,44)
DIRECT,OVERWRITE,VERSIONED=732007,732008,732013
CASES={('near',0),('overwrite',32),('mixed',0),('mixed',1024),('change',1024),('filler',1024)}
FAMILIES=32
RAW=('seed','arm','checkpoint_sha256','case','depth','gap','data_seed','family','input_sha256','keys','focus','truth0','truth1','pred0','pred1')


def checkpoints():
    with (WEIGHTS/'panel/checkpoints.csv').open(encoding='utf8',newline='') as f:
        registered={int(r['weight_seed']):r['checkpoint_sha256'] for r in csv.DictReader(f) if r['arm']=='ce'}
    for s in SEEDS:
        p=WEIGHTS/f'ce-s{s}.safetensors'; digest=sha256(p.read_bytes()).hexdigest()
        if digest!=registered[s]: raise ValueError(f'changed frozen student: {p}')
        yield s,p,digest


def standard():
    for task,gap,draw,h,q,y in panel(FAMILIES,DIRECT,OVERWRITE,guard_study=True):
        if (task,gap) not in CASES: continue
        if task=='near': original=y
        focus=np.zeros_like(y[:,0],dtype=bool)
        if task=='mixed':
            focus=y[:,0]!=original[:,0]
            if not (np.array_equal(focus,y[:,1]!=original[:,1]) and np.all(focus.sum(-1)==4)): raise ValueError('not four shared genuine rewrites')
        yield task,0,gap,draw,h,q,y,focus


def pack(x): return ' '.join(map(str,map(int,x)))


def record(writer,case,seed,arm,weight,pred):
    task,depth,gap,draw,h,q,y,focus=case
    if pred.shape!=y.shape: raise ValueError('unequal truth and response shapes')
    for f in range(len(h)):
        writer.writerow((seed,arm,weight,task,depth,gap,draw,f,_digest(h[f],q[f]),pack(q[f,0,:,1]),pack(focus[f]),
                         pack(y[f,0]),pack(y[f,1]),pack(pred[f,0]),pack(pred[f,1])))


def references(output):
    from binding_state_memory import SnapshotReader
    cases=list(standard())
    with (output/'references.csv').open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(RAW)
        for seed,path,digest in checkpoints():
            reader=SnapshotReader(path,capacity=16)
            for case in cases:
                h,q=case[4:6]
                for arm,fn in (('ordinary',reader.predict_fresh),('replay',reader.predict_replayed)):
                    pred=np.concatenate([fn(h[i:i+8],q[i:i+8]) for i in range(0,len(h),8)])
                    record(wr,case,seed,arm,digest,pred)
            print(f'seed {seed}: full-history and keyed/replay references complete',flush=True)


def geometries(wr,seed,digest,reader,local,arm):
    geometry=('','','',0)
    if reader.q is not None:
        q,u=np.asarray(reader.q),np.asarray(reader.u);err=q.astype(np.float64)@u.astype(np.float64).T-np.eye(len(q))
        cross=err.copy();np.fill_diagonal(cross,0)
        geometry=float(np.linalg.cond(q)),float(np.max(np.abs(cross))),float(np.max(np.abs(np.diag(err)))),q.nbytes+u.nbytes
    init=reader.start(1); mem=init.memory
    wr.writerow((seed,arm,digest,int(np.sum(local==np.arange(32,48)[None,:])),local.size,*geometry[:3],
                 mem.matrix.nbytes,mem.bound.nbytes,init.previous.nbytes,reader.codec.payloads.nbytes,geometry[3]))


def stress(output,readers,digest,*,indexed=False):
    prefix,period,tail,q,truth,focus=pattern(); basis=sha256()
    for x in (prefix,period,tail,q): basis.update(x.astype('<i4',copy=False).tobytes())
    block=16_384; span=period.shape[1]; rolled=np.tile(period,(1,block//span+1))
    filename='register-stress.csv' if indexed else 'stress.csv'
    with (output/filename).open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(('seed','arm','checkpoint_sha256','data_seed','basis_sha256','physical_gap_tokens','key','rewritten','truth0','truth1','pred0','pred1','max_relative_hidden_error','seconds'))
        for arm,reader in readers.items():
            if (arm=='register')!=indexed: continue
            start=perf_counter(); state=reader.advance(reader.start(2),prefix); at=0
            for gap in (0,16_384,1_000_000,10_000_000):
                while at<gap:
                    n=min(block,gap-at); offset=at%span
                    state=reader.advance(state,rolled[:,offset:offset+n]); at+=n
                fork=reader.advance(state,tail[0]); pred=reader.query(fork,q[0]); seconds=perf_counter()-start
                # Scoring only, after the blind read: compare with the neural vector of the last literal.
                addresses=q[0,:,:,1]-16
                observed=np.asarray(reader.read_vectors(fork,addresses)[0],np.float64)
                exact=np.asarray(reader.codec.payloads)[addresses,truth[0]-32].astype(np.float64)
                error=(np.linalg.norm(observed-exact,axis=-1)/np.maximum(np.linalg.norm(exact,axis=-1),1e-12)).max(0)
                for k,key in enumerate(q[0,0,:,1]):
                    wr.writerow((44,arm,digest,STRESS_SEED,basis.hexdigest(),at,key,int(focus[k]),*truth[0,:,k],*pred[:,k],float(error[k]),seconds))
                hits=(pred==truth[0]).all(0);f.flush();print(f'{arm} physical gap {at}: new {hits[focus].sum()}/4, old {hits[~focus].sum()}/4; {seconds:.2f}s',flush=True)


def version_shortcuts(output,cases):
    """Can a reader recover the surviving operands from the prefix, or from STORE alone?"""
    from addressed_binding_tokens import COPY
    from rwkv7m.cli.prepare_retrieval import FILLER_TOKEN
    from versioned_binding_panel import PREFIX_TOKENS,literal_reference

    with (output/'version-shortcuts.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(('depth','gap','data_seed','family','input_sha256','route','copied_both_worlds','copied_keys','all_both_worlds','all_keys'))
        for task,depth,gap,draw,h,q,y,focus in cases:
            if task!='version': continue
            keys=q[...,1].reshape(-1,q.shape[2])
            for route,visible in (('prefix',h[:,:,:PREFIX_TOKENS]),('stores_without_copy',np.where(h==COPY,FILLER_TOKEN,h))):
                pred=literal_reference(visible.reshape(-1,visible.shape[-1]),keys).reshape(y.shape)
                hit=(pred==y).all(axis=1)
                for fam in range(len(h)):
                    writer.writerow((depth,gap,draw,fam,_digest(h[fam],q[fam]),route,int(hit[fam,focus[fam]].sum()),
                                     int(focus[fam].sum()),int(hit[fam].sum()),hit.shape[-1]))


def feature_readers(path):
    from addressed_rwkv_reader import AddressedReader,FrozenBindingCodec
    codec=FrozenBindingCodec(path)
    return codec,{a:AddressedReader(codec,a) for a in ('coupled','isolated','register')}


def routes(output,include_stress):
    from versioned_binding_panel import literal_reference,versioned_cases
    cases=list(standard())
    for depth,gap,h,q,y,focus in versioned_cases(seed=VERSIONED,families=FAMILIES,depths=(2,24),gaps=(0,1024)):
        cases.append(('version',depth,gap,VERSIONED,h,q,y,focus))
    tables=[]
    for task,depth,gap,draw,h,q,y,focus in cases:
        gold=literal_reference(h.reshape(-1,h.shape[-1]),q[...,1].reshape(-1,q.shape[2])).reshape(y.shape)
        if not np.array_equal(gold,y): raise ValueError(f'public literal interpreter disagrees with labels in {task}/{depth}/{gap}')
        tables.append(gold)
    version_shortcuts(output,cases)
    with (output/'routes.csv').open('x',encoding='utf8',newline='') as f, (output/'geometry.csv').open('x',encoding='utf8',newline='') as g:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(RAW); gw=csv.writer(g,lineterminator='\n');gw.writerow(('seed','arm','checkpoint_sha256','local_correct','local_pairs','condition','max_cross_write','max_self_error','matrix_bytes','bound_bytes','parser_bytes','literal_cache_bytes','address_code_bytes_logical'))
        for seed,path,digest in checkpoints():
            codec,readers=feature_readers(path)
            local=codec.local_predictions()
            for arm,reader in readers.items(): geometries(gw,seed,digest,reader,local,arm)
            for case,table in zip(cases,tables,strict=True):
                h,q=case[4:6]
                for arm,reader in readers.items(): record(wr,case,seed,arm,digest,reader.predict(h,q))
                record(wr,case,seed,'literal',digest,table)
            f.flush();g.flush();print(f'seed {seed}: standard and versioned panels complete',flush=True)
            if seed==44 and include_stress:
                stress(output,readers,digest)
                stress(output,readers,digest,indexed=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('output',type=Path)
    ap.add_argument('--phase',choices=('references','routes','stress','register-stress','all'),default='all'); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.phase in ('all','references'): references(a.output)
    if a.phase in ('all','routes'): routes(a.output,include_stress=a.phase=='all')
    if a.phase in ('stress','register-stress'):
        _,path,digest=next(row for row in checkpoints() if row[0]==44)
        _,readers=feature_readers(path)
        stress(a.output,readers,digest,indexed=a.phase=='register-stress')
