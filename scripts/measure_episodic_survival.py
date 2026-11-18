"""Exact paired-answer/source survival in §22.21, not a long-stream speed test.

Writing a background at the shared query address sets both worlds to the same
payload; no later shared suffix can separate them. Count distinct harmful
forms (expectation) and their first actual occurrence (realized trajectory).
"""
import argparse
import csv
from pathlib import Path

import jax
import numpy as np

from benchmark_address_screening import checkpoints
from episodic_scene_data import CONFIRM_DRAWS, make_panel
from episodic_scene_features import WindowEncoder, public_sha
from probe_episodic_address import digest
from rwkv7m.io.episodic_address import COUNT_FORMAT, LEXICAL_FORMAT, load_head
from rwkv7m.model.episodic_address import empty_state, normalize, write_window
from rwkv7m.model.episodic_lexical import lexical_features, word_counts

LIVE, FORMS = 16, 8**3
HORIZONS = (0, 1024, 8192, 32768, 1_000_000, 10_000_000)  # physical tokens of background


def suffixes():
    """Enumerate the complete support via the actual generator, not a second grammar."""
    items = {}
    for draw in CONFIRM_DRAWS:
        gap = 4096
        while True:
            panel = make_panel('held', draw, 8, LIVE, gap)
            schedules = [np.unique(b[0, LIVE:], axis=0, return_index=True) for b in panel.blocks]
            if all(len(forms) == FORMS for forms, _ in schedules):
                break
            if gap >= 65536: raise ValueError('background grammar did not cover exactly 512 forms per family')
            gap *= 2
        for family, (forms, first) in enumerate(schedules):
            b = panel.blocks[family]; q = panel.queries[family]
            if not np.array_equal(b[0, LIVE:], b[1, LIVE:]): raise ValueError('suffix differs between worlds')
            items[draw, family] = (forms, first + 1, public_sha(b[:, :LIVE], q), public_sha(b[:, :LIVE+128], q))
    unique, inverse = np.unique(np.concatenate([v[0] for v in items.values()]), axis=0, return_inverse=True)
    return unique, {k:(ids, *row[1:]) for (k,row),ids in zip(items.items(),inverse.reshape(-1,FORMS),strict=True)}


@jax.jit
def writes(params, x):
    _, address, enabled = write_window(params, empty_state(len(x)), x)
    return address, enabled


def read_panel(path):
    with path.open(encoding='utf8', newline='') as file:
        rows = {}
        for r in csv.DictReader(file):
            if (r['arm'], r['surface'], r['live']) != ('factors', 'held', str(LIVE)): continue
            key = tuple(int(r[k]) for k in ('weight_seed','draw','family','question','gap_windows'))
            if key in rows: raise ValueError('duplicate comparison')
            rows[key] = r
    if len(rows) != 3*len(CONFIRM_DRAWS)*8*2*2: raise ValueError('incomplete near/far panel')
    return rows


def both(r, field):
    if field == 'source': return r['source0'] == r['source1'] == r['support']
    return r['pred0'] == r['gold0'] and r['pred1'] == r['gold1']


def measure(tokens, schedules, heads, panel, view):
    rows = read_panel(panel / 'readouts.csv'); hazards = []; scores = []
    weights = checkpoints() if view == 'hybrid' else ((s,None,'none') for s in (42,43,44))
    for seed, weight, weight_sha in weights:
        path = heads / f'factors-s{seed}.safetensors'; p,m,z,meta = load_head(path); head_sha = digest(path)
        fmt = LEXICAL_FORMAT if view == 'hybrid' else COUNT_FORMAT
        if (meta['arm'],meta['weight_seed'],meta['weight_sha'],meta['format'],meta['updates']) != ('factors',str(seed),weight_sha,fmt,'960'):
            raise ValueError('unregistered fit')
        features = lexical_features(WindowEncoder(weight).local(tokens),tokens) if view == 'hybrid' else word_counts(tokens)
        addr, active = map(np.asarray,writes(p, normalize(features,m,z)))
        sample = []
        for (draw,fam),(ids,first,near_sha,far_sha) in schedules.items():
            for q in (0,1):
                key = (seed,draw,fam,q); near,far = (rows[key+(g,)] for g in (0,128))
                if any(r['head_sha'] != head_sha or r['weight_sha'] != weight_sha or r['input_sha'] != sha
                       for r,sha in ((near,near_sha),(far,far_sha))): raise ValueError('head or public input mismatch')
                if (near['gold0'] == near['gold1'] or
                    any(near[k] != far[k] for k in ('gold0','gold1','support','read_address'))): raise ValueError('world/prefix mismatch')
                address = int(near['read_address']); destructive = active[ids] & (addr[ids] == address)
                n = int(destructive.sum()); first_hit = int(first[destructive].min()) if n else None
                intact = first_hit is None or first_hit > 128
                if intact:
                    if any(near[k]!=far[k] for k in ('pred0','pred1','source0','source1')): raise ValueError('unchanged slot disagrees with full replay')
                elif not (far['source0']==far['source1'] and int(far['source0'])>=LIVE and far['pred0']==far['pred1']):
                    raise ValueError('overwritten slot disagrees with full replay')
                a, s = both(near,'answer'), both(near,'source')
                if (intact and a, intact and s) != (both(far,'answer'), both(far,'source')): raise ValueError('128-window parity failed')
                sample.append((n,first_hit,a,s))
                hazards.append(dict(view=view,seed=seed,draw=draw,family=fam,question=q,head_sha=head_sha,near_input_sha=near_sha,
                                    far_input_sha=far_sha,read_address=address,near_answer=int(a),near_source=int(s),harmful_forms=n,
                                    possible_forms=FORMS,first_hit_window=first_hit,all_forms_observed_window=int(first.max())))
        for length in HORIZONS:
            g=length//8
            survived=[(hit is None or hit>g) for _,hit,_,_ in sample]
            prob=[(1.-n/FORMS)**g for n,_,_,_ in sample]
            scores.append(dict(view=view,seed=seed,background_tokens=length,questions=len(sample),
                               paired_answer=sum(ok*a for ok,(_,_,a,_) in zip(survived,sample,strict=True)),
                               paired_source=sum(ok*s for ok,(_,_,_,s) in zip(survived,sample,strict=True)),
                               expected_answer=sum(v*a for v,(_,_,a,_) in zip(prob,sample,strict=True)),
                               expected_source=sum(v*s for v,(_,_,_,s) in zip(prob,sample,strict=True))))
    return hazards,scores


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('hybrid_heads','hybrid_panel','count_heads','count_panel','output'): ap.add_argument(arg,type=Path)
    a=ap.parse_args(); tokens,schedules=suffixes(); hazards=[]; scores=[]
    for view,heads,panel in (('hybrid',a.hybrid_heads,a.hybrid_panel),('counts',a.count_heads,a.count_panel)):
        h,s=measure(tokens,schedules,heads,panel,view); hazards.extend(h); scores.extend(s)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (('hazards',hazards),('survival',scores)):
        with (a.output/f'{name}.csv').open('x',encoding='utf8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=tuple(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    print(f'{len(tokens)} distinct raw clauses; {len(hazards)} query/weight/view checks matched full near/far replay.')


if __name__=='__main__': main()
