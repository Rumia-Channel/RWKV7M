"""§22.18: can the same frozen reader infer a new route question from examples?"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from benchmark_address_screening import checkpoints, feature_readers, pack
from binding_state_memory import _read_fresh_queries
from probe_binding_distill import _digest
from probe_binding_stress import _advance_once
from route_transfer_panel import (COUNT, DATA_SEEDS, FAMILIES, KEY_BASE, MAX_STRIDE, NO_RULE,
                                  RULES, VALUE_BASE, infer_walk, make_panel, normalize_sources,
                                  questions, walk_tables)
from rwkv7m.model.state import init_rwkv_state, init_screen_state
from versioned_binding_panel import literal_reference

GAPS = (0, 1024)
METRICS = ('both_worlds','key_pairs','all_seven','families','abstentions','nonfinite','other_token_errors','no_rule_worlds')


def frozen_reads(codec, readers, prefix, examples, query, all_queries, need_edges):
    """One prefix per lane. Branch the state, not the label, for each demo or query."""
    lanes = len(prefix)
    model = codec.model
    state = init_rwkv_state(lanes, model.config), init_screen_state(lanes, model.config.screening)
    for start in range(0, prefix.shape[1], 64):
        state = _advance_once(model, jnp.asarray(prefix[:,start:start+64]), state)
    bank = {name: reader.advance(reader.start(lanes), prefix) for name, reader in readers.items()}
    estimates = {}
    if need_edges:
        estimates['ordinary'] = np.asarray(_read_fresh_queries(model, jnp.asarray(all_queries), state))
        estimates.update({name: r.query(bank[name], all_queries) for name, r in readers.items()})
    raw = defaultdict(list)
    for demo in examples:
        tokens = demo.reshape(lanes, -1)
        fresh = _advance_once(model, jnp.asarray(tokens), state)
        raw['ordinary'].append(np.asarray(_read_fresh_queries(model, jnp.asarray(query), fresh)))
        for name, reader in readers.items():
            raw[name].append(reader.query(reader.advance(bank[name], tokens), query))
    return {name: np.stack(rows) for name, rows in raw.items()}, estimates


def _known_rule(walks, rule, query):
    # Evaluation-only semantic oracle, independent of the generator's cycle order.
    keys = query[...,1] - KEY_BASE
    stride = {'next': 1, 'fourth': 4, 'previous': -1, 'conditional': 1 + keys%2}[rule]
    nodes = walks[MAX_STRIDE + stride, np.arange(len(query))[:,None], keys]
    return np.where(nodes < 0, -2, VALUE_BASE + nodes)


def _count(bucket, pred, gold, steps=None):
    both = (pred == gold).all(axis=0)
    bucket.update(both_worlds=int(both.sum()), key_pairs=both.size, all_seven=int(both.all()), families=1,
                  abstentions=int((pred == -2).sum()), nonfinite=int((pred == -1).sum()),
                  other_token_errors=int(((pred >= 0) & ((pred < VALUE_BASE) | (pred >= VALUE_BASE+COUNT))).sum()),
                  no_rule_worlds=0 if steps is None else int((steps == NO_RULE).sum()))


def _evaluate(codec, readers, seed, sha, draw, gap, panel, raw_writer, edge_writer, counts):
    known, alias, examples, query, gold = panel
    n, worlds, nq, _ = query.shape; lanes = n*worlds
    flat_query = query.reshape(lanes, nq, 3)
    flat_examples = examples.reshape(len(RULES), lanes, -1, 4)
    all_queries = questions(np.broadcast_to(np.arange(COUNT), (lanes, COUNT)))
    decoded = normalize_sources(alias)
    if not np.array_equal(decoded, known): raise ValueError('public legend did not reproduce the canonical prompt')
    one = literal_reference(decoded.reshape(lanes,-1), all_queries[...,1])
    oracle_walks = walk_tables(one)
    for i, rule in enumerate(RULES):
        if (not np.array_equal(_known_rule(oracle_walks, rule, flat_query).reshape(gold[i].shape), gold[i]) or
                not np.array_equal(_known_rule(oracle_walks, rule, flat_examples[i,...,:3]), flat_examples[i,...,3]) or
                not np.all(gold[i,:,0] != gold[i,:,1])):
            raise ValueError(f'cycle labels, visible edges or counterfactuals disagree: {rule}')
    # Proved identical above: the canonical run is also the exact normalized-input run.
    canonical, edges = frozen_reads(codec, readers, decoded.reshape(lanes,-1), flat_examples, flat_query, all_queries, True)
    novel, _ = frozen_reads(codec, readers, alias.reshape(lanes,-1), flat_examples, flat_query, all_queries, False)
    for name, edge in edges.items():
        walks = walk_tables(edge)
        for f in range(n):
            pair = slice(2*f,2*f+2)
            edge_writer.writerow((seed,sha,draw,f,gap,name,_digest(known[f],all_queries[pair]),
                                  pack(edge[2*f]),pack(edge[2*f+1]),pack(one[2*f]),pack(one[2*f+1])))
        for i, rule in enumerate(RULES):
            aided, step = infer_walk(walks, flat_examples[i], flat_query)
            for f in range(n):
                pair = slice(2*f,2*f+2); target = gold[i,f]; base = canonical[name][i,pair]
                aids = aided[pair]; steps = step[pair]
                demo = examples[i,f].reshape(2,-1)
                known_digest = _digest(np.concatenate((known[f],demo),axis=1),query[f])
                for form, unassisted in (('known',base), ('alias',novel[name][i,pair])):
                    digest = known_digest if form == 'known' else _digest(np.concatenate((alias[f],demo),axis=1),query[f])
                    key = (seed,gap,form,rule,name)
                    _count(counts[key+('raw',)],unassisted,target)
                    if form == 'alias': _count(counts[key+('legend',)],base,target)
                    _count(counts[key+('walk',)],aids,target,steps)
                    raw_writer.writerow((seed,sha,draw,f,gap,form,rule,name,digest,known_digest,pack(query[f,0,:,1]),
                                         *map(pack,target),*map(pack,unassisted),
                                         *(map(pack,base) if form=='alias' else ('','')),*map(pack,aids),*steps))


def run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    panels = [(draw,gap,make_panel(draw,gap)) for draw in DATA_SEEDS for gap in GAPS]
    counts = defaultdict(Counter)
    with (directory/'readouts.csv').open('x',encoding='utf8',newline='') as raw, (directory/'edges.csv').open('x',encoding='utf8',newline='') as edge:
        wr, ew = (csv.writer(f,lineterminator='\n') for f in (raw,edge))
        wr.writerow(('weight_seed','checkpoint_sha256','data_seed','family','gap','form','task','reader','input_sha256','normalized_sha256','keys',
                     'gold0','gold1','raw0','raw1','legend0','legend1','walk0','walk1','stride0','stride1'))
        ew.writerow(('weight_seed','checkpoint_sha256','data_seed','family','gap','reader','input_sha256',
                     'edges0','edges1','literal_edges0','literal_edges1'))
        for seed,path,sha in checkpoints():
            codec,readers = feature_readers(path)
            for draw,gap,panel in panels:
                _evaluate(codec,readers,seed,sha,draw,gap,panel,wr,ew,counts)
            raw.flush(); edge.flush(); print(f'weight {seed}: all five draws and both gap lengths complete', flush=True)
    with (directory/'scores.csv').open('x',encoding='utf8',newline='') as f:
        wr = csv.writer(f,lineterminator='\n'); wr.writerow(('weight_seed','gap','form','task','reader','aid',*METRICS))
        for key, values in sorted(counts.items()): wr.writerow((*key,*(values[m] for m in METRICS)))
    frozen = sorted({k[0] for k in counts}); names = sorted({k[4] for k in counts})
    with (directory/'gates.csv').open('x',encoding='utf8',newline='') as f:
        wr = csv.writer(f,lineterminator='\n'); wr.writerow(('reader','weights','conditions','minimum_raw_rate','all_reach_90pct'))
        for name in names:
            rows = [counts[s,g,'known',r,name,'raw'] for s in frozen for g in GAPS for r in ('fourth','previous')]
            if any(x['key_pairs'] != 7*FAMILIES*len(DATA_SEEDS) for x in rows): raise ValueError('incomplete transfer gate')
            passed = all(10*x['both_worlds'] >= 9*x['key_pairs'] for x in rows)
            wr.writerow((name,len(frozen),len(rows),min(x['both_worlds']/x['key_pairs'] for x in rows),passed))
            print(f'{name}: autonomous new-rule gate {passed}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('directory',type=Path)
    run(ap.parse_args().directory)
