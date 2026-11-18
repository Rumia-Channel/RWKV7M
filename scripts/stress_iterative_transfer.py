"""Two independent diagnostics for §22.19; never used for fitting or model selection."""
import argparse
import csv
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from addressed_binding_tokens import COUNT, KEY_BASE, VALUE_BASE
from addressed_rwkv_reader import AddressedReader, FrozenBindingCodec
from benchmark_address_screening import checkpoints
from iterative_read_frontend import _validated_arrays, head_to_edges, known_vocabulary, render_panel
from iterative_read_tasks import TEST_RULES, route_labels
from probe_iterative_reader import _load_test_cache, _pack, _sha
from route_transfer_panel import make_panel, questions
from rwkv7m.cli.prepare_retrieval import FILLER_TOKEN, STORE_TOKEN
from rwkv7m.io.iterative_reader import load_controller
from rwkv7m.model.iterative_reader import hard_read

# Fresh maps not used for fitting or the main panel. The same streams are used for all weights.
STREAM_SEEDS = (746101,746103,746107)
STREAM_GAPS = (0,1_000_000,10_000_000)
BLOCK_TOKENS = 16_384


def distinguishing_demos(query):
    """Choose by public names alone, before inspecting any demonstration answers.

    Both low bits must partition the triple nontrivially, differently from each
    other and differently from bits 2 and 3, even allowing complement/relabeling.
    """
    for combo in combinations(sorted(set(range(COUNT))-set(query)),3):
        src = np.asarray(combo,np.int32)
        bits = (src[None,:] >> np.arange(4)[:,None]) & 1
        if all(b.any() and (1-b).any() and all((b != other).any() and (b == other).any()
                   for j,other in enumerate(bits) if j != i) for i,b in enumerate(bits[:2])):
            return src
    return None


def _bit_answers(truth,src,bit):
    # Gold side only. Changing the single predicate is the intervention.
    return np.where((src >> bit) & 1, route_labels(truth,src,(2,2)), route_labels(truth,src,(1,1)))


def _head(directory,seed,digest):
    path = directory/f'isolated-s{seed}.safetensors'
    p, meta = load_controller(path)
    if any(meta[k] != v for k,v in {'frontend':'isolated','feedback':'true','weight_seed':str(seed),
                                      'weight_sha':digest,'steps':'640','cap':'6'}.items()):
        raise ValueError('expected registered full-budget isolated head')
    return p,_sha(path)


def predicate(cache,heads,output):
    fields = ('weight_seed','weight_sha','head_sha','draw','family','input_sha','bit','eligible','examples',
              'pairs','paired_correct','disagree_pairs','disagree_correct','all_seven','answer0','answer1','prediction0','prediction1')
    with (output/'switch-bit.csv').open('x',encoding='utf8',newline='') as stream:
        wr = csv.DictWriter(stream,fields,lineterminator='\n'); wr.writeheader()
        for seed,_,digest in checkpoints():
            data = _load_test_cache(cache,seed,digest); p,head_sha = _head(heads,seed,digest)
            keep = np.where(data['gap']==1024)[0]; sources = []; selected = []
            def base(i):
                return {'weight_seed':seed,'weight_sha':digest,'head_sha':head_sha,'draw':int(data['draw'][i]),
                        'family':int(data['family'][i]),'input_sha':data['digest'][i]}
            for i in keep:
                demo = distinguishing_demos(data['query'][i,0])
                if demo is None:
                    for bit in (0,1): wr.writerow({**base(i),'bit':bit,'eligible':0})
                else:
                    sources.append(demo); selected.append(i)
            if not selected: raise ValueError('no graph has identifiable low-bit demonstrations')
            demos = np.repeat(np.stack(sources)[:,None,:],2,axis=1)
            q,truth,edges = (data[name][selected] for name in ('query','truth','isolated'))
            disagree = ((q[:,0] ^ (q[:,0] >> 1)) & 1).astype(bool)
            for bit in (0,1):
                dst = _bit_answers(truth,demos,bit)
                out = hard_read(p,edges.reshape(-1,COUNT),demos.reshape(-1,3),dst.reshape(-1,3),q.reshape(-1,7))
                pred = np.asarray(out.answer).reshape(q.shape); answer = _bit_answers(truth,q,bit)
                pair = (pred==answer).all(axis=1)
                for j,i in enumerate(selected):
                    wr.writerow({**base(i),'bit':bit,'eligible':1,'examples':_pack(sources[j]),'pairs':7,
                        'paired_correct':int(pair[j].sum()),'disagree_pairs':int(disagree[j].sum()),
                        'disagree_correct':int((pair[j]&disagree[j]).sum()),'all_seven':int(pair[j].all()),
                        'answer0':_pack(answer[j,0]),'answer1':_pack(answer[j,1]),
                        'prediction0':_pack(pred[j,0]),'prediction1':_pack(pred[j,1])})


def _changed_stream(reader, p, draw):
    panel = make_panel(draw,0,families=1); public = render_panel(panel,heldout=True)
    truth,query,src = (a[0] for a in _validated_arrays(panel,public))
    raw = public.history[0]; legend_end = 4*COUNT; prefix_end = legend_end+3*COUNT
    legend, prefix, tail = raw[:,:legend_end],raw[:,:prefix_end],raw[:,prefix_end:]
    normalized = known_vocabulary(raw)
    demo_names = tail[0,1::3]; demo_nodes = normalized[0,prefix_end+1::3] - KEY_BASE
    if len(set(demo_names))!=9 or np.isin(demo_nodes,query).any(): raise ValueError('distractors touch a target station')
    rng = np.random.default_rng(np.random.SeedSequence((draw,913)))
    hashes = [sha256(row.astype('<i4',copy=False).tobytes()) for row in prefix]
    state = reader.advance(reader.start(2),normalized[:,:prefix_end]); at = 0; clock = perf_counter()
    for gap in STREAM_GAPS:
        while at < gap:
            n = min(BLOCK_TOKENS,gap-at); ids = rng.integers(9,size=(n//8,2),dtype=np.int32)
            events = np.full((len(ids),8),FILLER_TOKEN,np.int32)
            events[:,0] = STORE_TOKEN; events[:,1] = demo_names[ids[:,0]]; events[:,2] = VALUE_BASE+demo_nodes[ids[:,1]]
            raw_block = events.reshape(-1); batch = np.broadcast_to(raw_block,(2,n))
            for h in hashes: h.update(raw_block.astype('<i4',copy=False).tobytes())
            # Translate the observed aliases from the actual public legend, not the generator's graph.
            tokens = known_vocabulary(np.concatenate((legend,batch),axis=1))[:,legend_end:]
            state = reader.advance(state,tokens); at += n
        fork = reader.advance(state,normalized[:,prefix_end:]); question = questions(np.broadcast_to(np.arange(COUNT,dtype=np.int32),(2,COUNT)))
        edges = head_to_edges(reader.query(fork,question)); elapsed = perf_counter()-clock
        src_hash=[]
        for h,row in zip(hashes,tail,strict=True):
            current=h.copy(); current.update(row.astype('<i4',copy=False).tobytes()); src_hash.append(current.hexdigest())
        for task,pair in TEST_RULES.items():
            visible = route_labels(truth,src,pair)
            out = hard_read(p,edges,src,visible,query)
            pred = np.asarray(out.answer); answer = route_labels(truth,query,pair)
            yield (draw,gap,gap//8,*src_hash,task,int((edges==truth).sum()),edges.size,
                   int((pred==answer).all(axis=0).sum()),query.shape[1],int(out.reads.sum()),int(out.timed_out.sum()),
                   _pack(answer[0]),_pack(answer[1]),_pack(pred[0]),_pack(pred[1]),elapsed)


def distance(heads,output):
    with (output/'physical-stream.csv').open('x',encoding='utf8',newline='') as stream:
        wr = csv.writer(stream,lineterminator='\n'); wr.writerow(('weight_seed','weight_sha','head_sha','map_seed',
            'gap','physical_writes_per_world','input_sha0','input_sha1','task','primitive_correct','primitive_total',
            'both_worlds','key_pairs','read_total','timeouts','answer0','answer1','prediction0','prediction1','elapsed_seconds'))
        for seed,path,digest in checkpoints():
            p,head_sha = _head(heads,seed,digest); reader = AddressedReader(FrozenBindingCodec(path),'isolated')
            for draw in STREAM_SEEDS:
                for row in _changed_stream(reader,p,draw): wr.writerow((seed,digest,head_sha,*row))
                stream.flush()


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('study',choices=('predicate','distance'))
    ap.add_argument('cache',type=Path); ap.add_argument('heads',type=Path); ap.add_argument('output',type=Path)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if a.study=='predicate': predicate(a.cache,a.heads,a.output)
    else: distance(a.heads,a.output)
