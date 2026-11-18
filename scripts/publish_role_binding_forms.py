"""Publish answer-free school test forms, a separate answer key, and a split-overlap audit."""
import argparse
import csv
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

import role_binding_data as data
from role_binding_exam import ADV_DRAWS, MID_DRAWS, make_exam, public_document, render_exam


def public_json(document):
    return json.dumps(document,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def features(t):
    """Analyze published lexemes only. This oracle has no path into the fitted policies."""
    where={mark:t.index(mark) for mark in (data.GA,data.NI,data.WO)}
    a,b=(data.NAME_TOKENS.index(t[where[m]-1]) for m in (data.GA,data.NI))
    return tuple(sorted((a,b))),tuple(sorted(where,key=where.__getitem__))


def audit(output):
    groups={'basic':set(),'advanced':set(),'midterm':set()}; graded={k:set() for k in groups}; ids={k:set() for k in groups}
    def add(key,p,draw):
        ids[key].update((draw,i) for i in range(len(p.blocks)))
        public=(p.blocks,p.queries,p.cards) if hasattr(p,'cards') else (p.blocks,p.queries)
        groups[key].update(map(tuple,np.unique(np.concatenate([x.reshape(-1,8) for x in public]),axis=0).tolist()))
        f,w=p.blocks.shape[:2]; support=p.support.reshape(f,w,-1)
        source=p.blocks[np.arange(f)[:,None,None],np.arange(w)[None,:,None],support]
        graded[key].update(features(tuple(t))[0] for t in np.unique(source.reshape(-1,8),axis=0).tolist())
    for draw in data.TRAIN_DRAWS:
        for k,g in data.TRAIN_CELLS: add('basic',data.make_panel('train',draw,24,k,g),draw)
    for draw in ADV_DRAWS:
        for g in (0,8): add('advanced',make_exam('advanced',draw,16,g),draw)
    for draw in MID_DRAWS:
        for g in (0,128): add('midterm',make_exam('midterm',draw,8,g),draw)
    train=groups['basic'] | groups['advanced']; held={(a,b) for a in range(8) for b in range(a+1,8) if b-a in (2,6)}
    rows=[]
    for key,tokens in groups.items():
        pairs,orders=map(set,zip(*(features(t) for t in tokens),strict=True)); targets=graded[key]
        if key!='midterm' and pairs & held: raise ValueError('held pair in training, including ungraded text or printed card')
        if key=='midterm' and (not targets<=held or tokens & train): raise ValueError('official test shares a graded pair or an exact utterance with training')
        rows.append(dict(form=key,reads={'basic':1,'advanced':3,'midterm':2}[key],families=len(ids[key]),unique_utterances=len(tokens),
                         shared_utterances_with_train=len(tokens & train),phrase_orders=len(orders),public_pairs=len(pairs),
                         held_pairs_in_public_text=len(pairs & held),graded_pairs=len(targets),held_graded_pairs=len(targets & held)))
    with (output/'split-audit.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,tuple(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
    return rows


def publish(output):
    output.mkdir(parents=True,exist_ok=True); report=audit(output)
    bp=data.make_panel('train',data.TRAIN_DRAWS[0],1,4,0)
    basic=dict(blocks=bp.blocks[0,0].tolist(),queries=bp.queries[0].tolist(),cards=[],reads=1)
    basic_text='次の記録を読んで、二つの問いに物の名前で答えなさい。\n'+'\n'.join(data.gloss(b,japanese=True) for b in bp.blocks[0,0])+ '\n'+'\n'.join(data.gloss(q,japanese=True) for q in bp.queries[0])
    samples=[('basic',basic,basic_text)]
    for name,draw in (('advanced',ADV_DRAWS[0]),('midterm',MID_DRAWS[0])):
        p=make_exam(name,draw,1,0); samples.append((name,public_document(p,0,0),render_exam(p,0,0,name)))
    with (output/'problems.ja.md').open('x',encoding='utf8',newline='\n') as text:
        text.write('# 問題用紙の例（すべてオリジナル）\n\n各形式の先頭のデータseed・家族0・世界0。最初の2枚は学習用、最後は本番の中テスト。モデルには公開JSONのトークン列と照合表を渡す。\n')
        for (tag,doc,problem),title in zip(samples,('基礎','入試風の応用','中テスト'),strict=True):
            with (output/f'{tag}.json.example').open('x',encoding='utf8',newline='\n') as f: f.write(public_json(doc)+'\n')
            text.write(f'\n## {title}\n\n'+problem.strip()+'\n')
    with (output/'midterm-public.csv.gz').open('xb') as out, gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as gz, \
         io.TextIOWrapper(gz,encoding='utf8',newline='') as pub, (output/'midterm-key.csv').open('x',encoding='utf8',newline='') as key:
        w=csv.writer(pub,lineterminator='\n'); k=csv.writer(key,lineterminator='\n'); fields=('draw','family','world','background_tokens','document_sha256')
        w.writerow((*fields,'public_document')); k.writerow((*fields,'question','first_object','final_object','first_record','final_record'))
        count=0
        for draw in MID_DRAWS:
            for gap in (0,128):
                p=make_exam('midterm',draw,8,gap)
                for fam in range(8):
                    for world in range(4):
                        document=public_json(public_document(p,fam,world)); signature=sha256(document.encode()).hexdigest(); idx=(draw,fam,world,gap*8,signature)
                        w.writerow((*idx,document)); count+=1
                        for q in (0,1): k.writerow((*idx,q,*p.values[fam,world,q],*p.support[fam,world,q]))
    print(f'{count} answer-free papers; {2*count} answers in separate key; split audit: {report}')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('output',type=Path); publish(ap.parse_args().output)
