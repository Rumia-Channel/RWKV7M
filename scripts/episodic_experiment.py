"""Registered split/association checks for §22.20 feature files (not inference inputs)."""
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from episodic_scene_data import CONFIRM_DRAWS, DEV_DRAWS, TEST_CELLS, TEST_DRAWS, TRAIN_CELLS, TRAIN_DRAWS
from rwkv7m.model.episodic_address import CLASSES, FEATURES
from rwkv7m.model.episodic_lexical import lexical_features, word_counts


def load_cell(cache,split,weight,sha,surface,live,gap,*,lexical=False,counts_only=False):
    if split not in ('train','dev','test','confirm'): raise ValueError('unknown split')
    surfaces={'test':('seen','held','shift'),'confirm':('held',)}.get(split,('seen',))
    if surface not in surfaces or (live,gap) not in (TRAIN_CELLS if split in ('train','dev') else TEST_CELLS):
        raise ValueError('condition not registered')
    if counts_only:
        if lexical: raise ValueError('feature views are mutually exclusive')
        # This control uses raw texts, not the neural cache or any RWKV observations.
        from episodic_scene_features import _raw_cell, feature_table
        raw=_raw_cell(split,surface,live,gap)
        return feature_table(word_counts,raw.pop('raw_blocks'),raw.pop('raw_queries')) | raw
    path=Path(cache)/f'{split}-s{weight}-{surface}-k{live}-g{gap}.npz'
    with np.load(path,allow_pickle=False) as archive: d={name:archive[name] for name in archive.files}
    expected_meta={'split':split,'surface':surface,'live':live,'gap':gap,'format_version':1,'weight_seed':weight,'weight_sha':sha}
    if any(d[key].shape!=() or d[key].item()!=val for key,val in expected_meta.items()):
        raise ValueError(f'wrong condition, format or frozen RWKV association: {path}')
    draws={'train':TRAIN_DRAWS,'dev':DEV_DRAWS,'test':TEST_DRAWS,'confirm':CONFIRM_DRAWS}[split]; families=24 if split=='train' else 8
    rows=len(draws)*families; expected={(s,f) for s in draws for f in range(families)}
    if (set(zip(d['draw'].tolist(),d['family'].tolist(),strict=True))!=expected or
        d['input_sha'].shape!=(rows,) or any(not re.fullmatch('[0-9a-f]{64}',x) for x in d['input_sha'])):
        raise ValueError(f'bad public family or token hash manifest: {path}')
    for key,shape in {'blocks':(rows,2,live+gap),'queries':(rows,2),'support':(rows,2),'answers':(rows,2,2),
                      'draw':(rows,),'family':(rows,)}.items():
        if d[key].shape!=shape or d[key].dtype!=np.int32: raise ValueError(f'bad {key} in {path}')
    u=len(d['features'])
    if (not u or d['features'].shape!=(u,FEATURES) or d['ordinary'].shape!=(rows,2,2,FEATURES) or
        any(d[k].dtype!=np.float32 or not np.isfinite(d[k]).all() for k in ('features','ordinary'))):
        raise ValueError(f'bad neural observation in {path}')
    for key,upper in (('blocks',u),('queries',u),('support',live),('answers',CLASSES)):
        if np.any((d[key]<0)|(d[key]>=upper)): raise ValueError(f'{key} out of bounds in {path}')
    if (not np.array_equal(d['answers'][:,0],d['answers'][:,1,::-1]) or
        np.any(d['answers'][:,0,0]==d['answers'][:,1,0]) or np.any(d['support'][:,0]==d['support'][:,1])):
        raise ValueError(f'not a balanced two-scene swap in {path}')
    tokens=d['feature_tokens']
    if tokens.shape!=(u,8) or tokens.dtype!=np.int32 or np.any((tokens<0)|(tokens>=128)): raise ValueError(f'bad raw vocabulary table in {path}')
    for i,expected_sha in enumerate(d['input_sha']):
        checksum=sha256(tokens[d['blocks'][i]].astype('<i4',copy=False).tobytes())
        checksum.update(tokens[d['queries'][i]].astype('<i4',copy=False).tobytes())
        if checksum.hexdigest()!=expected_sha: raise ValueError(f'public history/query digest mismatch in {path}')
    if lexical:
        d['features']=lexical_features(d['features'],tokens)
        d['ordinary']=lexical_features(d['ordinary'],tokens[d['queries']][:,None])
    return d
