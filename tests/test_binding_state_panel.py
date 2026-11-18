from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return import_module('benchmark_binding_state'), import_module('score_binding_state')


def test_changed_distractor_values_and_true_overwrites(scripts):
    bench, _ = scripts
    by_case = {(t,g): (h,q,y) for t,g,_,h,q,y in bench.panel(2)}
    near, q, gold = by_case['near',0]
    for gap in (128,1024):
        repeated, _, _ = by_case['repeat',gap]; changing, cq, cy = by_case['change',gap]
        np.testing.assert_array_equal(changing[..., :63],near); np.testing.assert_array_equal(q,cq); np.testing.assert_array_equal(gold,cy)
        for i, worlds in enumerate(near):
            for w, prefix in enumerate(worlds):
                base = dict(zip(prefix[1::4],prefix[2::4],strict=True)); recent = dict(base); targets = set(q[i,w,:,1])
                for pos in range(63,63+gap,8):
                    marker,k,v = changing[i,w,pos:pos+3]
                    assert marker == 2 and k not in targets and v != recent[k]
                    assert tuple(repeated[i,w,pos:pos+3]) == (2,k,base[k])
                    recent[k] = v
                assert [recent[k] for k in q[i,w,:,1]] == gold[i,w].tolist()
    history, qs, labels = by_case['overwrite',32]
    for i, worlds in enumerate(history):
        for w, h in enumerate(worlds):
            writes = [(h[j+1],h[j+2]) for j in np.flatnonzero(h==2)]
            for k, y in zip(qs[i,w,:,1],labels[i,w],strict=True):
                stale, latest = [v for who,v in writes if who==k]; assert stale != latest == y


def test_joint_scores_require_same_worlds_and_same_families(scripts):
    _, s = scripts
    keys = tuple(range(16,24)); truth = (tuple(range(32,40)),tuple(range(33,41)))
    def partial(groups):
        # Only these entire four-key groups are correct in world 1. Invalid is never replaced with gold.
        return truth[0], tuple(x if k in groups else -1 for k,x in zip(keys,truth[1],strict=True))
    groups = tuple(s._programs(123,0,keys,2)); k0,k1 = (p.keys for p in groups)
    data = {(42,0,'near',0):(123,keys,dict(truth=truth,live=partial(k0),fifo=truth,keyed=truth)),
            (42,0,'change',1024):(123,keys,dict(truth=truth,live=partial(k1),fifo=truth,keyed=truth))}
    recall, maths = s.summarize(data)
    far = recall[42,'change',1024,'live']; assert (far['both_worlds'],far['near_and_far'],far['nonfinite']) == (4,0,4)
    ar = maths[42,'change',1024,2,'live']; assert (ar['program_pairs'],ar['both_outputs_both_worlds'],ar['near_and_far'],ar['invalid_worlds']) == (2,1,0,1)
    assert maths[42,'change',1024,2,'keyed']['near_and_far'] == 2


def test_mismatched_near_and_far_answers_fail_loudly(scripts):
    _, s = scripts
    keys = tuple(range(16,24)); y = (tuple(range(32,40)),tuple(range(33,41))); swapped = tuple(reversed(y))
    data = {(1,2,t,g):(5,keys,{arm: truth for arm in ('truth',*s.ARMS)}) for t,g,truth in [('near',0,y),('change',128,swapped)]}
    with pytest.raises(ValueError, match='changed operands'): s.summarize(data)
