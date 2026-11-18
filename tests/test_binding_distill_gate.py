import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_binding_distill_gate import CASES, FAMILIES, SEEDS, gate


def _panel():
    return {(s,a,t,g,k):dict(weight_seed=s,arm=a,task=t,gap=g,subset=k,key_pairs=n,both_worlds=n//2,nonfinite=0)
            for s in SEEDS for a in ('parent','ce','distill') for t,g in CASES
            for k in (('all','old','rewritten') if t=='mixed' else ('all',))
            for n in [FAMILIES*(2 if t=='overwrite' else 8 if k=='all' else 4)]}


def test_one_large_recovery_cannot_hide_other_parents_or_near_regressions():
    r=_panel(); r[44,'distill','mixed',1024,'rewritten']['both_worlds'] += 25
    def adopted(): return list(gate(r.values()))[-1][-1]
    assert not adopted()  # +25/384, but only one positive seed.
    r[43,'distill','mixed',1024,'rewritten']['both_worlds'] += 1
    assert adopted()
    r[42,'distill','overwrite',32,'all']['both_worlds'] -= 2  # 2/64 > tolerance.
    assert not adopted()


def test_incomplete_panel_cannot_return_a_decision():
    r=_panel(); r.pop((42,'ce','near',0,'all'))
    with pytest.raises(ValueError,match='missing'): list(gate(r.values()))
