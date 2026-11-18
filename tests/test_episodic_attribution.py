import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_episodic_address import FAIL, diagnose


def test_accidental_correct_classes_do_not_hide_address_or_overwrite_failures():
    # Same gold/prediction in four lanes. Source attribution distinguishes how it arrived.
    w=np.array([[2,5,6],[2,5,6],[2,2,6],[2,5,6],[2,5,6]],np.int32)
    enabled=np.ones_like(w,bool); enabled[0,0]=False
    r=np.array([[2],[5],[2],[2],[2]],np.int32)
    pred=np.array([[1],[1],[1],[7],[1]]); gold=np.ones_like(pred); pos=np.zeros_like(pred)
    owner,own,_,status=diagnose(w,enabled,r,pred,pos,gold)
    assert [FAIL[i] for i in status[:,0]]==['no_write','wrong_address','overwritten','bad_decode','fully_correct']
    np.testing.assert_array_equal(owner[:,0],[-1,1,1,0,0]); np.testing.assert_array_equal(own[:,0],[-1,0,1,0,0])
