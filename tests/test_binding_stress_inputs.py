import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_binding_state import panel
from probe_binding_stress import DATA_SEED, pattern


def test_two_complete_periods_match_the_reference_mixed_history():
    prefix,period,tail,q,y,new=pattern()
    reference={(name,gap):(h,questions,truth) for name,gap,_,h,questions,truth in panel(1,DATA_SEED,DATA_SEED+1,guard_study=True)}
    h,questions,truth=reference['mixed',1024]
    np.testing.assert_array_equal(np.concatenate((prefix[None],period[None],period[None],tail),axis=-1),h)
    np.testing.assert_array_equal(q,questions);np.testing.assert_array_equal(y,truth)
    np.testing.assert_array_equal(new,(truth!=reference['near',0][2])[0,0])
