import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bootstrap_binding_distill import FAMILIES, WEIGHT_SEEDS, intervals


def test_identical_checkpoints_do_not_triple_the_number_of_independent_families():
    rows={}; truth=((32,33,34,35),(36,37,38,39)); flags=((1,1,1,1),)*2
    for seed in WEIGHT_SEEDS:
        for family in range(FAMILIES):
            # Every checkpoint recovers the very same half of the families in CE; distill adds nothing.
            for arm in ('parent','ce','distill'):
                pred=truth if arm!='parent' or family<FAMILIES//2 else (truth[0],truth[0])
                rows[arm,seed,family,'mixed',1024]={'truth':truth,'pred':pred,'flags':flags}
    effect,null=list(intervals(rows))
    assert effect[4]==.5 and effect[5]==0 and effect[8]==.25
    assert effect[6]<=.35 and effect[7]>=.65  # 32 independent shared families, not 384 independent keys.
    assert null[4:]==(0.,0.,0.,0.,1.)
