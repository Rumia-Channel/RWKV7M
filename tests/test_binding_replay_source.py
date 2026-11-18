from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


def test_archived_readout_cannot_be_reused_for_a_different_world_or_question(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    probe = import_module('probe_binding_replay')
    h = np.array([[[2,16,33,1],[2,16,39,1]]],np.int32)
    q = np.array([[[[3,16,4]],[[3,16,4]]]],np.int32)
    archive = {(42,'mixed',1024,5): {'guarded0':'36','guarded1':'-1','input_sha256':probe._digest(h[0],q[0])}}
    a = lambda hx,qx: probe._prior(archive,42,'mixed',1024,5,hx,qx)
    np.testing.assert_array_equal(a(h,q),[[[36],[-1]]])
    with pytest.raises(ValueError, match='different input'):
        a(h[:,::-1],q)
    altered = q.copy(); altered[0,1,0,1] = 17
    with pytest.raises(ValueError, match='different input'):
        a(h,altered)
