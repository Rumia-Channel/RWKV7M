import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from addressed_binding_tokens import parse_chunk


def test_commands_commit_on_the_third_token_and_keep_lane_boundaries():
    # In lane 1 COPY is illegal (literal source); only its later STORE may commit.
    raw=np.array([[1,2,16,35,5,18,16,1],[5,18,42,1,1,2,19,47]],np.int32)
    def events(cuts):
        tail=np.full((2,2),-1,np.int32); rows=[[],[]]; start=0
        for stop in cuts:
            tail,x=parse_chunk(tail,raw[:,start:stop]); start=stop
            for b in range(2):rows[b].extend(tuple(r) for r in x[b] if r[0])
        return rows
    expected=[[(1,0,3),(2,2,0)],[(1,3,15)]]
    assert events([8])==events([1,2,3,4,5,6,7,8])==events([3,3,6,8])==expected


def test_frozen_rwkv_reads_the_copied_version_after_source_replacement():
    from addressed_rwkv_reader import AddressedReader, FrozenBindingCodec

    path=Path(__file__).resolve().parents[1]/'experiments/binding-state/distill/ce-s42.safetensors'
    codec=FrozenBindingCodec(path)
    reader=AddressedReader(codec,'isolated')
    # Isolate retention from unrelated local errors of the real 128-way head.
    values=np.arange(32,48)
    before,after=values[codec.local_predictions()[0]==values][:2]
    h=np.array([[2,16,before,5,18,16,2,16,after]],np.int32)
    state=reader.start(1)
    for chunk in np.split(h,[1,5,8],axis=1):
        state=reader.advance(state,chunk)
    questions=np.array([[[3,18,4],[3,16,4],[3,19,4]]],np.int32)
    np.testing.assert_array_equal(reader.query(state,questions),[[before,after,-2]])
    state=reader.advance(state,np.array([[5,18,19]],np.int32))
    np.testing.assert_array_equal(reader.query(state,questions),[[-2,after,-2]])
