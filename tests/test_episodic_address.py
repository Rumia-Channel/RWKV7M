import numpy as np
import jax
import jax.numpy as jnp
import pytest
from safetensors.numpy import save_file

from rwkv7m.io.episodic_address import COUNT_FORMAT, FORMAT, LEXICAL_FORMAT, load_head, save_head
from rwkv7m.model.episodic_address import (BITS,CLASSES,FEATURES,SLOTS,empty_state,hard_episode,
    init_head,read_queries,soft_episode,write_window)
from rwkv7m.model.episodic_lexical import lexical_features, word_counts


def fixed_policy():
    p=jax.tree.map(jnp.zeros_like,init_head(7)); p['input']=p['input'].at[:5,:5].set(jnp.eye(5))
    p['out_bias']=p['out_bias'].at[:2*BITS+1].set(-4.)
    for i,j in ((0,0),(1,BITS),(2,2*BITS),(3,2*BITS+1),(4,2*BITS+2)):
        p['out']=p['out'].at[i,j].set(4.)
    return p


def obs(address,gate,value):
    x=np.zeros(FEATURES,np.float32); x[0]=x[1]=3*address; x[2]=3*gate; x[3+value]=3; return x


def test_queries_read_different_cells_without_rewriting_them():
    p=fixed_policy(); a,b,noise,overwrite=(obs(0,1,1),obs(1,1,0),obs(0,0,0),obs(1,1,1))
    q=np.stack([a,b])[None]; state=empty_state(1)
    assert np.array_equal(read_queries(p,state,q)[0],[[-1,-1]])
    for x in (a,b,noise): state,_,_=write_window(p,state,x[None])
    np.testing.assert_array_equal(read_queries(p,state,q)[0],[[1,0]])
    np.testing.assert_array_equal(read_queries(p,state,q[:,::-1])[0],[[0,1]])
    updated,_,_=write_window(p,state,overwrite[None]); np.testing.assert_array_equal(read_queries(p,updated,q)[0],[[1,1]])
    np.testing.assert_array_equal(read_queries(p,state,q)[0],[[1,0]])
    assert state.bank.size==SLOTS*CLASSES and state.occupied.size==SLOTS
    batch=hard_episode(p,np.stack([a,b,noise,overwrite])[None],np.array([[1,1,1,0]],bool),q)
    np.testing.assert_array_equal(batch.answers,[[1,0]]); np.testing.assert_array_equal(batch.wrote,[[1,1,0,0]])


def test_soft_last_owner_matches_concentrated_hard_addresses_and_conserves_mass():
    p=fixed_policy(); x=np.stack([obs(0,1,1),obs(1,1,0),obs(0,1,0),obs(1,0,1)])[None]
    q=x[:,:2]; out=soft_episode(p,jnp.array(x),jnp.ones((1,4),bool),jnp.array(q),12.)
    np.testing.assert_allclose(out.owners,[[[0,0,1,0],[0,1,0,0]]],atol=2e-5)
    np.testing.assert_allclose(out.answers.sum(-1),1.,atol=1e-5); np.testing.assert_array_equal(out.answers.argmax(-1),[[0,0]])
    p=init_head(43)
    def ce(head):
        s=soft_episode(head,jnp.array(x),jnp.ones((1,4),bool),jnp.array(q),1.)
        return -jnp.log(s.answers[0,0,1])
    grad=np.asarray(jax.grad(ce)(p)['out']); assert np.isfinite(grad).all()
    for part in (grad[:,:BITS],grad[:,BITS:2*BITS],grad[:,2*BITS],grad[:,2*BITS+1:]): assert np.linalg.norm(part)>1e-7


def test_saved_one_probe_policy_retains_output_and_rejects_invalid_normalization(tmp_path):
    p=fixed_policy(); mean=np.zeros(FEATURES,np.float32); scale=mean+1; path=tmp_path/'head.safetensors'
    save_head(path,p,mean,scale,arm='source',weight_seed=42,weight_sha='b'*64,updates=960)
    pp,m,s,meta=load_head(path); np.testing.assert_array_equal(m,mean); np.testing.assert_array_equal(s,scale)
    x=np.stack([obs(0,1,1),obs(1,1,0)])[None]
    np.testing.assert_array_equal(hard_episode(pp,x,np.ones((1,2),bool),x).answers,[[1,0]])
    assert meta['arm']=='source' and meta['weight_sha']=='b'*64 and meta['format']==FORMAT
    broken=tmp_path/'wrong.safetensors'; save_file(dict(pp,mean=m,scale=-s),str(broken),metadata=meta)
    with pytest.raises(ValueError,match='scale'): load_head(broken)


def test_lexical_view_counts_repeats_and_shares_only_query_words_across_worlds(tmp_path):
    hidden=np.random.default_rng(6).normal(size=(1,2,2,256)).astype(np.float32)
    q=np.array([[[[34,34,34,20,20,3,3,3],[42,42,42,42,42,42,42,42]]]],np.int32)
    v=lexical_features(hidden,q)
    np.testing.assert_array_equal(v[...,:128],hidden[...,:128])
    np.testing.assert_array_equal(v[:,0,:,128:],v[:,1,:,128:])
    assert v[0,0,0,128+34]==3/8 and v[0,0,0,128+20]==2/8 and v[0,0,1,128+42]==1
    np.testing.assert_array_equal(v,lexical_features(hidden,q[...,::-1]))
    path=tmp_path/'lexical.safetensors'
    save_head(path,fixed_policy(),np.zeros(256,np.float32),np.ones(256,np.float32),
              arm='factors',weight_seed=43,weight_sha='e'*64,updates=960,lexical=True)
    assert load_head(path)[3]['format']==LEXICAL_FORMAT


def test_count_only_archive_has_no_backbone_and_reads_raw_words(tmp_path):
    p=init_head(42,counts_only=True); tokens=np.arange(32,dtype=np.int32).reshape(1,4,8)
    x=word_counts(tokens); valid=np.ones((1,4),bool)
    p['out_bias']=p['out_bias'].at[2*BITS].set(10.)
    before=hard_episode(p,x,valid,x[:,:2])
    path=tmp_path/'counts.safetensors'; mean=np.zeros(128,np.float32)
    save_head(path,p,mean,mean+1,arm='factors',weight_seed=42,weight_sha='none',counts_only=True,updates=960)
    pp,_,_,meta=load_head(path); after=hard_episode(pp,x,valid,x[:,:2])
    assert meta['format']==COUNT_FORMAT and meta['weight_sha']=='none'
    assert sum(v.size for v in pp.values())==17721
    np.testing.assert_array_equal(before.write_address,after.write_address)
    np.testing.assert_array_equal(before.answers,after.answers)
    # A 128-D head cannot be relabelled as a hybrid with an invented backbone.
    broken=tmp_path/'wrong.safetensors'
    save_file(dict(pp,mean=mean,scale=mean+1),str(broken),metadata={**meta,'format':LEXICAL_FORMAT,'weight_sha':'a'*64})
    with pytest.raises(ValueError,match='shape'): load_head(broken)
