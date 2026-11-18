import numpy as np
import jax
import jax.numpy as jnp
import pytest
from safetensors.numpy import save_file

from rwkv7m.io.episodic_address import load_head, save_head
from rwkv7m.model.episodic_address import (BITS,CLASSES,FEATURES,SLOTS,empty_state,hard_episode,
    init_head,read_queries,soft_episode,write_window)


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
    assert meta['arm']=='source' and meta['weight_sha']=='b'*64
    broken=tmp_path/'wrong.safetensors'; save_file(dict(pp,mean=m,scale=-s),str(broken),metadata=meta)
    with pytest.raises(ValueError,match='scale'): load_head(broken)
