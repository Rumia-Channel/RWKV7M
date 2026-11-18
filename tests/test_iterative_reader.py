import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m.model.iterative_reader import hard_read, init_controller, soft_read


def calibrated_controller():
    # This exact policy isolates recurrent and halt semantics from optimizer luck.
    p = jax.tree.map(jnp.zeros_like, init_controller(1))
    p['field_base'] = jnp.array([24.,24.])
    p['relevance'] = jnp.array([9.,0.,0.,0.])
    p['emit_linear'] = jnp.array([20.,0.,0.,0.]); p['emit_bias'] = jnp.array([-10.])
    return p


def test_questions_halt_at_different_rounds_and_finished_answers_do_not_drift():
    p = calibrated_controller()
    edges = np.roll(np.arange(16,dtype=np.int32),-1)[None]
    src = np.array([[0,1,2]],np.int32); dst = np.array([[1,3,3]],np.int32)
    q = np.array([[4,5]],np.int32)
    result = hard_read(p,edges,src,dst,q,cap=6)
    np.testing.assert_array_equal(result.answer,[[5,7]])
    np.testing.assert_array_equal(result.reads,[[1,2]])
    assert int(result.rounds) == 2 and not np.asarray(result.timed_out).any()
    short = hard_read(p,edges,src,dst,q,cap=1)
    np.testing.assert_array_equal(short.answer,[[5,-2]])
    np.testing.assert_array_equal(short.timed_out,[[False,True]])
    reset = hard_read(p,edges,src,dst,q,feedback=False)
    np.testing.assert_array_equal(reset.answer,[[5,-2]])
    np.testing.assert_array_equal(reset.reads,[[1,6]])


def test_changing_only_a_later_edge_changes_the_answer():
    p = calibrated_controller()
    cycle = np.array([1,2,3,4,5,6,7,0,9,10,11,12,13,14,15,8],np.int32)
    edges = np.stack([cycle,cycle]); edges[1,[9,10]] = edges[1,[10,9]]
    src = np.array([[0,1,2]]*2,np.int32); dst = src+2; q = np.array([[8],[8]],np.int32)
    result = hard_read(p,edges,src,dst,q)
    np.testing.assert_array_equal(result.answer,[[10],[11]])
    np.testing.assert_array_equal(result.reads,[[2],[2]])
    masked = hard_read(p,edges,src,np.full_like(dst,16),q)
    assert np.all(np.asarray(masked.timed_out)) and np.all(np.asarray(masked.answer) == -2)


def test_endpoint_gradient_reaches_address_relevance_and_halt_and_mass_is_conserved():
    p = init_controller(43)
    edges = jnp.array([[1,2,3,4,5,6,7,0,9,10,11,12,13,14,15,8]],jnp.int32)
    src = jnp.array([[0,1,2]]); dst = jnp.array([[1,3,3]]); q = jnp.array([[8,9]])
    def cost(params):
        o = soft_read(params,edges,src,dst,q)
        return -jnp.log(o.mass[0,0,9]*o.mass[0,1,11])
    grad = jax.grad(cost)(p)
    for field in ('field_base','field_context','relevance','emit_linear','emit_out'):
        assert np.isfinite(grad[field]).all() and np.linalg.norm(grad[field]) > 1e-9
    o = soft_read(p,edges.at[0,9].set(16),src,dst,q)
    np.testing.assert_allclose(o.mass.sum(-1)+o.timeout_mass,1.,rtol=2e-6,atol=2e-6)


def test_no_heldout_mapping_is_a_training_mapping_modulo_either_cycle_length():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from iterative_read_tasks import SPLITS, TEST_RULES, TRAIN_PAIRS
    for n in (7,9):
        taught = {tuple(x % n for x in pair) for pair in TRAIN_PAIRS}
        assert all(tuple(x%n for x in pair) not in taught for name,pair in TEST_RULES.items() if name!='next')
    seeds = [set(draws) for draws,_,_ in SPLITS.values()]
    assert all(not pair.intersection(other) for i,pair in enumerate(seeds) for other in seeds[:i])
