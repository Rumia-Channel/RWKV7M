"""A suffix teacher must not erase a changed final write or supervise an old key."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_binding_distill import distill_loss


def test_soft_gradient_repeats_across_nuisance_not_across_values_or_keys():
    # Write position changes between families. Version 1 in family 1 has a wrong teacher.
    pos = jnp.array([0, 1]); logits = jnp.zeros((2, 4, 2, 5))
    truth = jnp.zeros((2, 4, 2), dtype=jnp.int32).at[0, :, 0].set(jnp.array([1, 2, 1, 2])).at[1, :, 1].set(jnp.array([3, 4, 3, 4]))
    teachers = jax.nn.one_hot(jnp.array([[1,2],[3,0]]),5) * 4

    def gradient(weight):
        return jax.grad(lambda x: distill_loss(x,truth,teachers,pos,weight)[0])(logits)

    extra = np.asarray(gradient(.5) - gradient(0.0))
    assert int(distill_loss(logits,truth,teachers,pos,.5)[1][2]) == 6
    np.testing.assert_array_equal(extra[:, :2], extra[:, 2:])
    assert extra[0,0,0,1] < 0 < extra[0,1,0,1]
    assert extra[1,0,1,3] < 0
    assert not extra[0, :, 1].any() and not extra[1, :, 0].any()
    assert not extra[1, 1::2, 1].any()


def test_wrong_or_absent_teacher_leaves_ordinary_supervised_loss():
    logits = jnp.array([[[[1.,0.,-1.],[0.,3.,1.]]]*4]); truth = jnp.ones((1,4,2),dtype=jnp.int32)
    bad = jnp.broadcast_to(jnp.array([2.,1.,0.]),(1,2,3))
    student_loss, (_, kl, count) = distill_loss(logits,truth,bad,jnp.array([0]),.5)
    anchor_loss = distill_loss(logits,truth,None,None,.5)[0]
    assert float(student_loss) == pytest.approx(float(anchor_loss)) and int(count) == 0 and float(kl) == 0.
