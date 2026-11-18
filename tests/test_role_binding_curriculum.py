from itertools import product
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_role_binding import follow_card


def test_soft_final_loss_integrates_alternative_predicted_paths_not_teacher_forced_keys():
    rng=np.random.default_rng(29); p=rng.uniform(.1,2.,size=(2,10,8)).astype(np.float32); p/=p.sum(-1,keepdims=True)
    out=np.asarray(follow_card(jnp.asarray(p),3)); expected=np.zeros((2,2,8))
    for b,q,a,c in product(range(2),range(2),range(8),range(8)):
        expected[b,q]+=p[b,q,a]*p[b,2+a,c]*p[b,2+c]
    np.testing.assert_allclose(out,expected,rtol=3e-6,atol=1e-7)
    # Changing just an unchosen-but-possible intermediate object must affect the final objective.
    grad=jax.grad(lambda logits:-jnp.log(follow_card(jax.nn.softmax(logits,axis=-1),3)[0,0,6]))(jnp.log(p))
    assert np.linalg.norm(np.asarray(grad[0,2:]),axis=-1).min()>1e-5
    np.testing.assert_array_equal(follow_card(p[:,:2],1),p[:,:2])
