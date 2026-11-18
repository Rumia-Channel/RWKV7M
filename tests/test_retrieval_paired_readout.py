from dataclasses import replace
from importlib import import_module
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rwkv7m import load_model_safetensors
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.state import init_rwkv_state, init_screen_state


@pytest.fixture
def inputs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    histories = import_module('probe_retrieval_distance').histories
    score = import_module('retrieval_paired_readout').paired_final_logits
    p, _, q, _ = histories(3450, 1, 'filler')
    prefix = np.repeat(p.reshape(1, 2, 1, -1), 2, axis=2)
    return root, score, np.concatenate([prefix, q[:, :, :2]], axis=-1)


def test_forked_queries_match_independent_world_and_key_rollouts(inputs):
    root, score, ids = inputs
    params, cfg, _ = load_model_safetensors(root / 'experiments/retention/parents/long-s42.safetensors')
    cfg = replace(cfg, dtype='float32')
    model = initialize_nnx_model(jax.random.key(0), cfg)
    load_linen_params_into_nnx(model, params)

    def initial(n):
        return init_rwkv_state(n, cfg), init_screen_state(n, cfg.screening)

    @nnx.jit
    def full(model, x, r, s):
        hidden, *_ = model.compute_recurrent_hidden(x, r, s, phase='read_write', deterministic=True)
        return model.compute_logits(hidden[:, -1:, :])[:, 0, :]

    ref = full(model, jnp.asarray(ids.reshape(4, -1)), *initial(4))
    actual = score(model, ids, *initial(2)).reshape(ref.shape)
    np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual.argmax(-1), ref.argmax(-1))


def test_conflicting_query_prefix_is_not_silently_dropped(inputs):
    _, score, ids = inputs
    ids[0, 0, 1, 2] += 1  # Same question family, different stored value.
    with pytest.raises(ValueError, match='share a prefix'):
        score(None, ids, (), ())
