import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from rwkv7m.io.role_binding import FORMAT, VERSION, load_policy, save_policy
from rwkv7m.model import episodic_address
from rwkv7m.model.role_binding import (
    ARMS, Router, hard_marked, induce_router, init_policy, marked_read,
    marked_write, project, route, soft_marked,
)


def questions():
    return np.array([[30, a, 20, 31, b, 21, 32, 33]
                     for a in (10, 11, 12) for b in (10, 11, 12) if a != b], np.int32)


def test_induction_uses_shared_variable_predecessors_and_deterministic_pair_ranking():
    q = questions()
    router = induce_router(q.reshape(2, 3, 8))
    np.testing.assert_array_equal(router.markers, [20, 21])
    np.testing.assert_array_equal(router.lexicon, [10, 11, 12])
    assert router.markers.dtype == router.lexicon.dtype == np.int32
    reversed_router = induce_router(q[::-1])
    np.testing.assert_array_equal(reversed_router.markers, router.markers)
    np.testing.assert_array_equal(reversed_router.lexicon, router.lexicon)
    pairs = [(a, b) for a in (10, 11, 12) for b in (10, 11, 12) if a != b]
    # The lowest numeric pair (1,2) is less variable than (1,4).
    ranked = np.array([[a, 1, a, 2, a, 3, b, 4] for a, b in pairs], np.int32)
    np.testing.assert_array_equal(induce_router(ranked).markers, [1, 4])
    # Several equally informative pairs remain; numeric order breaks the tie.
    tied = np.array([[a, 20, b, 21, a, 22, b, 23] for a, b in pairs], np.int32)
    np.testing.assert_array_equal(induce_router(tied).markers, [20, 21])


@pytest.mark.parametrize('bad', [
    np.empty((0, 8), np.int32), questions()[:1], questions().astype(np.float32),
    np.zeros((2, 7), np.int32), np.full((2, 8), 128, np.int32),
])
def test_induction_rejects_missing_variable_marker_pair_or_nonraw_input(bad):
    with pytest.raises(ValueError):
        induce_router(bad)


def test_routes_share_addresses_across_phrase_permutations_without_role_aliases():
    q = questions()
    router = induce_router(q)
    expected = np.array([(a - 10) + 8 * (b - 10)
                         for a in (10, 11, 12) for b in (10, 11, 12) if a != b], np.int32)
    np.testing.assert_array_equal(route(router, q), expected)
    # Both phrases remain intact, but their surface order and all filler words change.
    permuted = np.array([[b, 21, 40, 41, a, 20, 42, 43]
                         for a in (10, 11, 12) for b in (10, 11, 12) if a != b], np.int32)
    np.testing.assert_array_equal(route(router, permuted), expected)
    reversed_roles = q.copy()
    reversed_roles[:, [2, 5]] = reversed_roles[:, [5, 2]]
    assert np.all(route(router, reversed_roles) != expected)
    assert route(router, q.reshape(2, 3, 8)).shape == (2, 3)
    assert route(router, q[0]).shape == ()
    assert route(router, q).dtype == np.int32
    assert route(router, np.empty((0, 8), np.int32)).shape == (0,)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'initial', 'unknown-low', 'unknown-high',
                                     'negative', 'nonpublic'])
def test_malformed_raw_phrases_abstain_instead_of_aliasing_a_live_address(mutation):
    router = induce_router(questions())
    malformed = questions()[0].copy()
    if mutation == 'duplicate':
        malformed[-1] = 20
    elif mutation == 'missing':
        malformed[2] = 34
    elif mutation == 'initial':
        malformed[[0, 2]] = malformed[[2, 0]]
    elif mutation == 'unknown-low':
        malformed[1] = 9
    elif mutation == 'unknown-high':
        malformed[4] = 13
    elif mutation == 'negative':
        malformed[-1] = -1
    else:
        malformed[-1] = 128
    routed = route(router, np.stack((questions()[0], malformed)))
    np.testing.assert_array_equal(routed, [8, -1])
    assert routed.dtype == np.int32


@pytest.mark.parametrize('arm,width,hidden,outputs', [
    ('ordinary', 256, 68, 8), ('bag', 128, 118, 21), ('learned', 256, 64, 21),
    ('marked-count', 128, 129, 9), ('marked-rwkv', 256, 67, 9),
])
def test_all_five_heads_have_exact_layout_and_seeded_scaling(arm, width, hidden, outputs):
    params = init_policy(42, arm)
    assert set(params) == {'input', 'bias', 'out', 'out_bias'}
    assert {key: value.shape for key, value in params.items()} == {
        'input': (width, hidden), 'bias': (hidden,), 'out': (hidden, outputs), 'out_bias': (outputs,)}
    rng = np.random.default_rng(42)
    np.testing.assert_array_equal(params['input'], rng.normal(0., width ** -.5, (width, hidden)).astype(np.float32))
    np.testing.assert_array_equal(params['out'], rng.normal(0., hidden ** -.5, (hidden, outputs)).astype(np.float32))
    np.testing.assert_array_equal(params['bias'], np.zeros(hidden, np.float32))
    np.testing.assert_array_equal(params['out_bias'], np.zeros(outputs, np.float32))
    if arm in ('ordinary', 'bag', 'learned'):
        original = episodic_address.init_head(42, ordinary=arm == 'ordinary', counts_only=arm == 'bag')
        for key in params:
            np.testing.assert_array_equal(params[key], original[key])
    x = jnp.ones((2, width), jnp.float32)
    np.testing.assert_array_equal(project(params, x), episodic_address.project(params, x))
    assert all(value.dtype == np.float32 for value in params.values())


def small_policy(width):
    params = {'input': jnp.zeros((width, 3), jnp.float32), 'bias': jnp.zeros(3, jnp.float32),
              'out': jnp.zeros((3, 9), jnp.float32), 'out_bias': jnp.zeros(9, jnp.float32)}
    params['input'] = params['input'].at[:3].set(jnp.eye(3))
    params['out'] = params['out'].at[0, 0].set(4.).at[1, 1].set(12.).at[2, 2].set(12.)
    params['out_bias'] = params['out_bias'].at[0].set(-4.)
    return params


def observation(width, enabled, value):
    x = np.zeros(width, np.float32)
    x[0] = 3 * enabled
    x[1 + value] = 3
    return x


@pytest.mark.parametrize('width', [3, 11])
def test_overwrites_abstentions_and_invalid_routes_match_concentrated_soft_and_streaming(width):
    params = small_policy(width)
    first = [(1, 0), (1, 1), (1, 1), (0, 0), (1, 0), (1, 0), (1, 0)]
    second = [(1, 1), (1, 0), (1, 0), (0, 1), (1, 1), (1, 1), (1, 1)]
    blocks = np.array([[observation(width, gate, value) for gate, value in lane]
                       for lane in (first, second)])
    valid = np.ones((2, 7), bool)
    valid[:, -1] = False
    write_addr = np.array([[17, 3, 17, 17, -1, 64, 3], [3, 17, 3, 3, -1, 64, 17]], np.int32)
    read_addr = np.array([[17, 3, 0, 9, -1, 64], [17, 3, 0, 9, -1, 64]], np.int32)
    hard = hard_marked(params, blocks, valid, write_addr, read_addr)
    expected = np.array([[1, 1, -1, -1, -1, -1], [0, 0, -1, -1, -1, -1]], np.int32)
    np.testing.assert_array_equal(hard.answers, expected)
    np.testing.assert_array_equal(hard.read_address, np.where(read_addr < 64, read_addr, -1))
    np.testing.assert_array_equal(hard.write_address, np.where(write_addr < 64, write_addr, -1))
    np.testing.assert_array_equal(hard.wrote, [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    state = episodic_address.empty_state(2)
    np.testing.assert_array_equal(marked_read(state, read_addr)[0], np.full((2, 6), -1))
    for t in range(blocks.shape[1]):
        state, address, enabled = marked_write(params, state, blocks[:, t], valid[:, t], write_addr[:, t])
        np.testing.assert_array_equal(address, hard.write_address[:, t])
        np.testing.assert_array_equal(enabled, hard.wrote[:, t])
    np.testing.assert_array_equal(marked_read(state, read_addr)[0], expected)
    np.testing.assert_array_equal(marked_read(state, read_addr[:, ::-1])[0], expected[:, ::-1])
    np.testing.assert_array_equal(state.occupied.sum(axis=-1), [2, 2])
    assert state.bank.shape == (2, 64, 8) and state.occupied.shape == (2, 64)
    soft = soft_marked(params, blocks, valid, write_addr, read_addr, 12.)
    owners = np.zeros((2, 6, 7), np.float32)
    owners[0, 0, 2] = owners[0, 1, 1] = owners[1, 0, 1] = owners[1, 1, 2] = 1
    np.testing.assert_allclose(soft.owners, owners, atol=1e-6)
    np.testing.assert_array_equal(soft.answers[:, :2].argmax(axis=-1), expected[:, :2])
    np.testing.assert_allclose(soft.answers[:, 2:], 1 / 8, atol=1e-6)
    np.testing.assert_allclose(soft.answers.sum(axis=-1), 1., atol=1e-6)


def test_soft_marginal_uses_strict_suffix_survival_and_uniform_empty_mass():
    params = jax.tree.map(jnp.zeros_like, small_policy(3))
    params['out_bias'] = params['out_bias'].at[-1].set(10.)
    blocks = np.zeros((1, 3, 3), np.float32)
    output = soft_marked(params, blocks, np.ones((1, 3), bool),
                         np.array([[4, 4, 5]], np.int32), np.array([[4, 5, 0, -1, 64]], np.int32), 2.)
    owners = np.array([[[.25, .5, 0], [0, 0, .5], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], np.float32)
    np.testing.assert_allclose(output.owners, owners, atol=1e-7)
    payload = np.exp(np.array([-10.] * 7 + [0.], np.float32))
    payload /= payload.sum()
    occupied = owners.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(output.answers, occupied * payload + (1 - occupied) / 8, atol=1e-7)


def test_marked_soft_training_differentiates_gate_and_payload_without_slot_targets():
    params = init_policy(43, 'marked-count')
    blocks = np.random.default_rng(8).normal(size=(1, 3, 128)).astype(np.float32)

    def loss(head):
        output = soft_marked(head, blocks, np.ones((1, 3), bool),
                             np.array([[0, 1, 0]], np.int32), np.array([[0, 1]], np.int32), 1.)
        return -jnp.log(output.answers[0, 0, 2]) - jnp.log(output.answers[0, 1, 5])

    gradient = jax.grad(loss)(params)
    assert all(np.isfinite(value).all() for value in gradient.values())
    assert np.linalg.norm(gradient['out'][:, 0]) > 0
    assert np.linalg.norm(gradient['out'][:, 1:]) > 0


def test_empty_passage_is_an_empty_bank_not_a_spurious_zero_address_write():
    params = small_policy(3)
    blocks = np.empty((1, 0, 3), np.float32)
    valid, writes = np.empty((1, 0), bool), np.empty((1, 0), np.int32)
    reads = np.array([[0, -1, 64]], np.int32)
    hard = hard_marked(params, blocks, valid, writes, reads)
    np.testing.assert_array_equal(hard.answers, [[-1, -1, -1]])
    assert hard.wrote.shape == hard.write_address.shape == (1, 0)
    soft = soft_marked(params, blocks, valid, writes, reads, 1.)
    assert soft.owners.shape == (1, 3, 0)
    np.testing.assert_allclose(soft.answers, 1 / 8)


def artifact(arm):
    params = init_policy(42, arm)
    width = params['input'].shape[0]
    tensors = {key: np.asarray(value) for key, value in params.items()}
    tensors.update(mean=np.zeros(width, np.float32), scale=np.ones(width, np.float32))
    meta = {'format': FORMAT, 'version': VERSION, 'arm': arm, 'seed': '42', 'updates': '12',
            'curriculum': 'mixed', 'weight_sha': 'none' if arm in ('bag', 'marked-count') else 'b' * 64}
    router = induce_router(questions()) if arm.startswith('marked-') else None
    if router is not None:
        tensors.update(router_markers=router.markers, router_lexicon=router.lexicon)
    return tensors, meta, router


@pytest.mark.parametrize('arm', ARMS)
def test_portable_policy_roundtrip_and_exclusive_creation(tmp_path, arm):
    tensors, metadata, router = artifact(arm)
    params = {key: value for key, value in tensors.items() if key in ('input', 'bias', 'out', 'out_bias')}
    path = tmp_path / f'{arm}.safetensors'
    save_policy(path, params, tensors['mean'], tensors['scale'], arm=arm, seed=42,
                weight_sha=metadata['weight_sha'], updates=12, router=router)
    loaded, mean, scale, meta, loaded_router = load_policy(path)
    assert meta == metadata and all(isinstance(value, str) for value in meta.values())
    assert set(loaded) == set(params)
    for key in params:
        np.testing.assert_array_equal(loaded[key], params[key])
        assert loaded[key].dtype == np.float32
    np.testing.assert_array_equal(mean, tensors['mean'])
    np.testing.assert_array_equal(scale, tensors['scale'])
    if router is None:
        assert loaded_router is None
    else:
        assert isinstance(loaded_router, Router)
        np.testing.assert_array_equal(loaded_router.markers, router.markers)
        np.testing.assert_array_equal(loaded_router.lexicon, router.lexicon)
        assert loaded_router.markers.dtype == loaded_router.lexicon.dtype == np.int32
        np.testing.assert_array_equal(route(loaded_router, questions()), route(router, questions()))
    with pytest.raises(FileExistsError):
        save_policy(path, params, mean, scale, arm=arm, seed=42, weight_sha=meta['weight_sha'],
                    updates=12, router=router)


@pytest.mark.parametrize('arm', ['marked-count', 'marked-rwkv'])
@pytest.mark.parametrize('missing', ['router_markers', 'router_lexicon', 'both'])
def test_marked_artifact_cannot_omit_router_tensors(tmp_path, arm, missing):
    tensors, meta, _ = artifact(arm)
    for key in ('router_markers', 'router_lexicon') if missing == 'both' else (missing,):
        tensors.pop(key)
    path = tmp_path / 'missing.safetensors'
    save_file(tensors, str(path), metadata=meta)
    with pytest.raises(ValueError, match='router'):
        load_policy(path)


@pytest.mark.parametrize('marked,unmarked', [('marked-count', 'bag'), ('marked-rwkv', 'ordinary')])
def test_marked_payload_head_cannot_be_relabelled_as_an_unmarked_policy(tmp_path, marked, unmarked):
    tensors, meta, _ = artifact(marked)
    tensors.pop('router_markers')
    tensors.pop('router_lexicon')
    path = tmp_path / 'relabelled.safetensors'
    save_file(tensors, str(path), metadata={**meta, 'arm': unmarked})
    with pytest.raises(ValueError, match='shape'):
        load_policy(path)


def test_router_is_required_exactly_for_marked_arms_on_save_and_load(tmp_path):
    router = induce_router(questions())
    for arm in ('ordinary', 'marked-count'):
        tensors, meta, _ = artifact(arm)
        params = {key: tensors[key] for key in ('input', 'bias', 'out', 'out_bias')}
        wrong_router = router if arm == 'ordinary' else None
        with pytest.raises(ValueError, match='router'):
            save_policy(tmp_path / f'{arm}.safetensors', params, tensors['mean'], tensors['scale'],
                        arm=arm, seed=42, weight_sha=meta['weight_sha'], updates=12, router=wrong_router)
    tensors, meta, _ = artifact('ordinary')
    tensors.update(router_markers=router.markers, router_lexicon=router.lexicon)
    path = tmp_path / 'unexpected.safetensors'
    save_file(tensors, str(path), metadata=meta)
    with pytest.raises(ValueError, match='router'):
        load_policy(path)


@pytest.mark.parametrize('markers,lexicon', [
    (np.array([20, 21], np.int64), np.array([10, 11], np.int32)),
    (np.array([20, 21], np.int32), np.array([10, 11], np.int64)),
    (np.array([21, 20], np.int32), np.array([10, 11], np.int32)),
    (np.array([20, 20], np.int32), np.array([10, 11], np.int32)),
    (np.array([20, 128], np.int32), np.array([10, 11], np.int32)),
    (np.array([20], np.int32), np.array([10, 11], np.int32)),
    (np.array([20, 21], np.int32), np.array([11, 10], np.int32)),
    (np.array([20, 21], np.int32), np.array([10, 10], np.int32)),
    (np.array([20, 21], np.int32), np.array([-1, 10], np.int32)),
    (np.array([20, 21], np.int32), np.array([10], np.int32)),
    (np.array([20, 21], np.int32), np.arange(9, dtype=np.int32)),
])
def test_invalid_router_dtype_order_range_and_cardinality_are_rejected(tmp_path, markers, lexicon):
    tensors, meta, _ = artifact('marked-count')
    params = {key: tensors[key] for key in ('input', 'bias', 'out', 'out_bias')}
    bad = Router(markers, lexicon)
    with pytest.raises(ValueError, match='router'):
        save_policy(tmp_path / 'save.safetensors', params, tensors['mean'], tensors['scale'],
                    arm='marked-count', seed=42, weight_sha='none', updates=12, router=bad)
    tensors.update(router_markers=markers, router_lexicon=lexicon)
    path = tmp_path / 'load.safetensors'
    save_file(tensors, str(path), metadata=meta)
    with pytest.raises(ValueError, match='router'):
        load_policy(path)


@pytest.mark.parametrize('arm', ARMS)
def test_sha_claim_must_match_whether_the_arm_uses_rwkv(tmp_path, arm):
    tensors, meta, router = artifact(arm)
    wrong_sha = 'c' * 64 if meta['weight_sha'] == 'none' else 'none'
    params = {key: tensors[key] for key in ('input', 'bias', 'out', 'out_bias')}
    with pytest.raises(ValueError, match='weight_sha'):
        save_policy(tmp_path / 'save.safetensors', params, tensors['mean'], tensors['scale'],
                    arm=arm, seed=42, weight_sha=wrong_sha, updates=12, router=router)
    path = tmp_path / 'load.safetensors'
    save_file(tensors, str(path), metadata={**meta, 'weight_sha': wrong_sha})
    with pytest.raises(ValueError, match='weight_sha'):
        load_policy(path)


@pytest.mark.parametrize('key,value', [
    ('format', 'rwkv7m.episodic_address'), ('version', '2'), ('arm', 'unrecognized'),
    ('seed', '-1'), ('seed', '42.0'), ('updates', '-1'), ('updates', '12.0'),
    ('curriculum', 'unregistered'),
    ('weight_sha', 'f' * 63), ('weight_sha', 'G' * 64), ('unexpected', 'label'),
])
def test_policy_rejects_invalid_metadata_and_format(tmp_path, key, value):
    tensors, meta, _ = artifact('learned')
    path = tmp_path / 'metadata.safetensors'
    save_file(tensors, str(path), metadata={**meta, key: value})
    with pytest.raises(ValueError):
        load_policy(path)


@pytest.mark.parametrize('kind', ['float64', 'nonfinite', 'shape', 'zero-scale', 'negative-scale', 'extra', 'missing'])
def test_policy_rejects_non_fp32_nonfinite_wrong_shape_or_nonpositive_scale(tmp_path, kind):
    tensors, meta, _ = artifact('marked-count')
    if kind == 'float64':
        tensors['input'] = tensors['input'].astype(np.float64)
    elif kind == 'nonfinite':
        tensors['mean'][0] = np.nan
    elif kind == 'shape':
        tensors['out_bias'] = np.zeros(8, np.float32)
    elif kind == 'zero-scale':
        tensors['scale'][0] = 0
    elif kind == 'negative-scale':
        tensors['scale'][0] = -1
    elif kind == 'extra':
        tensors['unused'] = np.zeros(1, np.float32)
    else:
        tensors.pop('bias')
    path = tmp_path / 'invalid.safetensors'
    save_file(tensors, str(path), metadata=meta)
    with pytest.raises(ValueError, match='shape|dtype|nonfinite|scale|tensor'):
        load_policy(path)
