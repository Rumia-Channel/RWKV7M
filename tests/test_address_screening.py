"""Behavioural tests for the address screened matrix delta memory.

The memory is a numeric primitive: it keeps payload vectors under learned
address codes and never sees tokens, labels or scores.  The tests pin the
address geometry, the delta rule, the blind copy semantics and the no-op
guarantees.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rwkv7m.model.address_screening import (
    ADDRESS_MODES,
    EVENT_COPY,
    EVENT_LITERAL,
    EVENT_NOP,
    AddressMemory,
    apply_events,
    apply_events_jit,
    initial_memory,
    make_address_codes,
    read_memory,
    read_memory_jit,
)


KEY_COUNT = 4
VALUE_DIM = 3
PAYLOAD_COUNT = 3


def key_embeddings():
    """Key geometry with two near-parallel keys (non-orthogonal features)."""
    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((KEY_COUNT, 8))
    embeddings[2] = embeddings[2] + 0.9 * embeddings[0]
    return embeddings


def payload_bank():
    rng = np.random.default_rng(11)
    return rng.standard_normal((KEY_COUNT, PAYLOAD_COUNT, VALUE_DIM)).astype(np.float32)


def ill_conditioned_embeddings():
    """Full rank geometry whose condition number sits above the fixed limit."""
    embeddings = np.zeros((3, 8))
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 0] = 1.0
    embeddings[2, 1] = 1e-3
    embeddings[2, 2] = 1e-4
    return embeddings


def events(rows):
    return np.array(rows, dtype=np.int32)


def codes(mode):
    return make_address_codes(key_embeddings(), mode)


class TestAddressCodes:
    def test_codes_are_float32_square_and_unit_normed(self):
        for mode in ADDRESS_MODES:
            query, write = codes(mode)
            assert query.shape == (KEY_COUNT, KEY_COUNT)
            assert write.shape == (KEY_COUNT, KEY_COUNT)
            assert query.dtype == np.float32
            assert write.dtype == np.float32
            np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1.0, rtol=1e-5, atol=1e-6)

    def test_isolated_codes_form_a_dual_basis(self):
        query, write = codes('isolated')
        np.testing.assert_allclose(
            query @ write.T, np.eye(KEY_COUNT), rtol=1e-4, atol=1e-5
        )

    def test_coupled_codes_share_one_matrix(self):
        query, write = codes('coupled')
        np.testing.assert_array_equal(write, query)

    def test_codes_keep_the_cosine_gram_of_the_keys(self):
        embeddings = key_embeddings()
        normalised = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        gram = normalised @ normalised.T
        for mode in ADDRESS_MODES:
            query, _ = codes(mode)
            np.testing.assert_allclose(query @ query.T, gram, rtol=1e-4, atol=1e-5)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            make_address_codes(key_embeddings(), 'diagonal')

    def test_rank_deficient_geometry_is_rejected(self):
        with pytest.raises(ValueError):
            make_address_codes(np.ones((3, 2)), 'isolated')
        duplicated = np.zeros((3, 4))
        duplicated[0, 0] = 1.0
        duplicated[1, 1] = 1.0
        duplicated[2, 0] = 1.0
        with pytest.raises(ValueError):
            make_address_codes(duplicated, 'isolated')

    def test_the_same_geometry_domain_applies_to_both_modes(self):
        for mode in ADDRESS_MODES:
            with pytest.raises(ValueError):
                make_address_codes(ill_conditioned_embeddings(), mode)

    def test_broken_embeddings_are_rejected(self):
        with pytest.raises(ValueError):
            make_address_codes(np.zeros((2, 2)), 'isolated')
        with pytest.raises(ValueError):
            make_address_codes(np.full((2, 2), np.nan), 'isolated')
        with pytest.raises(ValueError):
            make_address_codes(np.zeros(4), 'isolated')


class TestMemoryState:
    def test_initial_memory_is_zeroed_and_unbound(self):
        state = initial_memory(2, KEY_COUNT, VALUE_DIM)
        assert isinstance(state, AddressMemory)
        assert state.matrix.shape == (2, KEY_COUNT, VALUE_DIM)
        assert state.bound.shape == (2, KEY_COUNT)
        assert state.matrix.dtype == jnp.float32
        assert state.bound.dtype == jnp.bool_
        np.testing.assert_array_equal(np.asarray(state.matrix), np.zeros((2, KEY_COUNT, VALUE_DIM)))
        np.testing.assert_array_equal(np.asarray(state.bound), np.zeros((2, KEY_COUNT), bool))

    def test_invalid_sizes_are_rejected(self):
        with pytest.raises(ValueError):
            initial_memory(1, 0, VALUE_DIM)
        with pytest.raises(ValueError):
            initial_memory(1, KEY_COUNT, 0)
        with pytest.raises(TypeError):
            initial_memory(1, KEY_COUNT, 2.5)


class TestDeltaRule:
    def test_literal_write_then_read_returns_the_payload(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[[EVENT_LITERAL, 2, 1]]]),
        )
        values, valid = read_memory(state, query, events([[2]]))
        np.testing.assert_allclose(values[0, 0], payloads[2, 1], rtol=1e-4, atol=1e-5)
        assert bool(valid[0, 0])

    def test_an_unwritten_address_holds_nothing(self):
        query, write = codes('isolated')
        state = initial_memory(1, KEY_COUNT, VALUE_DIM)
        values, valid = read_memory(state, query, events([[0, 3]]))
        np.testing.assert_array_equal(np.asarray(values), np.zeros((1, 2, VALUE_DIM)))
        np.testing.assert_array_equal(np.asarray(valid), np.zeros((1, 2), bool))

    def test_a_later_write_replaces_the_earlier_value(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[[EVENT_LITERAL, 0, 0], [EVENT_LITERAL, 0, 1]]]),
        )
        values, _ = read_memory(state, query, events([[0]]))
        np.testing.assert_allclose(values[0, 0], payloads[0, 1], rtol=1e-4, atol=1e-5)

    def test_blind_copy_keeps_the_value_of_the_instant_of_assignment(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[[EVENT_LITERAL, 0, 0], [EVENT_COPY, 1, 0], [EVENT_LITERAL, 0, 1]]]),
        )
        values, valid = read_memory(state, query, events([[0, 1]]))
        # the source moved on, the copy still holds the value it was given
        np.testing.assert_allclose(values[0, 0], payloads[0, 1], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(values[0, 1], payloads[0, 0], rtol=1e-4, atol=1e-5)
        assert bool(valid[0, 0]) and bool(valid[0, 1])

    def test_isolated_addresses_ignore_an_unrelated_write(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[[EVENT_LITERAL, 0, 0], [EVENT_LITERAL, 2, 1]]]),
        )
        values, _ = read_memory(state, query, events([[0]]))
        np.testing.assert_allclose(values[0, 0], payloads[0, 0], rtol=1e-4, atol=1e-5)

    def test_coupled_addresses_let_an_unrelated_write_through(self):
        query, write = codes('coupled')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[[EVENT_LITERAL, 0, 0], [EVENT_LITERAL, 2, 1]]]),
        )
        values, _ = read_memory(state, query, events([[0]]))
        cosine = float(query[0] @ query[2])
        assert cosine > 0.5  # the keys really are close in feature space
        expected = payloads[0, 0] * (1.0 - cosine**2) + cosine * payloads[2, 1]
        np.testing.assert_allclose(values[0, 0], expected, rtol=1e-4, atol=1e-5)
        assert np.linalg.norm(values[0, 0] - payloads[0, 0]) > 0.1


class TestNoOps:
    def test_a_nop_leaves_its_lane_untouched(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        real = [[EVENT_LITERAL, 0, 0], [EVENT_COPY, 1, 0], [EVENT_LITERAL, 2, 1]]
        quiet = real + [[EVENT_NOP, 3, 2]] * 3
        noisy = []
        for event in real:
            noisy.append([EVENT_NOP, 3, 2])
            noisy.append(event)
        state = apply_events(
            initial_memory(2, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([quiet, noisy]),
        )
        np.testing.assert_array_equal(np.asarray(state.matrix)[0], np.asarray(state.matrix)[1])
        np.testing.assert_array_equal(np.asarray(state.bound)[0], np.asarray(state.bound)[1])

    def test_a_self_copy_changes_nothing(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(2, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([
                [[EVENT_LITERAL, 1, 0], [EVENT_NOP, 0, 0]],
                [[EVENT_LITERAL, 1, 0], [EVENT_COPY, 1, 1]],
            ]),
        )
        np.testing.assert_array_equal(np.asarray(state.matrix)[0], np.asarray(state.matrix)[1])
        np.testing.assert_array_equal(np.asarray(state.bound)[0], np.asarray(state.bound)[1])

    def test_chunked_events_match_a_single_call(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        stream = events([[
            [EVENT_LITERAL, 0, 0],
            [EVENT_COPY, 1, 0],
            [EVENT_NOP, 3, 2],
            [EVENT_LITERAL, 2, 1],
            [EVENT_COPY, 3, 2],
        ]])
        whole = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM), query, write, payloads, stream
        )
        chunked = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM), query, write, payloads, stream[:, :3]
        )
        chunked = apply_events(chunked, query, write, payloads, stream[:, 3:])
        np.testing.assert_allclose(
            np.asarray(chunked.matrix), np.asarray(whole.matrix), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(chunked.bound), np.asarray(whole.bound))


class TestUnbound:
    def test_a_copy_from_an_unbound_source_erases_the_destination(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[
                [EVENT_LITERAL, 1, 0],
                [EVENT_COPY, 2, 1],
                [EVENT_COPY, 2, 3],
            ]]),
        )
        values, valid = read_memory(state, query, events([[1, 2]]))
        assert bool(valid[0, 0])
        assert not bool(valid[0, 1])
        np.testing.assert_array_equal(np.asarray(values[0, 1]), np.zeros(VALUE_DIM))
        np.testing.assert_allclose(values[0, 0], payloads[1, 0], rtol=1e-4, atol=1e-5)

    def test_an_out_of_range_address_reads_as_invalid(self):
        query, write = codes('isolated')
        state = initial_memory(1, KEY_COUNT, VALUE_DIM)
        values, valid = read_memory(state, query, events([[KEY_COUNT + 3, -1]]))
        np.testing.assert_array_equal(np.asarray(values), np.zeros((1, 2, VALUE_DIM)))
        np.testing.assert_array_equal(np.asarray(valid), np.zeros((1, 2), bool))

    def test_garbage_event_indices_never_leak_non_finite_values(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = apply_events(
            initial_memory(1, KEY_COUNT, VALUE_DIM),
            query,
            write,
            payloads,
            events([[
                [EVENT_NOP, -1, 999],
                [EVENT_LITERAL, 0, 0],
                [EVENT_LITERAL, 1, 777],
            ]]),
        )
        assert np.all(np.isfinite(np.asarray(state.matrix)))
        assert bool(np.asarray(state.bound)[0, 0])
        assert not bool(np.asarray(state.bound)[0, 1])


class TestGradients:
    def test_payload_gradients_are_finite_and_reach_the_copied_value(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        stream = events([[[EVENT_LITERAL, 0, 0], [EVENT_COPY, 1, 0]]])
        keys = events([[1]])

        def loss(bank):
            state = apply_events(
                initial_memory(1, KEY_COUNT, VALUE_DIM), query, write, bank, stream
            )
            values, _ = read_memory(state, query, keys)
            return jnp.sum(values * jnp.asarray(payloads[0, 0]))

        gradients = jax.grad(loss)(payloads)
        assert np.all(np.isfinite(np.asarray(gradients)))
        assert np.any(np.asarray(gradients) != 0.0)
        # the copied value is the one the read depends on
        assert np.any(np.asarray(gradients)[0, 0] != 0.0)


class TestInterfaces:
    def test_jit_wrappers_agree_with_the_plain_functions(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        stream = events([[[EVENT_LITERAL, 0, 0], [EVENT_COPY, 1, 0]]])
        state = initial_memory(1, KEY_COUNT, VALUE_DIM)
        plain = apply_events(state, query, write, payloads, stream)
        jitted = apply_events_jit(state, query, write, payloads, stream)
        np.testing.assert_array_equal(np.asarray(plain.matrix), np.asarray(jitted.matrix))
        np.testing.assert_array_equal(np.asarray(plain.bound), np.asarray(jitted.bound))
        keys = events([[0, 1]])
        plain_values, plain_valid = read_memory(plain, query, keys)
        jitted_values, jitted_valid = read_memory_jit(plain, query, keys)
        np.testing.assert_array_equal(np.asarray(plain_values), np.asarray(jitted_values))
        np.testing.assert_array_equal(np.asarray(plain_valid), np.asarray(jitted_valid))

    def test_mismatched_shapes_are_rejected(self):
        query, write = codes('isolated')
        payloads = payload_bank()
        state = initial_memory(1, KEY_COUNT, VALUE_DIM)
        with pytest.raises(ValueError):
            apply_events(state, query, write, payloads[:, :, :1], events([[[EVENT_LITERAL, 0, 0]]]))
        with pytest.raises(ValueError):
            apply_events(state, query, write, payloads, events([[[EVENT_LITERAL, 0]]]))
        with pytest.raises(ValueError):
            read_memory(state, query, events([[0], [1]]))
