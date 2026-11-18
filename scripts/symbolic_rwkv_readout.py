"""Blind, greedy first-line readout of portable RWKV text checkpoints.

Native dependencies are imported only after the checkpoint/vocabulary contract
passes. This module does not import the symbolic generator, solver, or answers.
"""

import hashlib
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _check_checkpoint(checkpoint, vocab_file):
    path = Path(checkpoint)
    if path.is_dir():
        path /= "model.safetensors"
    if path.suffix != ".safetensors":
        raise ValueError("text readout requires a portable .safetensors checkpoint")
    # The format reader validates offsets and exposes shapes without loading weights or JAX.
    from safetensors import safe_open
    with safe_open(path, framework="numpy") as weights:
        metadata = weights.metadata() or {}
        names = weights.keys()
        shapes = {
            name: weights.get_slice(name).get_shape()
            for name in ("lm_head/kernel", "token_embedding/embedding") if name in names
        }
    config_json = metadata.get("rwkv7m_config_json")
    if not config_json:
        raise ValueError("checkpoint is missing rwkv7m config metadata")
    config = json.loads(config_json)
    vocab_size = int(config["vocab_size"])
    tokenizer_format = metadata.get("tokenizer_format")
    if tokenizer_format != "rwkv_vocab":
        raise ValueError(
            "text readout requires tokenizer_format=rwkv_vocab; "
            f"got {tokenizer_format!r} with vocab_size={vocab_size}. "
            "KV-only integer-ID checkpoints cannot answer text prompts."
        )
    vocab = Path(vocab_file) if vocab_file is not None else _ROOT / "data/rwkv_vocab_v20230424.txt"
    data = vocab.read_bytes()
    actual_hash = hashlib.sha256(data).hexdigest()
    if metadata.get("tokenizer_vocab_sha256") != actual_hash:
        raise ValueError(f"checkpoint tokenizer vocabulary SHA-256 does not match {vocab}")
    token_ids = [int(line.split(" ", 1)[0]) for line in data.decode("utf-8").splitlines()]
    if not token_ids or min(token_ids) < 0:
        raise ValueError("text vocabulary must contain nonnegative token IDs")
    maximum = max(token_ids)
    if maximum >= vocab_size:
        raise ValueError(
            f"text vocabulary maximum token ID {maximum} does not fit config "
            f"vocab_size={vocab_size}; a mislabeled KV-only/too-small text head is not compatible"
        )
    d_model = int(config["d_model"])
    for name, expected in (
        ("lm_head/kernel", [d_model, vocab_size]),
        ("token_embedding/embedding", [vocab_size, d_model]),
    ):
        actual = shapes.get(name)
        if actual != expected:
            raise ValueError(
                f"incompatible text tensor {name}: shape {actual}, expected {expected}; "
                "changing metadata does not make a KV-only head text-compatible"
            )
    return path, vocab


def _chunk_sizes(length):
    """Bound prefill at 64 tokens, using only power-of-two remainder shapes."""
    while length:
        size = 64 if length >= 64 else 1 << (length.bit_length() - 1)
        yield size
        length -= size


def _prediction(data):
    try:
        return data.decode("utf-8").strip(), None
    except UnicodeDecodeError:
        return "", "invalid_utf8"


def _native_reader(checkpoint, vocab_file, max_new_tokens):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from rwkv7m.io import load_model_safetensors
    from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
    from rwkv7m.model.nnx_model import initialize_nnx_model
    from rwkv7m.model.state import init_rwkv_state, init_screen_state
    from rwkv7m.tokenizer import RWKVTokenizer

    params, config, _ = load_model_safetensors(checkpoint)
    model = initialize_nnx_model(jax.random.key(0), config)
    load_linen_params_into_nnx(model, params)
    tokenizer = RWKVTokenizer(vocab_file)
    initial = init_rwkv_state(1, config), init_screen_state(1, config.screening)

    @nnx.jit
    def prefill(model, ids, rwkv, screen):
        _, rwkv, screen, _ = model.compute_recurrent_hidden(
            ids, rwkv, screen, phase="read_write", deterministic=True,
        )
        return rwkv, screen

    @nnx.jit
    def step(model, ids, rwkv, screen):
        hidden, rwkv, screen, _ = model.compute_recurrent_hidden(
            ids, rwkv, screen, phase="read_write", deterministic=True,
        )
        # Never create [prompt_length, vocab_size] logits.
        logits = model.compute_logits(hidden[:, -1:, :])[0, 0].astype(jnp.float32)
        return jnp.argmax(logits), jnp.all(jnp.isfinite(logits)), rwkv, screen

    def reader(prompt):
        if not isinstance(prompt, str):
            raise TypeError("reader accepts a prompt string only")
        ids = tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt must encode at least one token")
        # JAX states are immutable; every independent row starts at these zeros.
        rwkv, screen = initial
        offset = 0
        for size in _chunk_sizes(len(ids)):
            chunk = jnp.asarray([ids[offset:offset + size]], dtype=jnp.int32)
            offset += size
            if offset < len(ids):
                rwkv, screen = prefill(model, chunk, rwkv, screen)
            else:
                token, finite, rwkv, screen = step(model, chunk, rwkv, screen)
        generated = 0
        text = bytearray()

        def result(prediction, stop):
            return {
                "prediction": prediction,
                "prompt_tokens": len(ids),
                "generated_tokens": generated,
                "stop": stop,
            }

        for _ in range(max_new_tokens):
            if not bool(finite):
                return result("", "nonfinite_logits")
            token_id = int(token)
            generated += 1
            if token_id == 0:  # EOS is not a vocabulary byte string.
                prediction, error = _prediction(text)
                return result(prediction, error or "eos")
            piece = tokenizer.idx2token.get(token_id)
            if piece is None:
                return result("", "invalid_token")
            # Stop inside a token if necessary; do not scan the accumulated answer on every token.
            ends = [pos for pos in (piece.find(b"\n"), piece.find(b"\r")) if pos >= 0]
            if ends:
                text.extend(piece[:min(ends)])
                prediction, error = _prediction(text)
                return result(prediction, error or "newline")
            text.extend(piece)
            if generated < max_new_tokens:
                token, finite, rwkv, screen = step(
                    model, jnp.asarray([[token_id]], dtype=jnp.int32), rwkv, screen,
                )
        # An unterminated prefix is never promoted to a correct answer.
        return result("", "max_new_tokens")

    return reader


def make_reader(checkpoint, *, vocab_file=None, max_new_tokens=64):
    """Return reader(prompt) with prediction, token counts, and a stop reason.

    Newline or EOS terminates the first generated answer line. Invalid IDs,
    nonfinite logits, invalid UTF-8, and exhausted budgets yield an empty answer.
    No answer/private-key input is accepted. Token counts include sampled EOS.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    checkpoint, vocab = _check_checkpoint(checkpoint, vocab_file)
    return _native_reader(checkpoint, vocab, max_new_tokens)
