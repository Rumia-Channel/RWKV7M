import hashlib
from importlib import import_module
import json
from pathlib import Path
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "experiments/retention/parents/long-s42.safetensors"
VOCAB = ROOT / "data/rwkv_vocab_v20230424.txt"


@pytest.fixture
def readout(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = import_module("symbolic_rwkv_readout")

    def forbidden_native(*args):
        raise AssertionError("incompatible checkpoint reached native model loading")

    monkeypatch.setattr(module, "_native_reader", forbidden_native)
    return module


def relabel(tmp_path, *, vocab_size=None, metadata=None):
    """Alter a real published artifact's labels, retaining all tensor bytes."""
    original = PUBLISHED.read_bytes()
    size = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8:8 + size])
    labels = header["__metadata__"]
    labels.update({
        "tokenizer_format": "rwkv_vocab",
        "tokenizer_vocab_sha256": hashlib.sha256(VOCAB.read_bytes()).hexdigest(),
    })
    if vocab_size is not None:
        config = json.loads(labels["rwkv7m_config_json"])
        config["vocab_size"] = vocab_size
        labels["rwkv7m_config_json"] = json.dumps(config)
        labels["vocab_size"] = str(vocab_size)
    labels.update(metadata or {})
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / "relabelled.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + original[8 + size:])
    return path


@pytest.mark.parametrize("relative", [
    "parents/long-s42.safetensors",
    "trained/adaptive-s42.safetensors",
    "midpoint/midpoint-s42.safetensors",
])
def test_published_kv_checkpoints_are_not_text_models(readout, relative):
    checkpoint = ROOT / "experiments/retention" / relative
    with pytest.raises(ValueError, match="tokenizer_format=rwkv_vocab.*vocab_size=128"):
        readout.make_reader(checkpoint)


def test_rwkv_vocab_label_does_not_rescue_a_128_id_head(readout, tmp_path):
    checkpoint = relabel(tmp_path)
    with pytest.raises(ValueError, match="maximum token ID 65529.*vocab_size=128"):
        readout.make_reader(checkpoint)


def test_padded_text_config_cannot_disguise_actual_kv_head(readout, tmp_path):
    checkpoint = relabel(tmp_path, vocab_size=65536)
    with pytest.raises(ValueError, match="lm_head/kernel.*shape \\[128, 128\\].*expected \\[128, 65536\\]"):
        readout.make_reader(checkpoint)


def test_explicit_vocab_must_match_checkpoint_bytes(readout, tmp_path):
    checkpoint = relabel(tmp_path, vocab_size=65536)
    # Same token IDs and size, but a different byte-to-ID mapping.
    lines = VOCAB.read_bytes().splitlines(keepends=True)
    first, second = lines[:2]
    lines[0] = b"1 " + second.split(b" ", 1)[1]
    lines[1] = b"2 " + first.split(b" ", 1)[1]
    supplied = tmp_path / "different_vocab.txt"
    supplied.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="vocabulary SHA-256 does not match"):
        readout.make_reader(checkpoint, vocab_file=supplied)


@pytest.mark.parametrize("digest", ["", "0" * 64])
def test_project_vocab_requires_actual_hash(readout, tmp_path, digest):
    checkpoint = relabel(tmp_path, vocab_size=65536, metadata={"tokenizer_vocab_sha256": digest})
    with pytest.raises(ValueError, match="vocabulary SHA-256 does not match"):
        readout.make_reader(checkpoint)


def test_matching_explicit_vocab_can_have_padded_ids(readout, tmp_path):
    vocab = tmp_path / "plumbing_only_vocab.txt"
    vocab.write_bytes(b"1 'A' 1\n7 'B' 1\n")
    checkpoint = relabel(tmp_path, metadata={
        "tokenizer_vocab_sha256": hashlib.sha256(vocab.read_bytes()).hexdigest(),
    })
    # Only tests artifact compatibility, not text competence of relabelled weights.
    assert readout._check_checkpoint(checkpoint, vocab) == (checkpoint, vocab)


@pytest.mark.parametrize("length,expected", [
    (1, [1]), (63, [32, 16, 8, 4, 2, 1]), (64, [64]),
    (65, [64, 1]), (191, [64, 64, 32, 16, 8, 4, 2, 1]),
    (1024, [64] * 16),
])
def test_prefill_chunks_never_clip_or_pad(readout, length, expected):
    sizes = list(readout._chunk_sizes(length))
    assert sizes == expected
    assert sum(sizes) == length


def test_answer_bytes_are_decoded_only_after_tokens_are_joined(readout):
    assert readout._prediction(bytearray(" 3+2i ", "utf-8")) == ("3+2i", None)
    assert readout._prediction(bytearray("数", "utf-8")) == ("数", None)
    assert readout._prediction(bytearray(b"\xe6\x95")) == ("", "invalid_utf8")


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_output_budget_is_positive_integer(readout, budget):
    with pytest.raises(ValueError, match="positive integer"):
        readout.make_reader(PUBLISHED, max_new_tokens=budget)
